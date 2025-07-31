# File: src/malthusjax/operators/mutation/base.py
"""
Base classes for mutation operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Tuple

import jax  # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from functools import partial # type: ignore

from malthusjax.core.population.base import AbstractPopulation
from malthusjax.operators.base import AbstractGeneticOperator

P = TypeVar('P', bound=AbstractPopulation)

class AbstractMutation(AbstractGeneticOperator[P], Generic[P]):
    """Abstract base class for mutation operators with optimized JAX JIT compilation.
    
    Mutation operators build and return JIT-compiled functions that operate directly
    on genome data arrays for maximum performance.
    """
    
    def __init__(self, mutation_rate: float = 0.01) -> None:
        """Initialize mutation operator with mutation rate.
        
        Args:
            mutation_rate: Probability of mutation for each element.
        """
        super().__init__()
        if not (0 <= mutation_rate <= 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        self.mutation_rate = mutation_rate
        
    def build(self, population: P) -> Callable:
        """Build the mutation operator and return JIT-compiled function.
        
        Args:
            population: Population to build the operator for (used for shape inference).
            
        Returns:
            JIT-compiled function with signature:
            (genome_data_array, random_keys_array, mutation_rate) -> mutated_genome_data_array
            
        example:
            ```python
            mutation_fn = mutation_operator.build(population)
            mutated_genomes = mutation_fn(genome_data, random_keys)
            ```
            
        Raises:
            ValueError: If population is empty or does not have a validate method.
            TypeError: If population is not an instance of AbstractPopulation.
            RuntimeError: If the mutation function cannot be built due to invalid population.    
            
        """
        if population.size == 0:
            raise ValueError("Cannot build mutation operator for empty population")
        
        if not hasattr(population, 'validate'):
            raise ValueError(f"Type {type(population)} must have a validate method")
        population.validate()
        
        # Create the core mutation function
        core_fn = self._create_mutation_function()
        
        # Create vectorized version for batch processing
        vectorized_fn = jax.vmap(
            lambda genome, key: core_fn(genome, key, self.mutation_rate),
            in_axes=(0, 0)
        )
        
        # JIT compile the vectorized function
        self._compiled_fn = jax.jit(vectorized_fn)
        self.built = True
        
        return self._compiled_fn
    
    
    @abstractmethod
    def _create_mutation_function(self) -> Callable:
        """Create the core mutation function to be vectorized and JIT-compiled.
        
        This function should be pure and operate on a single genome.
        
        Returns:
            Function with signature (genome_data, random_key, mutation_rate) -> mutated_genome_data
        """
        pass
    
    def call(self, population: P, random_key: jax.Array, **kwargs) -> P:
        """Apply mutation to the population using the compiled function.
        
        Args:
            population: Input population to mutate.
            random_key: JAX random key for reproducibility.
            
        Returns:
            New population with mutated individuals.
        """
        if not self.built:
            self.build(population)
        
        # Get the compiled function
        mutation_fn = self.get_compiled_function()
        
        # Create subkeys for each solution
        num_solutions = population.size
        subkeys = jax.random.split(random_key, num_solutions)
        
        # Get all solutions and their genome data
        solutions = population.get_solutions()
        genomes = [solution.genome for solution in solutions]
        genome_data = jnp.stack([genome.to_tensor() for genome in genomes])
        
        # Apply the compiled mutation function - this is where the JIT magic happens
        mutated_genome_data = mutation_fn(genome_data, subkeys)
        
        # Create a new population with the same parameters
        new_population = population.__class__(
            solution_class=population.solution_class,
            max_size=population.max_size,
            random_init=False,
            random_key=None
        )
        
        # Create new solutions with mutated genomes
        for i, solution in enumerate(solutions):
            new_genome = solution.genome.__class__.from_tensor(
                mutated_genome_data[i],
            )
            
            new_solution = solution.clone()
            new_solution.genome = new_genome
            new_population.add_solution(new_solution)
        
        return new_population

    def get_compiled_function(self) -> Callable:
        """Get the compiled function after building.
        
        Returns:
            The compiled function returned by build().
            
        Raises:
            ValueError: If operator hasn't been built yet.
        """
        if not self.built or self._compiled_fn is None:
            raise ValueError("Operator not built. Call build() first.")
        return self._compiled_fn