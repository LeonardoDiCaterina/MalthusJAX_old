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
        self._compiled_fn: Optional[Callable] = self._create_mutation_function()

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
            TypeError: If population is not an instance of AbstractPopulation.
            RuntimeError: If the mutation function cannot be built due to invalid population.    
            
        """
        if len(population) == 0:
            raise ValueError("Cannot build mutation operator for empty population")
        

        if not isinstance(population, AbstractPopulation):
            raise TypeError("Population must be an instance of AbstractPopulation")

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
        num_solutions = len(population)
        subkeys = jax.random.split(random_key, num_solutions)
        
        # Get all solutions and their genome data
        genome_data = population.to_stack()
        
        # Apply the compiled mutation function - this is where the JIT magic happens
        mutated_genome_data = mutation_fn(genome_data, subkeys)

        # Create a new population from the mutated genome data
        new_population = population.from_stack(mutated_genome_data)
        
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
    
    def get_batched_function(self) -> Callable:
        """Get the batched (vectorized) function before JIT compilation.
        
        Returns:
            The batched function created during build().

        Raises:
            ValueError: If operator hasn't been built yet.
        """
        return jax.jit(jax.vmap(
            lambda genome, key: self._create_mutation_function()(genome, key, self.mutation_rate),
            in_axes=(0, 0)
        ))
        
    