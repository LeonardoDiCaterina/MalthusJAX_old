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

class AbstractCrossover(AbstractGeneticOperator[P], Generic[P]):
    """Abstract base class for crossover operators with optimized JAX JIT compilation.
    
    Crossover operators build and return JIT-compiled functions that operate directly
    on genome data arrays for maximum performance.
    """
    
    def __init__(self, crossover_rate: float = 0.01) -> None:
        """Initialize crossover operator with crossover rate.
        
        Args:
            crossover_rate: Probability of crossover for each element.
        """
        super().__init__()
        if not (0 <= crossover_rate <= 1):
            raise ValueError("Crossover rate must be between 0 and 1")
        self.crossover_rate = crossover_rate
        self._compiled_fn: Optional[Callable] = self._create_crossover_function()
    def build(self, population: P) -> Callable:
        """Build the corssover operator and return JIT-compiled function.
        
        Args:
            population: Population to build the operator for (used for shape inference).
            
        Returns:
            JIT-compiled function with signature:
            (genome_data_array, random_keys_array, crossover_rate) -> crossed_genome_data_array
        """
        if len(population) == 0:
            raise ValueError("Cannot build crossover operator for empty population")
        
        
        # Create the core crossover function
        core_fn = self._create_crossover_function()
        
        # Create vectorized version for batch processing
        vectorized_fn = jax.vmap(core_fn, in_axes=(0, 0))

        # JIT compile the vectorized function
        self._compiled_fn = jax.jit(vectorized_fn)
        self.built = True
        
        return self._compiled_fn
    
    @abstractmethod
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function to be vectorized and JIT-compiled.
        
        This function should be pure and operate on a single genome.
        
        Returns:
            Function with signature (genome_data, random_key, crossover_rate) -> crossed_genome_data
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
        crossover_fn = self.get_batched_function()

        # Create subkeys for each solution
        num_solutions = len(population)
        subkeys = jax.random.split(random_key, num_solutions)
        
        # Get all solutions and their genome data
        genome_data = population.to_stack()
        
        # Apply the compiled crossover function - this is where the JIT magic happens
        crossed_genome_data = crossover_fn(genome_data, subkeys)

        # Create a new population from the mutated genome data
        new_population = population.from_stack(crossed_genome_data)
        
        return new_population

    def get_batched_function(self) -> Callable:
        """Get the batched (vectorized) function before JIT compilation.
        
        Returns:
            The batched function created during build().

        """
        crossover_fn = self.get_compiled_function()
        # Vectorized crossover function
        def vectorized_crossover(pop1_stack: jax.Array, random_keys: jax.Array) -> jax.Array:
            #shuffle the second population to create pairs
            pop2_stack = jax.random.permutation(random_keys[0], pop1_stack, axis=0)
            return jax.vmap(crossover_fn)(pop1_stack, pop2_stack, random_keys)

        return vectorized_crossover
