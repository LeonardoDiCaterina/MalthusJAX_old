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
        
    def build(self, population: P) -> Callable:
        """Build the corssover operator and return JIT-compiled function.
        
        Args:
            population: Population to build the operator for (used for shape inference).
            
        Returns:
            JIT-compiled function with signature:
            (genome_data_array, random_keys_array, crossover_rate) -> crossed_genome_data_array
        """
        if population.size == 0:
            raise ValueError("Cannot build crossover operator for empty population")
        
        if not hasattr(population, 'validate'):
            raise ValueError(f"Type {type(population)} must have a validate method")
        population.validate()
        
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
    
    def call(self, population_or_stack: P | jax.Array, random_key: jax.Array, **kwargs) -> P | jax.Array:
        """Apply crossover to the population or stack using the compiled function.

        Args:
            population_or_stack: Input population or stack of genomes to cross.
            random_key: JAX random key for reproducibility.

        Returns:
            New population or crossed stack.
        """
        # Determine if input is a population or a stack
        is_population = hasattr(population_or_stack, "to_stack") and hasattr(population_or_stack, "from_stack")

        if is_population:
            population = population_or_stack
            if not self.built:
                self.build(population)
            crossover_fn = self.get_compiled_function()
            num_solutions = len(population)
            subkeys = jax.random.split(random_key, num_solutions)
            genome_data = population.to_stack()
            crossed_genome_data = crossover_fn(genome_data, subkeys)
            new_population = population.from_stack(crossed_genome_data)
            return new_population
        else:
            genome_data = population_or_stack
            num_solutions = genome_data.shape[0]
            subkeys = jax.random.split(random_key, num_solutions)
            if not self.built:
                raise RuntimeError("Crossover operator must be built with a population before calling with a stack.")
            crossover_fn = self.get_compiled_function()
            crossed_genome_data = crossover_fn(genome_data, subkeys)
            return crossed_genome_data