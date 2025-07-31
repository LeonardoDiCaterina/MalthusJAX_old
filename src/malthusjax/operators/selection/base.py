"""
Base classes for selection operators with optimized JAX JIT support.
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

class AbstractSelectionOperator(AbstractGeneticOperator[P], Generic[P]):
    """Abstract base class for selection operators with optimized JAX JIT compilation.

    Selection operators build and return JIT-compiled functions that operate directly
    on genome data arrays for maximum performance.
    """

    def __init__(self) -> None:
        """Initialize selection operator with selection rate.
        
        Args:
            selection_rate: Probability of selection for each element.
        """
        super().__init__()

    def build(self, population: P) -> Callable:
        """Build the selection operator and return JIT-compiled function.

        Args:
            population: Population to build the operator for (used for shape inference).
            
        Returns:
            JIT-compiled function with signature:
            (genome_data_array, random_keys_array, selection_rate) -> selected_genome_data_array

        example:
            ```python
            selection_fn = selection_operator.build(population)
            selected_genomes = selection_fn(genome_data, random_keys)
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

        # Create the core selection function
        core_fn = self._create_selection_function(pop_size=population.size)

        self._compiled_fn = jax.jit(core_fn)
        self.built = True
        
        return self._compiled_fn
    
    
    @abstractmethod
    def _create_selection_function(self) -> Callable:
        """Create the core selection function to be vectorized and JIT-compiled.

        This function should be pure and operate on a single genome.
        
        Returns:
            Function with signature (genome_data, random_key, selection_rate) -> selected_genome_data
        """
        pass
    
    def call(self, population: P, random_key: jax.Array, **kwargs) -> P:
        """Apply selection to the population using the compiled function.
        
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
        
        fitness_values = population.get_fitness_values()
        selected_indices = mutation_fn(fitness_values, random_key)
        
        population

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