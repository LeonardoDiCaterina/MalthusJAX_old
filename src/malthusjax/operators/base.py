"""
Abstract base classes for genetic layers in MalthusJAX.

This module defines the layer abstractions that enable Keras-like composition
of genetic operations for evolutionary algorithms.
"""
# Path: src/malthusjax/operators/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Union

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore

from malthusjax.core.population.base import AbstractPopulation

P = TypeVar('P', bound=AbstractPopulation)

class AbstractGeneticOperator(ABC, Generic[P]):
    """Abstract base class for genetic operators in MalthusJAX.
    
    All genetic operators follow a pattern where build() returns a (potentially JIT-compiled) 
    function that can be applied to genome data arrays directly.
    """
    
    def __init__(self) -> None:
        """Initialize the genetic operator."""
        self.built = True

    def build(self, population: P) -> Callable:
        """Build the operator and return the compiled function.
        
        Args:
            population: Population to build the operator for (used for shape/type inference).
            
        Returns:
            Compiled function that operates on genome data arrays.
            Function signature depends on operator type:
            - Selection: (genome_data, fitness_values, random_key, **params) -> selected_indices
            - Crossover: (parent1_data, parent2_data, random_key, **params) -> (child1_data, child2_data)
            - Mutation: (genome_data, random_key, **params) -> mutated_genome_data
        """
        self.built = True
        return self._compiled_fn
    
    def get_compiled_function(self) -> Callable:
        """Get the compiled function after building.
        
        Returns:
            The compiled function returned by build().
            
        Raises:
            ValueError: If operator hasn't been built yet.
        """
        return self._compiled_fn
    
    @abstractmethod 
    def call(self, population: P, random_key: jax.Array, fitness_values: Optional[jax.Array] = None, **kwargs) -> P:
        """Apply the genetic operation to the population using the compiled function.
        
        This method handles the population-level orchestration while delegating
        the core computation to the compiled function from build().
        
        Args:
            population: Input population to apply the operation to.
            random_key: JAX random key for reproducibility.
            **kwargs: Additional operator-specific parameters.
            
        Returns:
            New population after applying the genetic operation.
        """
        pass

    def __call__(self, population: P, random_key: Optional[jax.Array] = None, fitness_values: Optional[jax.Array] = None, **kwargs) -> P:
        """Apply the operator to a population.
        
        Args:
            population: Input population to apply the operation to.
            random_key: JAX random key for reproducibility.
            **kwargs: Additional operator-specific parameters.
            
        Returns:
            New population after applying the genetic operation.
        """
        if not self.built:
            self.build(population)
            
        if random_key is None:
            random_key = jr.PRNGKey(0)
            
        return self.call(
            population=population,
            random_key=random_key,
            fitness_values=fitness_values,
            **kwargs
        )