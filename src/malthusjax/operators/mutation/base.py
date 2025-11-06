"""
Base classes for mutation operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Tuple

import jax  # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from functools import partial # type: ignore
import functools

from malthusjax.operators.base import AbstractGeneticOperator

class AbstractMutation(AbstractGeneticOperator, ABC):
    """Abstract base class for mutation operators.
    
    Mutation operators return a pure function with the signature:
    (key: PRNGKey, genome: jax.Array) -> mutated_genome: jax.Array
    """

    def __init__(self, mutation_rate: float , n_outputs: int = 1) -> None:
        """
        Initialize mutation operator.
        
        Args:
            mutation_rate: Probability of mutation (behavior depends on operator).
        """
        super().__init__()
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
        self.n_outputs = n_outputs
        
    @abstractmethod
    def get_compiled_function(self) -> Callable:
        """
        Returns a JIT-compiled function for performing mutation.
        
        The function will have the signature:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """