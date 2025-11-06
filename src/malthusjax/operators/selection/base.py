"""
Base classes for selection operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Callable
from malthusjax.operators.base import AbstractGeneticOperator
import jax # type: ignore

class AbstractSelectionOperator(AbstractGeneticOperator, ABC):
    """Abstract base class for selection operators.
    
    Selection operators return a pure function with the signature:
    (key: PRNGKey, fitness_values: jax.Array) -> selected_indices: jax.Array
    """

    def __init__(self, number_of_choices: int) -> None:
        """
        Initialize the selection operator.
        
        Args:
            number_of_choices: The number of individuals to select (e.g., population size).
        """
        super().__init__()
        self.number_of_choices = number_of_choices

    @abstractmethod
    def get_compiled_function(self) -> Callable:
        """
        Returns a JIT-compiled function for performing selection.
        
        The function will have the signature:
        (key: jax.Array, fitness_values: jax.Array) -> selected_indices: jax.Array
        """
        pass