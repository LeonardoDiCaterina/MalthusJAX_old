"""
Base classes for selection operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from malthusjax.operators.base import AbstractGeneticOperator
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

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
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for performing selection.
        
        The function will have the signature:
        (key: jax.Array, fitness_values: jax.Array) -> selected_indices: jax.Array
        """
        pass
    
    def verify_signature(self, test_fitness_shape: Tuple[int, ...] = (10,)) -> bool:
        """
        Verify that the compiled function follows the correct signature.
        
        Args:
            test_fitness_shape: Shape of test fitness array for validation.
            
        Returns:
            True if signature is correct, False otherwise.
            
        Raises:
            Exception with details if the signature test fails.
        """ 
        
        try:
            # Create test data
            test_key = jax.random.PRNGKey(0)
            test_fitness = jax.numpy.ones(test_fitness_shape, dtype=jax.numpy.float32)
            
            # Get the pure function
            selection_fn = self.get_pure_function()
            compiled_fn = jax.jit(selection_fn)
            
            # Call the function
            selected_indices = compiled_fn(test_key, test_fitness)
            
            # Check output shape
            expected_shape = (self.number_of_choices,)
            if selected_indices.shape != expected_shape:
                raise ValueError(f"Expected output shape {expected_shape}, got {selected_indices.shape}")
            
            return True
        except Exception as e:
            raise RuntimeError(f"Signature verification failed: {e}")