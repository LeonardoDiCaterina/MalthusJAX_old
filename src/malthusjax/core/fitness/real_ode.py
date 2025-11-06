from functools import partial
from typing import Any

from traitlets import Callable
from malthusjax.core.base import Compatibility
from jax import Array # type: ignore
import jax.numpy as jnp # type: ignore
import jax # type: ignore
from jax import jit, vmap # type: ignore
from .base import AbstractFitnessEvaluator




def taylor_series_from_coefficients(coefficients: jnp.ndarray):
    """
    Create a function that computes the Taylor series expansion from given coefficients.
    
    Args:
        coefficients (jax.numpy.ndarray): Coefficients of the Taylor series.
        
    Returns:
        function: A function that computes the Taylor series expansion.
    """
    
    def __init__(self, target_function: Callable, x_values: jnp.ndarray) -> None:
        self.target_function = jit(target_function)
        self.x_values = x_values
        super().__init__()

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:
        
        raise NotImplementedError("This method should be implemented by subclasses.")
        pass





class TaylorSeriesFitnessEvaluator(AbstractFitnessEvaluator):


    def __init__(self, target_function: Callable, x_values: jnp.ndarray) -> None:
        self.target_function = jit(target_function)
        self.x_values = x_values
        super().__init__()

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:
        raise NotImplementedError("This method should be implemented by subclasses.")
        pass

