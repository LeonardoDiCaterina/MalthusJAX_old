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
    
    @jit
    def taylor_series(x):
        return jnp.sum(coefficients * x**jnp.arange(len(coefficients)))
    
    return taylor_series




class TaylorSeriesFitnessEvaluator(AbstractFitnessEvaluator):


    def __init__(self, target_function: Callable, x_values: jnp.ndarray) -> None:
        self.target_function = jit(target_function)
        self.x_values = x_values
        super().__init__()

    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        """
        Tensor-only version of Taylor series fitness function.
        
        Args:
            genome_tensor: JAX array representing genome
            
        Returns:
            Fitness value as float
        """
        taylor_func = taylor_series_from_coefficients(genome_tensor)
        print("Evaluating fitness for genome tensor:", genome_tensor)
        y_pred = vmap(taylor_func)(self.x_values)
        y_true = self.target_function(self.x_values)
        mse = jnp.mean((y_true - y_pred) ** 2)
        return mse


