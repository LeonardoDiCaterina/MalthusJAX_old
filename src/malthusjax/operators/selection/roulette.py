"""
Roulette Wheel Selection implementation for MalthusJAX.
"""
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from typing import Callable
from malthusjax.operators.selection.base import AbstractSelectionOperator
import functools

class RouletteSelection(AbstractSelectionOperator):
    """
    Selects individuals using roulette wheel (fitness-proportionate) selection.
    Assumes fitness values are non-negative.
    """

    def __init__(self, number_choices: int) -> None:
        super().__init__(number_of_choices=number_choices)

    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for roulette wheel selection.
        """
        # Bake static parameters into the pure function
        return functools.partial(
            _roulette_selection,
            number_of_choices=self.number_of_choices
        )

# --- Pure JAX Function ---

@jax.jit
def _roulette_selection(
    key: jax.Array,
    fitness_values: jax.Array,
    number_of_choices: int
) -> jax.Array:
    """
    Pure JAX function for roulette wheel selection.
    
    Args:
        key: PRNGKey
        fitness_values: 1D array of non-negative fitnesses.
        number_of_choices: Static int. Total individuals to select.
        
    Returns:
        1D array of indices for the selected individuals.
    """
    # 1. Calculate probabilities
    fitness_sum = jnp.sum(fitness_values)
    # Add epsilon to avoid division by zero if all fitnesses are 0
    probabilities = (fitness_values + 1e-6) / (fitness_sum + 1e-6)
    
    # 2. Choose indices based on probabilities
    selected_indices = jax.random.choice(
        key,
        jnp.arange(fitness_values.shape[0]),
        shape=(number_of_choices,),
        p=probabilities
    )
    return selected_indices