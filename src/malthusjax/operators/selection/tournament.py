"""
Tournament Selection implementation for MalthusJAX.
"""
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from typing import Optional, Callable # type: ignore

from malthusjax.operators.selection.base import AbstractSelectionOperator
import functools

class TournamentSelection(AbstractSelectionOperator):
    """
    Selects individuals using tournament selection.
    """

    def __init__(self, number_of_choices: int, tournament_size: int = 4) -> None:
        """
        Args:
            number_of_choices: The total number of individuals to select.
            tournament_size: The number of individuals competing in each tournament.
        """
        super().__init__(number_of_choices=number_of_choices)
        self.tournament_size = tournament_size

    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for tournament selection.
        """
        # Bake static parameters into the pure function
        return functools.partial(
            _tournament_selection,
            number_of_choices=self.number_of_choices,
            tournament_size=self.tournament_size
        )

# --- Pure JAX Function ---

def _tournament_selection(
    key: jax.Array,
    fitness_values: jax.Array,
    number_of_choices: int,
    tournament_size: int
) -> jax.Array:
    """
    Pure JAX function for tournament selection.
    
    Args:
        key: PRNGKey
        fitness_values: 1D array of fitnesses for the entire population.
        number_of_choices: Static int. Total number of winners to select.
        tournament_size: Static int. Participants per tournament.
        
    Returns:
        1D array of indices for the selected individuals.
    """
    population_size = fitness_values.shape[0]
    
    # Create keys for each choice
    keys = jar.split(key, number_of_choices)
    
    # vmap the selection of a single winner
    return jax.vmap(
        _select_one_winner,
        in_axes=(0, None, None, None)
    )(keys, fitness_values, population_size, tournament_size)

def _select_one_winner(
    key: jax.Array,
    fitness_values: jax.Array,
    population_size: int,
    tournament_size: int
) -> int:
    """Selects a single winner from one tournament."""
    # 1. Randomly pick tournament participants
    participant_indices = jar.randint(
        key,
        shape=(tournament_size,),
        minval=0,
        maxval=population_size
    )
    
    # 2. Get their fitness values
    participant_fitnesses = fitness_values[participant_indices]
    
    # 3. Find the index of the winner *within the tournament*
    winner_local_index = jnp.argmax(participant_fitnesses)
    
    # 4. Return the global index of the winner
    return participant_indices[winner_local_index]