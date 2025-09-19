"""
Tournament Selection implementation for MalthusJAX.

Tournament selection chooses individuals by running tournaments between
randomly selected candidates and picking the winner (highest fitness).
"""
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from typing import Optional # type: ignore

from malthusjax.core.base import Compatibility, ProblemTypes
from malthusjax.core.population.population import Population
from malthusjax.operators.selection.base import AbstractSelectionOperator

from typing import Callable, Generic, TypeVar

P = TypeVar('P', bound='Population')

class TournamentSelection(AbstractSelectionOperator[P], Generic[P]):
    """Tournament selection operator with JAX JIT compilation.

    This operator selects individuals based on tournament selection.
    """

    def __init__(self, number_of_tournaments: int = 10, tournament_size: int = 4) -> None:
        #self._init_kwargs = {'number_of_tournaments': number_of_tournaments, 'tournament_size': tournament_size}
        self.number_of_tournaments = number_of_tournaments
        self.tournament_size = tournament_size
        super().__init__()


    def _create_selection_function(self) -> Callable:
        """Create the core tournament selection function to be vectorized and JIT-compiled.
        
        Returns:
            Function with signature (genome_data, random_key, selection_rate) -> selected_genome_data
        """

        number_of_tournaments = self.number_of_tournaments
        tournament_size = self.tournament_size
        @jax.jit
        def tournament_selection(fitness_scores, key):
            """Tournament selection function that will be JIT compiled.
            
            Args:
                fitness_scores: Array of fitness values for each individual
                key: JAX random key for reproducibility
                
            Returns:
                Array of indices of tournament winners
            """
            # Generate random tournament indices
            tournament_indices = jax.random.randint(
                key, 
                (number_of_tournaments, tournament_size), 
                0, 
                fitness_scores.shape[0]
            )
            
            # Get fitness values for tournament participants
            fitness_matrix = jnp.take(fitness_scores, tournament_indices)
            
            # Find winners (indices with maximum fitness in each tournament)
            local_winners = jnp.argmax(fitness_matrix, axis=1)
            
            # Convert local indices back to global population indices
            tournament_winners = tournament_indices[jnp.arange(number_of_tournaments), local_winners]

            return tournament_winners
        return tournament_selection
        