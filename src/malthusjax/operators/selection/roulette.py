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

class RouletteSelection(AbstractSelectionOperator[P], Generic[P]):
    """Roulette selection operator with JAX JIT compilation.

    This operator selects individuals based on their fitness proportion.
    """

    def __init__(self, number_choices: int) -> None:
        self.number_choices = number_choices
        #self._init_kwargs = {'number_choices': number_choices}
        super().__init__()


    def _create_selection_function(self) -> Callable:
        """Create the core roulette selection function to be vectorized and JIT-compiled.
        Returns:
            Function with signature (genome_data, random_key, selection_rate) -> selected_genome_data
        """
        
        number_choices = self.number_choices
        
        @jax.jit
        def roulette_selection(fitness_scores, key):
            """Roulette selection function that will be JIT compiled.

            Args:
            fitness_scores: Array of fitness values for each individual
            key: JAX random key for reproducibility
            
            Returns:
            Array of indices of selected individuals    
            """
            # Calculate probabilities based on fitness scores
            probabilities = fitness_scores / jnp.sum(fitness_scores)    
            # Choose random indices based on the probabilities
            selected_indices = jax.random.choice(
            key, 
            jnp.arange(fitness_scores.shape[0]), 
            shape=(number_choices,), 
            p=probabilities
            )
            return selected_indices     
        return roulette_selection

