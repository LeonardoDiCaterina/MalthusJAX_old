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
        super().__init__()
        self.number_of_tournaments = number_of_tournaments
        self.tournament_size = tournament_size
    
    def _create_selection_function(self, pop_size:int) -> Callable:
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
                pop_size
            )
            
            # Get fitness values for tournament participants
            fitness_matrix = jnp.take(fitness_scores, tournament_indices)
            
            # Find winners (indices with maximum fitness in each tournament)
            local_winners = jnp.argmax(fitness_matrix, axis=1)
            
            # Convert local indices back to global population indices
            tournament_winners = tournament_indices[jnp.arange(number_of_tournaments), local_winners]

            return tournament_winners
        return tournament_selection
    
    def call(self, population: P, random_key: jax.Array, **kwargs) -> P:
        """Apply tournament selection to the population using the compiled function.
        
        Args:
            population: Input population to select from.
            random_key: JAX random key for reproducibility.
            
        Returns:
            New population with selected individuals.
        """
        if not self.built:
            self.build(population)
        
        # Get the compiled function
        selection_fn = self.get_compiled_function()
        
        # Create subkeys for each solution        
        # Evaluate fitness scores for the genomes
        fitness_scores = population.get_fitness_values()

        # Apply the compiled selection function - this is where the JIT magic happens
        tournament_winners_indices = selection_fn(fitness_scores, random_key)

        # Select the winning genomes using the indices
        winning_genomes = [population[i] for i in tournament_winners_indices]


        # Create a new population with the selected genomes
        new_population = population.from_solution_list(winning_genomes)

        return new_population
        