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

    def __init__(self, number_choices=10) -> None:
        super().__init__()
        self.number_choices = number_choices

    def _create_selection_function(self, pop_size:int) -> Callable:
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
            jnp.arange(pop_size), 
            shape=(number_choices,), 
            p=probabilities
            )
            return selected_indices     
        return roulette_selection

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
                
        # Evaluate fitness scores for the genomes
        fitness_scores = population.get_fitness_values()

        # Apply the compiled selection function - this is where the JIT magic happens
        tournament_winners_indices = selection_fn(fitness_scores, random_key)

        # Select the winning genomes using the indices
        winning_genomes = [population[i] for i in tournament_winners_indices]


        # Create a new population with the selected genomes
        new_population = population.from_solution_list(winning_genomes)

        return new_population


