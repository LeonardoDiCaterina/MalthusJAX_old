from typing import Optional, Callable

import jax # type: ignore 
import jax.numpy as jnp # type: ignore 
import jax.random as jar # type: ignore 

from malthusjax.core.population.population import Population
from malthusjax.operators.mutation.base import AbstractMutation

class CategoricalFlipMutation(AbstractMutation[Population]):
    """Categorical flip mutation operator for categorical genomes.

    Builds and returns a JIT-compiled function that flips categories in categorical genomes
    with a specified probability.
    """

    def __init__(self, mutation_rate: float = 0.01, num_categories: int = 2) -> None:
        """Initialize categorical flip mutation operator.

        Args:
            mutation_rate: Probability of flipping each category.
            num_categories: Number of categories for the genome.
        """
        super().__init__(mutation_rate=mutation_rate)
        self.num_categories = num_categories        

    def _create_mutation_function(self) -> Callable:
        """Create the core bit flip mutation function.
        
        Returns:
            Pure function for bit flip mutation that can be JIT-compiled.
        """

        num_categories = self.num_categories
        def categorical_flip_mutation(genome_data, random_key, mutation_rate):
            """Core categorical flip mutation function.

            Args:
                genome_data: Categorical genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of flipping each category.

            Returns:
                Mutated genome data array.
            """
            # Create mutation mask with probability mutation_rate
            mutation_mask_bool = jar.uniform(random_key, genome_data.shape) < mutation_rate
            mutation_mask_array = jar.randint(random_key, genome_data.shape, 0, num_categories)
            # Apply mutation by flipping bits where mask is True
            return jnp.where(mutation_mask_bool, 
                            mutation_mask_array, 
                            genome_data).astype(genome_data.dtype)

        return categorical_flip_mutation


class ScrambleMutation(AbstractMutation[Population]):
    """Scramble mutation operator for categorical genomes.

    Builds and returns a JIT-compiled function that randomly shuffles portions
    of the genome with a specified probability.
    """
    
    def __init__(self, mutation_rate: float = 0.01) -> None:
        """Initialize scramble mutation operator.
        
        Args:
            mutation_rate: Probability of scrambling the genome.
        """
        super().__init__(mutation_rate=mutation_rate)

    def _create_mutation_function(self) -> Callable:
        """Create the core scramble mutation function.
        
        Returns:
            Pure function for scramble mutation that can be JIT-compiled.
        """
        def scramble_mutation(genome_data, random_key, mutation_rate):
            """Core scramble mutation function.
            
            Args:
                genome_data: Categorical genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of scrambling.
                
            Returns:
                Mutated genome data array.
            """
            # Decide whether to scramble based on mutation_rate
            should_scramble = jar.uniform(random_key) < mutation_rate
            scrambled = jar.permutation(random_key, genome_data)
            return jnp.where(should_scramble, scrambled, genome_data)
        
        return scramble_mutation
    

class SwapMutation(AbstractMutation[Population]):
    """Swap mutation operator for categorical genomes.

    Builds and returns a JIT-compiled function that swaps two categories in the genome
    with a specified probability.
    """
    
    def __init__(self, mutation_rate: float = 0.01) -> None:
        """Initialize swap mutation operator.
        
        Args:
            mutation_rate: Probability of swapping two bits.
        """
        super().__init__(mutation_rate=mutation_rate)

    def _create_mutation_function(self) -> Callable:
        """Create the core swap mutation function.
        
        Returns:
            Pure function for swap mutation that can be JIT-compiled.
        """
        def swap_mutation(genome_data, random_key, mutation_rate):
            """Core swap mutation function.
            
            Args:
                genome_data: Categorical genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of swapping two categories.
                
            Returns:
                Mutated genome data array.
            """
            def swap_two_categories(data, key):
                idx1, key = jar.randint(key, (2,), 0, data.shape[0]), jar.split(key)[0]
                idx2, key = jar.randint(key, (2,), 0, data.shape[0]), jar.split(key)[0]
                data = data.at[idx1[0]].set(data[idx2[0]])
                data = data.at[idx2[0]].set(data[idx1[0]])
                return data

            should_swap = jar.uniform(random_key) < mutation_rate
            mutated_data = jax.lax.cond(should_swap,
                                        swap_two_categories,
                                        lambda d, k: d,
                                        genome_data,
                                        random_key)
            return mutated_data
        
        return swap_mutation
    
