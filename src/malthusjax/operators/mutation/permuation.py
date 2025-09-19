from typing import Optional, Callable

import jax # type: ignore 
import jax.numpy as jnp # type: ignore 
import jax.random as jar # type: ignore 

from malthusjax.core.population.population import Population
from malthusjax.operators.mutation.base import AbstractMutation
        
        
class ScrambleMutation(AbstractMutation[Population]):
    """Scramble mutation operator for binary genomes.
    
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
                genome_data: Binary genome data array.
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
    """Swap mutation operator for binary genomes.
    
    Builds and returns a JIT-compiled function that swaps two bits in the genome
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
                genome_data: Binary genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of swapping two bits.
                
            Returns:
                Mutated genome data array.
            """
            def swap_two_bits(data, key):
                idx1, key = jar.randint(key, (2,), 0, data.shape[0]), jar.split(key)[0]
                idx2, key = jar.randint(key, (2,), 0, data.shape[0]), jar.split(key)[0]
                data = data.at[idx1[0]].set(data[idx2[0]])
                data = data.at[idx2[0]].set(data[idx1[0]])
                return data

            should_swap = jar.uniform(random_key) < mutation_rate
            mutated_data = jax.lax.cond(should_swap,
                                        swap_two_bits,
                                        lambda d, k: d,
                                        genome_data,
                                        random_key)
            return mutated_data
        
        return swap_mutation