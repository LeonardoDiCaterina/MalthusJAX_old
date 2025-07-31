from typing import Optional, Callable

import jax # type: ignore 
import jax.numpy as jnp # type: ignore 
import jax.random as jar # type: ignore 

from malthusjax.core.population.population import Population
from malthusjax.operators.mutation.base import AbstractMutation

class BitFlipMutation(AbstractMutation[Population]):
    """Bit flip mutation operator for binary genomes.
    
    Builds and returns a JIT-compiled function that flips bits in binary genomes
    with a specified probability.
    """
    
    def __init__(self, mutation_rate: float = 0.01) -> None:
        """Initialize bit flip mutation operator.
        
        Args:
            mutation_rate: Probability of flipping each bit.
        """
        super().__init__(mutation_rate=mutation_rate)
    
    def _create_mutation_function(self) -> Callable:
        """Create the core bit flip mutation function.
        
        Returns:
            Pure function for bit flip mutation that can be JIT-compiled.
        """
        def bit_flip_mutation(genome_data, random_key, mutation_rate):
            """Core bit flip mutation function.
            
            Args:
                genome_data: Binary genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of flipping each bit.
                
            Returns:
                Mutated genome data array.
            """
            # Create mutation mask with probability mutation_rate
            mutation_mask = jar.uniform(random_key, genome_data.shape) < mutation_rate
            # Apply mutation by flipping bits where mask is True
            return jnp.where(mutation_mask, 
                            ~genome_data.astype(bool), 
                            genome_data).astype(genome_data.dtype)
        
        return bit_flip_mutation
        
        
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