from typing import Optional, Callable

import jax # type: ignore 
import jax.numpy as jnp # type: ignore 
import jax.random as jar # type: ignore 

from malthusjax.operators.mutation.base import AbstractMutation

import functools

class BitFlipMutation(AbstractMutation):
    """Bit flip mutation operator for binary genomes.
    
    Builds and returns a JIT-compiled function that flips bits in binary genomes
    with a specified probability.
    """

    def __init__(self, mutation_rate: float = None, n_outputs: int = 1) -> None:
        """Initialize bit flip mutation operator.
        
        Args:
            mutation_rate: Probability of flipping each bit.
            n_outputs: Number of output genomes.
        """
        super().__init__(mutation_rate = mutation_rate,n_outputs=n_outputs)
        
    def get_compiled_function(self):
        return self._create_mutation_function(fixed_mutation= self.mutation_rate is not None)

    def _create_mutation_function(self, fixed_mutation: bool = False) -> Callable:
        """Create the core bit flip mutation function.
        
        Arguments:
            fixed_mutation: If True, the mutation rate is passed as an argument to the function.
                           If False, the mutation rate is fixed from the instance variable.
        
        Returns:
            Pure function for bit flip mutation that can be JIT-compiled.
        """
        def bit_flip_mutation(genome_data, random_key, mutation_rate, n_outputs):
            """Core bit flip mutation function.
            
            Args:
                genome_data: Binary genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of flipping each bit.
                
            Returns:
                Mutated genome data array.
            """
            # Create mutation mask with probability mutation_rate
            mutation_mask = jar.uniform(random_key, (n_outputs,)+ genome_data.shape) < mutation_rate
            print(f"mutation_mask shape: {mutation_mask.shape}")
            # Apply mutation by flipping bits where mask is True
            # stack genome_data n_outputs times if needed
            """            if n_outputs > 1:
                genome_data = jnp.stack([genome_data]*n_outputs)
                mutation_mask = jar.uniform(random_key, (n_outputs,)+ genome_data.shape) < mutation_rate

                mutation_mask = jnp.stack([mutation_mask]*n_outputs)"""
            
            return jnp.where(mutation_mask, 
                            ~genome_data.astype(bool), 
                            genome_data).astype(genome_data.dtype)
                
        if self.mutation_rate is not None:
            return functools.partial(bit_flip_mutation, mutation_rate=self.mutation_rate, n_outputs=self.n_outputs)
        return functools.partial(bit_flip_mutation, n_outputs=self.n_outputs)
                
        
class ScrambleMutation(AbstractMutation):
    """Scramble mutation operator for binary genomes.
    
    Builds and returns a JIT-compiled function that randomly shuffles portions
    of the genome with a specified probability.
    """

    def __init__(self, mutation_rate: float = None, n_outputs: int = 1) -> None:
        """Initialize scramble mutation operator.
        
        Args:
            mutation_rate: Probability of scrambling the genome.
            n_outputs: Number of output genomes.
        """
        super().__init__(mutation_rate = mutation_rate,n_outputs=n_outputs)
        
    def get_compiled_function(self):
        return self._create_mutation_function(fixed_mutation= self.mutation_rate is not None)

    def _create_mutation_function(self, fixed_mutation: bool = False) -> Callable:
        """Create the core scramble mutation function.
        Arguments:
            fixed_mutation: If True, the mutation rate is passed as an argument to the function.
                           If False, the mutation rate is fixed from the instance variable.
        
        Returns:
            Pure function for scramble mutation that can be JIT-compiled.
        """
        
        def scramble_mutation(genome_data, random_key, mutation_rate, n_outputs):
            """Core scramble mutation function.
            
            Args:
                genome_data: Binary genome data array.
                random_key: JAX random key.
                mutation_rate: Probability of scrambling the genome.
                
            Returns:
                Mutated genome data array.
            """

            # Decide whether to scramble based on mutation_rate
            should_scramble = jar.uniform(random_key, (n_outputs,)) < mutation_rate
            scrambled = jar.permutation(random_key, (n_outputs,)+ genome_data.shape)
            
            # stack genome_data n_outputs times if needed
            if n_outputs > 1:
                genome_data = jnp.stack([genome_data]*n_outputs)
                should_scramble = jnp.stack([should_scramble]*n_outputs)
            
            return jnp.where(should_scramble[:, None], scrambled, genome_data)
        
        if self.mutation_rate is not None:
            return functools.partial(scramble_mutation, mutation_rate=self.mutation_rate, n_outputs=self.n_outputs)
        return functools.partial(scramble_mutation, n_outputs=self.n_outputs)

    

class SwapMutation(AbstractMutation):
    """Swap mutation operator for binary genomes.
    
    Builds and returns a JIT-compiled function that swaps two bits in the genome
    with a specified probability.
    
    Arguments:
        fixed_mutation: If True, the mutation rate is passed as an argument to the function.
                       If False, the mutation rate is fixed from the instance variable.
        Returns:
            Pure function for swap mutation that can be JIT-compiled.
    """

    def __init__(self, mutation_rate: float = None, n_outputs: int = 1) -> None:
        """Initialize swap mutation operator.
        
        Args:
            mutation_rate: Probability of swapping two bits.
            n_outputs: Number of output genomes.
        """
        super().__init__(mutation_rate = mutation_rate,n_outputs=n_outputs)
        
    def get_compiled_function(self):
        return self._create_mutation_function(fixed_mutation= self.mutation_rate is not None)

    def _create_mutation_function(self, fixed_mutation: bool = False) -> Callable:
        """Create the core swap mutation function.
        
        Returns:
            Pure function for swap mutation that can be JIT-compiled.
        """
        def swap_mutation(genome_data, random_key, mutation_rate, n_outputs):
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

            should_swap = jar.uniform(random_key, (n_outputs,)) < mutation_rate
            mutated_data = jax.vmap(lambda d, k, s: jax.lax.cond(s,
                                                                    swap_two_bits,
                                                                    lambda d, k: d,
                                                                    d,
                                                                    k),
                                    in_axes=(0, 0, 0))(jnp.stack([genome_data]*n_outputs),
                                                        jar.split(random_key, n_outputs),
                                                        should_swap)
            return mutated_data
        
        if self.mutation_rate is not None:
            return functools.partial(swap_mutation, mutation_rate=self.mutation_rate, n_outputs=self.n_outputs)
        return functools.partial(swap_mutation, n_outputs=self.n_outputs)