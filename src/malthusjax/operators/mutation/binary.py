from typing import Optional, Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.mutation.base import AbstractMutation
import functools


class BitFlipMutation(AbstractMutation):
    """Bit flip mutation operator for binary genomes.
    
    Flips bits in binary genomes with a specified probability.
    """

    def __init__(self, mutation_rate: float) -> None:
        """Initialize bit flip mutation operator.
        
        Args:
            mutation_rate: Probability of flipping each bit (0.0 to 1.0).
        """
        super().__init__(mutation_rate=mutation_rate)
        
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for bit flip mutation.
        
        The function signature is:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        return functools.partial(
            _bit_flip_mutation,
            mutation_rate=self.mutation_rate
        )


class ScrambleMutation(AbstractMutation):
    """Scramble mutation operator for binary genomes.
    
    Randomly shuffles the entire genome with a specified probability.
    """

    def __init__(self, mutation_rate: float) -> None:
        """Initialize scramble mutation operator.
        
        Args:
            mutation_rate: Probability of scrambling the genome (0.0 to 1.0).
        """
        super().__init__(mutation_rate=mutation_rate)
        
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for scramble mutation.
        The function signature is:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        return functools.partial(
            _scramble_mutation,
            mutation_rate=self.mutation_rate
        )


class SwapMutation(AbstractMutation):
    """Swap mutation operator for binary genomes.
    
    Swaps two randomly selected bits in the genome with a specified probability.
    """

    def __init__(self, mutation_rate: float) -> None:
        """Initialize swap mutation operator.
        
        Args:
            mutation_rate: Probability of performing a swap (0.0 to 1.0).
        """
        super().__init__(mutation_rate=mutation_rate)
        
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for swap mutation.
        
        The function signature is:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        return functools.partial(
            _swap_mutation,
            mutation_rate=self.mutation_rate
        )


# --- Pure JAX Functions ---

@jax.jit
def _bit_flip_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float
) -> jax.Array:
    """Flips bits in a binary genome based on mutation_rate."""
    
    # Create mutation mask with probability mutation_rate
    mutation_mask = jar.bernoulli(key, p=mutation_rate, shape=genome.shape)
    
    # Apply mutation by flipping bits where mask is True
    return jnp.where(mutation_mask, 
                    ~genome.astype(bool), 
                    genome).astype(genome.dtype)


@jax.jit
def _scramble_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float
) -> jax.Array:
    """Scrambles (shuffles) the genome based on mutation_rate."""
    
    key_decide, key_shuffle = jar.split(key)
    
    # Decide whether to scramble based on mutation_rate
    should_scramble = jar.uniform(key_decide) < mutation_rate
    
    # Create scrambled version
    scrambled_genome = jar.permutation(key_shuffle, genome, axis=0)
    
    # Return scrambled version if should_scramble, else original
    return jnp.where(should_scramble, scrambled_genome, genome)


@jax.jit
def _swap_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float
) -> jax.Array:
    """Swaps two randomly selected bits based on mutation_rate."""
    
    key_decide, key_idx1, key_idx2 = jar.split(key, 3)
    
    # Decide whether to swap based on mutation_rate
    should_swap = jar.uniform(key_decide) < mutation_rate
    
    def do_swap(genome_to_swap):
        # Select two random indices
        idx1 = jar.randint(key_idx1, (), 0, genome_to_swap.shape[0])
        idx2 = jar.randint(key_idx2, (), 0, genome_to_swap.shape[0])
        
        # Perform the swap
        swapped = genome_to_swap.at[idx1].set(genome_to_swap[idx2])
        swapped = swapped.at[idx2].set(genome_to_swap[idx1])
        return swapped
    
    # Use jax.lax.cond to conditionally apply the swap
    return jax.lax.cond(
        should_swap,
        do_swap,
        lambda x: x,  # identity function
        genome
    )


__all__ = [
    "BitFlipMutation",
    "ScrambleMutation", 
    "SwapMutation"
]