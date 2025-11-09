from typing import Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar  # type: ignore
from malthusjax.operators.mutation.base import AbstractMutation
import functools

class ScrambleMutation(AbstractMutation):
    """
    Scramble Mutation.
    The `mutation_rate` is the probability that the genome is scrambled *at all*.
    """
    def __init__(self, mutation_rate: float) -> None:
        super().__init__(mutation_rate=mutation_rate)

    def get_pure_function(self) -> Callable:
        return functools.partial(
            _scramble_mutation,
            mutation_rate=self.mutation_rate
        )

class SwapMutation(AbstractMutation):
    """
    Swap Mutation.
    Swaps two random genes in the genome.
    The `mutation_rate` is the probability that a swap occurs *at all*.
    """
    def __init__(self, mutation_rate: float) -> None:
        super().__init__(mutation_rate=mutation_rate)

    def get_pure_function(self) -> Callable:
        return functools.partial(
            _swap_mutation,
            mutation_rate=self.mutation_rate
        )

# --- Pure JAX Functions ---

@jax.jit
def _scramble_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float
) -> jax.Array:
    """Conditionally scrambles (shuffles) the entire genome."""
    key1, key2 = jar.split(key)
    
    # Conditionally apply the permutation
    return jax.lax.cond(
        jar.bernoulli(key1, p=mutation_rate),
        lambda g, k: jar.permutation(k, g), # If True: shuffle
        lambda g, k: g,                     # If False: return unchanged
        genome,
        key2
    )

@jax.jit
def _swap_two_genes(genome: jax.Array, key: jax.Array) -> jax.Array:
    """Pure function to swap two genes."""
    indices = jar.randint(key, shape=(2,), minval=0, maxval=genome.shape[0])
    idx1, idx2 = indices[0], indices[1]
    
    val1 = genome[idx1]
    val2 = genome[idx2]
    
    genome = genome.at[idx1].set(val2)
    genome = genome.at[idx2].set(val1)
    return genome

@jax.jit
def _swap_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float
) -> jax.Array:
    """Conditionally swaps two genes based on mutation_rate."""
    key1, key2 = jar.split(key)
    
    # Conditionally apply the swap
    return jax.lax.cond(
        jar.bernoulli(key1, p=mutation_rate),
        _swap_two_genes,     # If True: call swap function
        lambda g, k: g,      # If False: return genome unchanged
        genome,
        key2
    )