from typing import Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.mutation.base import AbstractMutation
import functools

# Import permutation-style mutations, as they are also
# valid operations on categorical genomes.
from .permutation import SwapMutation, ScrambleMutation # type: ignore

class CategoricalFlipMutation(AbstractMutation):
    """
    Replaces a gene with a new, uniformly random category.
    - `mutation_rate`: Probability *each gene* will be mutated.
    - `num_categories`: The total number of categories (e.g., 0 to N-1).
    """
    
    def __init__(self, mutation_rate: float, num_categories: int) -> None:
        """
        Args:
            mutation_rate: Probability (0.0 to 1.0) of mutating each gene.
            num_categories: The number of categories to sample from (e.g., 10).
        """
        super().__init__(mutation_rate=mutation_rate)
        if num_categories <= 1:
            raise ValueError("num_categories must be 2 or more.")
        self.num_categories = num_categories

    def get_compiled_function(self) -> Callable:
        """
        Returns a JIT-compiled function for categorical flip mutation.
        
        The function signature is:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        return functools.partial(
            _categorical_flip_mutation,
            mutation_rate=self.mutation_rate,
            num_categories=self.num_categories
        )

# --- Pure JAX Function ---

@jax.jit
def _categorical_flip_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float,
    num_categories: int
) -> jax.Array:
    """Replaces genes with a new random category based on mutation_rate."""
    
    key_mask, key_vals = jar.split(key)
    
    # 1. Create a mask to decide *which* genes to mutate
    mutation_mask = jar.bernoulli(key_mask, p=mutation_rate, shape=genome.shape)
    
    # 2. Create an array of *new random values* for all positions
    new_values = jar.randint(
        key_vals,
        shape=genome.shape,
        minval=0,
        maxval=num_categories
    )
    
    # 3. Apply new values only where the mask is True
    #    Also, ensure the new value is different from the old one (optional, but good)
    #    A simpler jnp.where is more standard:
    mutated_genome = jnp.where(mutation_mask, new_values, genome)
    
    return mutated_genome.astype(genome.dtype)


# --- Re-export permutation ops for convenience ---
# These operators work perfectly on categorical genomes too.
__all__ = [
    "CategoricalFlipMutation",
    "SwapMutation",
    "ScrambleMutation"
]