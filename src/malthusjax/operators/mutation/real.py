from typing import Callable
import  jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.mutation.base import AbstractMutation
import functools

class BallMutation(AbstractMutation):
    """
    Adds random noise from a uniform distribution to each gene.
    - `mutation_rate`: Probability *each gene* will be mutated.
    - `mutation_strength`: The max radius of the mutation (noise is in [-strength, +strength]).
    """
    
    def __init__(self, mutation_rate: float, mutation_strength: float = 0.1) -> None:
        """
        Args:
            mutation_rate: Probability (0.0 to 1.0) that any single gene will be mutated.
            mutation_strength: The radius of the uniform noise to add.
        """
        super().__init__(mutation_rate=mutation_rate)
        self.mutation_strength = jnp.abs(mutation_strength)

    def get_compiled_function(self) -> Callable:
        """
        Returns a JIT-compiled function for ball mutation.
        
        The function signature is:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        return functools.partial(
            _ball_mutation,
            mutation_rate=self.mutation_rate,
            mutation_strength=self.mutation_strength
        )

# --- Pure JAX Function ---

@jax.jit
def _ball_mutation(
    key: jax.Array,
    genome: jax.Array,
    mutation_rate: float,
    mutation_strength: float
) -> jax.Array:
    """Applies uniform noise to genes based on mutation_rate."""
    
    key_noise, key_mask = jar.split(key)
    
    # 1. Create noise for *all* genes
    perturbation = jar.uniform(
        key_noise,
        shape=genome.shape,
        minval=-mutation_strength,
        maxval=mutation_strength
    )
    
    # 2. Create a mask to decide *which* genes to apply noise to
    mutation_mask = jar.bernoulli(key_mask, p=mutation_rate, shape=genome.shape)
    
    # 3. Apply noise only where the mask is True
    return jnp.where(mutation_mask, genome + perturbation, genome)