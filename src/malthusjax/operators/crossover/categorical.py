from malthusjax.operators.crossover.base import AbstractCrossover
from functools import partial
from typing import Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

class UniformCrossover(AbstractCrossover):
    """
    Uniform Crossover for categorical genomes. `n_outputs` can be 1 or 2.
    - If `n_outputs=1`: Offspring has genes from P1 or P2 based on `crossover_rate`.
    - If `n_outputs=2`: Offspring1 is (P1 where mask, P2 where ~mask)
                        Offspring2 is (P2 where mask, P1 where ~mask)
    """
    def get_compiled_function(self) -> Callable:
        return partial(
            _uniform_crossover,
            crossover_rate=self.crossover_rate,
            n_outputs=self.n_outputs
        )

class SinglePointCrossover(AbstractCrossover):
    """
    Single-point crossover for categorical genomes. `n_outputs` can be 1 or 2.
    - If `n_outputs=1`: Offspring is [P1_head, P2_tail]
    - If `n_outputs=2`: Offspring1 is [P1_head, P2_tail]
                        Offspring2 is [P2_head, P1_tail]
    """
    def get_compiled_function(self) -> Callable:
        return partial(
            _single_point_crossover,
            n_outputs=self.n_outputs
        )

# --- Pure JAX Functions ---
# Note: These are identical to the binary versions, as the logic
# (selecting indices from parents) is genome-type-agnostic.

@partial(jax.jit, static_argnames=["n_outputs"])
def _uniform_crossover(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array,
    crossover_rate: float,
    n_outputs: int
) -> jax.Array:
    """Pure JAX uniform crossover. Handles 1 or 2 outputs."""
    
    mask = jar.bernoulli(key, p=crossover_rate, shape=parent1.shape)
    
    offspring1 = jnp.where(mask, parent1, parent2)
    
    if n_outputs == 1:
        return offspring1.reshape((1,) + parent1.shape)
    
    if n_outputs == 2:
        offspring2 = jnp.where(mask, parent2, parent1)
        return jnp.stack([offspring1, offspring2])
    
    # Fallback for n_outputs > 2 (just vmap it)
    keys = jar.split(key, n_outputs)
    return jax.vmap(
        lambda k: jnp.where(jar.bernoulli(k, p=crossover_rate, shape=parent1.shape), parent1, parent2)
    )(keys)


@partial(jax.jit, static_argnames=["n_outputs"])
def _single_point_crossover(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array,
    n_outputs: int
) -> jax.Array:
    """Pure JAX single-point crossover."""
    
    crossover_point = jar.randint(key, shape=(), minval=0, maxval=parent1.shape[0])
    mask = jnp.arange(parent1.shape[0]) < crossover_point
    
    offspring1 = jnp.where(mask, parent1, parent2)
    
    if n_outputs == 1:
        return offspring1.reshape((1,) + parent1.shape)
    
    # n_outputs == 2 (or more, but 2 is standard)
    offspring2 = jnp.where(mask, parent2, parent1)
    return jnp.stack([offspring1, offspring2])