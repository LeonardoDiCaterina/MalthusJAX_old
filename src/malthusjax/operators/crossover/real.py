from malthusjax.operators.crossover.base import AbstractCrossover
from functools import partial
from typing import Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

# --- Uniform Crossover ---

class UniformCrossover(AbstractCrossover):
    """
    Uniform Crossover. `n_outputs` can be 1 or 2.
    - If `n_outputs=1`: Offspring has genes from P1 or P2 based on `crossover_rate`.
    - If `n_outputs=2`: Offspring1 is (P1 where mask, P2 where ~mask)
                        Offspring2 is (P2 where mask, P1 where ~mask)
    """
    def get_pure_function(self) -> Callable:
        return partial(
            _uniform_crossover,
            crossover_rate=self.crossover_rate,
            n_outputs=self.n_outputs
        )

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
    
    # n_outputs == 2
    offspring2 = jnp.where(mask, parent2, parent1)
    return jnp.stack([offspring1, offspring2])

# --- Single Point Crossover ---

class SinglePointCrossover(AbstractCrossover):
    """
    Single-point crossover. `n_outputs` can be 1 or 2.
    - If `n_outputs=1`: Offspring is [P1_head, P2_tail]
    - If `n_outputs=2`: Offspring1 is [P1_head, P2_tail]
                        Offspring2 is [P2_head, P1_tail]
    """
    def get_pure_function(self) -> Callable:
        # Crossover rate is ignored for single point, but part of base class
        return partial(
            _single_point_crossover,
            n_outputs=self.n_outputs
        )

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
    
    # n_outputs == 2
    offspring2 = jnp.where(mask, parent2, parent1)
    return jnp.stack([offspring1, offspring2])

# --- Average Crossover ---

class AverageCrossover(AbstractCrossover):
    """
    Average (Blend) Crossover (BLX-alpha). `n_outputs` can be 1 or 2.
    The `crossover_rate` is used as the 'alpha' blending factor.
    - Offspring1 = alpha*P1 + (1-alpha)*P2
    - Offspring2 = alpha*P2 + (1-alpha)*P1
    """
    def __init__(self, blend_rate: float, n_outputs: int = 2) -> None:
        """
        Args:
            blend_rate: The 'alpha' for blending (0.0 to 1.0).
                        0.5 is a simple average.
            n_outputs: Number of offspring (1 or 2).
        """
        # Pass blend_rate as the crossover_rate to the base class
        super().__init__(crossover_rate=blend_rate, n_outputs=n_outputs)

    def get_pure_function(self) -> Callable:
        return partial(
            _average_crossover,
            blend_rate=self.crossover_rate, # Use the rate as the blend_rate
            n_outputs=self.n_outputs
        )

@partial(jax.jit, static_argnames=["n_outputs"])
def _average_crossover(
    key: jax.Array, # Key is unused, but part of the standard signature
    parent1: jax.Array,
    parent2: jax.Array,
    blend_rate: float,
    n_outputs: int
) -> jax.Array:
    """Pure JAX average (blend) crossover."""
    
    offspring1 = (blend_rate * parent1) + ((1.0 - blend_rate) * parent2)
    
    if n_outputs == 1:
        return offspring1.reshape((1,) + parent1.shape)

    # n_outputs == 2
    offspring2 = (blend_rate * parent2) + ((1.0 - blend_rate) * parent1)
    return jnp.stack([offspring1, offspring2])