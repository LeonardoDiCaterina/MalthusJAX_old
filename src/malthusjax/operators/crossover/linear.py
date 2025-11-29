"""
Linear genome crossover operators.

Implements crossover operators tailored for linear genomes
that preserve topological validity.
"""

from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig


@struct.dataclass
class LinearCrossover(BaseCrossover[LinearGenome, LinearGenomeConfig]):
    """
    Linear GP uniform crossover operator.
    
    Performs coin-flip mixing of operation codes and arguments
    between two parent genomes.
    """
    # Dynamic parameters
    mixing_ratio: float = 0.5  # Probability to take from parent1 vs parent2

    def _cross_one(self, key: chex.PRNGKey, parent1: LinearGenome, parent2: LinearGenome, 
                  config: LinearGenomeConfig) -> LinearGenome:
        """Apply crossover to produce one offspring."""
        # Generate mixing mask: True = take from parent1, False = take from parent2
        mask = jax.random.bernoulli(key, self.mixing_ratio, parent1.ops.shape)

        # Mix operation codes
        child_ops = jnp.where(mask, parent1.ops, parent2.ops)

        # Mix arguments (broadcast mask to match argument dimensions)
        mask_expanded = mask[:, None]  # Shape (L,) -> (L, 1)
        child_args = jnp.where(mask_expanded, parent1.args, parent2.args)

        # Uniform crossover between topologically valid parents produces valid offspring
        # No autocorrect needed
        return parent1.replace(ops=child_ops, args=child_args)