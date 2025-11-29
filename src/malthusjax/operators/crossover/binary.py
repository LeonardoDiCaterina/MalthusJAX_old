from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from flax import struct
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import chex # type: ignore

@struct.dataclass
class UniformCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Uniform Crossover using new batch-first paradigm.
    
    Produces multiple offspring where each bit comes from parent1 or parent2
    based on crossover probability.
    """
    # --- DYNAMIC PARAMS (runtime tunable) ---
    crossover_rate: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Create one offspring via uniform crossover."""
        mask = jar.bernoulli(key, p=self.crossover_rate, shape=p1.bits.shape)
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        return BinaryGenome(bits=offspring_bits)

@struct.dataclass
class SinglePointCrossover(BaseCrossover[BinaryGenome, BinaryGenomeConfig]):
    """
    Single-point crossover using new batch-first paradigm.
    
    Creates offspring by swapping segments at a random crossover point.
    """
    
    def _cross_one(self, key: chex.PRNGKey, p1: BinaryGenome, p2: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Create one offspring via single-point crossover."""
        length = p1.bits.shape[0]
        crossover_point = jar.randint(key, shape=(), minval=1, maxval=length)
        
        # Use masking approach for JAX compatibility
        indices = jnp.arange(length)
        mask = indices < crossover_point
        offspring_bits = jnp.where(mask, p1.bits, p2.bits)
        return BinaryGenome(bits=offspring_bits)

__all__ = ["UniformCrossover", "SinglePointCrossover"]