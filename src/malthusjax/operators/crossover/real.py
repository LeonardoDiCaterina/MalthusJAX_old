from malthusjax.operators.base import BaseCrossover
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from flax import struct
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import chex # type: ignore


@struct.dataclass
class BlendCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Blend Crossover (BLX-α) using new batch-first paradigm.
    
    Creates offspring by sampling from intervals extended beyond the parent range.
    Very effective for real-valued optimization problems.
    """
    # --- DYNAMIC PARAMS (runtime tunable) ---
    crossover_rate: float = 0.9
    alpha: float = 0.5  # Extension factor
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """Create one offspring via blend crossover."""
        k1, k2 = jar.split(key)
        
        # Check if crossover should occur
        apply_crossover = jar.bernoulli(k1, p=self.crossover_rate)
        
        def do_crossover():
            """Perform BLX-α crossover."""
            # Calculate gamma (interval extension)
            diff = jnp.abs(p1.values - p2.values)
            gamma = 1.0 + 2.0 * self.alpha
            
            # Define sampling interval
            cmin = jnp.minimum(p1.values, p2.values) - self.alpha * diff
            cmax = jnp.maximum(p1.values, p2.values) + self.alpha * diff
            
            # Sample offspring values
            random_vals = jar.uniform(k2, shape=p1.values.shape)
            offspring_values = cmin + random_vals * (cmax - cmin)
            
            # Apply bounds if specified
            if hasattr(config, 'bounds') and config.bounds is not None:
                if isinstance(config.bounds, tuple) and len(config.bounds) == 2:
                    # Single bounds for all genes
                    min_val, max_val = config.bounds
                    offspring_values = jnp.clip(offspring_values, min_val, max_val)
            
            return offspring_values
        
        def no_crossover():
            """Return parent1 without crossover."""
            return p1.values
        
        offspring_values = jax.lax.cond(apply_crossover, do_crossover, no_crossover)
        return RealGenome(values=offspring_values)


@struct.dataclass
class SimulatedBinaryCrossover(BaseCrossover[RealGenome, RealGenomeConfig]):
    """
    Simulated Binary Crossover (SBX) using new batch-first paradigm.
    
    Simulates the behavior of single-point crossover in binary representations
    for real-valued variables. Commonly used in NSGA-II.
    """
    # --- DYNAMIC PARAMS (runtime tunable) ---
    crossover_rate: float = 0.9
    eta: float = 20.0  # Distribution index
    
    def _cross_one(self, key: chex.PRNGKey, p1: RealGenome, p2: RealGenome, config: RealGenomeConfig) -> RealGenome:
        """Create one offspring via SBX crossover."""
        k1, k2 = jar.split(key)
        
        # Check if crossover should occur
        apply_crossover = jar.bernoulli(k1, p=self.crossover_rate)
        
        def do_sbx_crossover():
            """Perform SBX crossover."""
            # Generate random numbers for each gene
            u = jar.uniform(k2, shape=p1.values.shape)
            
            # Calculate beta values based on distribution index
            beta = jnp.where(
                u <= 0.5,
                (2.0 * u) ** (1.0 / (self.eta + 1.0)),
                (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (self.eta + 1.0))
            )
            
            # Create offspring (take first child)
            c1 = 0.5 * ((1.0 + beta) * p1.values + (1.0 - beta) * p2.values)
            
            # Apply bounds if specified
            if hasattr(config, 'bounds') and config.bounds is not None:
                if isinstance(config.bounds, tuple) and len(config.bounds) == 2:
                    min_val, max_val = config.bounds
                    c1 = jnp.clip(c1, min_val, max_val)
            
            return c1
        
        def no_crossover():
            """Return parent1 without crossover."""
            return p1.values
        
        offspring_values = jax.lax.cond(apply_crossover, do_sbx_crossover, no_crossover)
        return RealGenome(values=offspring_values)


__all__ = ["BlendCrossover", "SimulatedBinaryCrossover"]