"""
Compatibility layer for MalthusJAX v1 to v2 migration.

Provides adapters and backwards compatibility for existing Level 3 engines
during the transition to the new architecture.
"""

import warnings
from typing import Any, Optional
import jax.numpy as jnp
from malthusjax.core.base import BaseGenome, BasePopulation


class JAXTensorizable:
    """Legacy base class - replaced by new BaseGenome/BasePopulation system."""
    
    def __init__(self, random_key: Optional[jnp.ndarray] = None):
        warnings.warn(
            "JAXTensorizable is deprecated. Use BaseGenome/BasePopulation instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.jax_random_key = random_key
        
    @property
    def random_key(self) -> Optional[jnp.ndarray]:
        """Get JAX random key with auto-splitting."""
        if self.jax_random_key is not None:
            import jax.random as jar
            self.jax_random_key, subkey = jar.split(self.jax_random_key)
            return subkey
        return self.jax_random_key
    
    @random_key.setter 
    def random_key(self, value: Optional[jnp.ndarray]) -> None:
        """Set JAX random key."""
        self.jax_random_key = value

    def to_tensor(self) -> jnp.ndarray:
        """Legacy tensor conversion - override in subclasses."""
        raise NotImplementedError("to_tensor must be implemented by subclasses")

    @classmethod
    def from_tensor(cls, tensor: jnp.ndarray, **kwargs: Any) -> 'JAXTensorizable':
        """Legacy tensor deserialization - override in subclasses."""
        raise NotImplementedError("from_tensor must be implemented by subclasses")


class GenomeAdapter:
    """
    Adapter to bridge old genome interfaces to new BaseGenome system.
    
    Can be used temporarily while migrating Level 3 engines.
    """
    
    def __init__(self, new_genome: BaseGenome):
        self.genome = new_genome
        
    def to_tensor(self) -> jnp.ndarray:
        """Convert to tensor representation."""
        # For linear genomes, concatenate ops and args
        if hasattr(self.genome, 'ops') and hasattr(self.genome, 'args'):
            return jnp.concatenate([
                self.genome.ops.flatten(),
                self.genome.args.flatten()
            ])
        else:
            raise NotImplementedError(f"Don't know how to tensorize {type(self.genome)}")
    
    @classmethod
    def from_tensor(cls, tensor: jnp.ndarray, genome_cls: type, config: Any) -> 'GenomeAdapter':
        """Create from tensor representation."""
        # This is a basic implementation - may need customization
        warnings.warn(
            "GenomeAdapter.from_tensor is a basic implementation - "
            "consider migrating to new BaseGenome system",
            UserWarning,
            stacklevel=2
        )
        raise NotImplementedError("from_tensor not fully implemented in adapter")


def migrate_legacy_population(legacy_pop: Any, new_genome_cls: type) -> BasePopulation:
    """
    Helper function to migrate legacy population structures.
    
    Args:
        legacy_pop: Old population object
        new_genome_cls: New genome class to migrate to
        
    Returns:
        New BasePopulation instance
    """
    warnings.warn(
        "migrate_legacy_population is a helper for migration - "
        "consider fully migrating to new system",
        UserWarning,
        stacklevel=2
    )
    
    # Basic implementation - customize as needed
    raise NotImplementedError("Legacy migration helper needs customization")


# Migration guide message
_MIGRATION_GUIDE = """
MalthusJAX Migration Guide (v1 -> v2)

Key Changes:
1. JAXTensorizable -> BaseGenome/BasePopulation
2. Operators now support configurable num_offspring
3. Evaluators use automatic vectorization
4. Selection operates on fitness arrays directly

Migration Steps:
1. Replace JAXTensorizable inheritance with BaseGenome
2. Update operators to use new base classes
3. Use BasePopulation for population management
4. Update fitness evaluation to new evaluator system

For detailed migration help, see documentation.
"""

def print_migration_guide():
    """Print migration guide for v1 to v2 transition."""
    print(_MIGRATION_GUIDE)