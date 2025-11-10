"""
Binary genome implementation for MalthusJAX.

This module provides a simple binary string genome representation suitable for
binary optimization problems and as a reference implementation.
"""

from dataclasses import dataclass
import functools
from typing import Any, Optional, Dict, Tuple, Callable

import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenome, AbstractGenomeConfig

@dataclass(frozen=True)
class RealGenomeConfig(AbstractGenomeConfig):
    array_shape: Tuple
    min_val: float
    max_val: float
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'RealGenomeConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'array_shape': self.array_shape, 'min_val': self.min_val, 'max_val': self.max_val}

# Register each concrete config as PyTree
jax.tree_util.register_pytree_node(
    RealGenomeConfig,
    lambda obj: ([], obj.__dict__),
    lambda aux, _: RealGenomeConfig(**aux)
)


class RealGenome(AbstractGenome):
    """
    Real-valued genome representation.
    
    Represents candidate solutions as real-valued arrays.
    """

    def __init__(self,
                 array_shape:Tuple[int, ...],
                 min_val: float,
                 max_val: float,
                 random_init: bool = False,
                 random_key: Optional[jnp.ndarray] = None,
                 **kwargs: Any):
        """
        Initialize Real genome.
        
        Args:
            array_shape: array_shape of the real array
            min_val: Minimum value for each element
            max_val: Maximum value for each element
            random_init: Whether to randomly initialize
            random_key: Random seed
            **kwargs: Additional metadata
        """
        self.array_shape = array_shape
        if max_val < min_val:
            raise ValueError(f"max_val {max_val} must be greater than min_val {min_val}")
                             
        self.max_val = max_val
        self.min_val = min_val
        
        # Handle random key conversion
        if random_key is None:
            raise ValueError("random_key must be provided for random initialization")
        elif isinstance(random_key, int):
            random_key = jar.PRNGKey(random_key)

        # Store initialization parameters for serialization
        self._init_params = {
            'array_shape': array_shape,
            'min_val': min_val,
            'max_val': max_val
        }
        
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            **kwargs
        )

        '''def _random_init(self) -> None:
        """Randomly initialize the genome as a real array."""
        self.genome = jar.uniform(self.random_key,
                                  shape=self.array_shape,
                                  minval=self.min_val,
                                  maxval=self.max_val)'''

    def _validate(self) -> bool:
        """Validate that genome contains only 0s and 1s."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != self.array_shape:
            print(f"{self.genome} = {self.array_shape}")
            return False
            

        self._is_valid = jnp.all(jnp.logical_and(self.genome >= self.min_val, self.genome <= self.max_val)).item()
        return self._is_valid
    
    # === JAX JIT Compatibility ===
    

    @classmethod
    def get_genome_config_class(cls) -> Any:
        """Get the genome config class associated with this genome type."""
        return RealGenomeConfig
        
    def get_config(self):
        """Get the genome configuration."""
        return RealGenomeConfig(array_shape=self.array_shape, min_val=self.min_val, max_val=self.max_val)
    
    
    @classmethod
    def get_random_initialization_pure_from_config(cls, config: RealGenomeConfig) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive a random key and return a tensor."""
        
        def init_fn(random_key: jar.PRNGKey, init_config:RealGenomeConfig) -> jnp.ndarray:
            genome = jar.uniform(random_key,
                                 shape=init_config.array_shape,
                                 minval=init_config.min_val,
                                 maxval=init_config.max_val)
            return genome
        
    
        return functools.partial(init_fn, init_config=config)
    
    @classmethod
    def get_random_initialization_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization from a config dict."""
        
        config = RealGenomeConfig.from_dict(config_dict)
        return cls.get_random_initialization_pure_from_config(config)
    

    @classmethod
    def get_autocorrection_pure_from_config(cls, config:RealGenomeConfig = None) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compilable correction function for the solution."""
        def correction_fn(sol: jax.Array, correction_cofig:RealGenomeConfig) -> jax.Array:
            return jnp.clip(sol, correction_cofig.min_val, correction_cofig.max_val)

        return functools.partial(correction_fn, correction_cofig=config)
    

    @classmethod
    def get_validation_pure_from_config(self, validation_config:RealGenomeConfig = None) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable validation function."""
        def validation_fn(sol: jax.Array, validation_config:RealGenomeConfig) -> bool:
            is_within_bounds = jnp.all(jnp.logical_and(sol >= validation_config.min_val, sol <= validation_config.max_val))
            return is_within_bounds.item()
        
        return functools.partial(validation_fn, validation_config=validation_config)

    
    @classmethod
    def get_distance_pure_from_config(cls, config:RealGenomeConfig = None, type:str = "euclidean") -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compilable distance function
            you can choose between 'euclidean', 'manhattan' and 'chebyshev' distance types."""
        if type == "euclidean":
            def distance_fn(sol1, sol2):
                return jnp.sqrt(jnp.sum((sol1 - sol2) ** 2)).astype(float)
        if type == "manhattan":
            def distance_fn(sol1, sol2):
                return jnp.sum(jnp.abs(sol1 - sol2)).astype(float)
        if type == "chebyshev":
            def distance_fn(sol1, sol2):
                return jnp.max(jnp.abs(sol1 - sol2)).astype(float)
        return distance_fn
        

    def to_tensor(self, dtype = jnp.float32) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(dtype)

    def distance(self, other: 'RealGenome', type:str = 'euclidean',dtype = jnp.float32) -> float:
        """Calculate Hamming distance between two binary genomes."""
        if not isinstance(other, RealGenome):
            return float('inf')
            
        if self.array_shape != other.array_shape:
            return float('inf')
        
        distance_fn, _ = self.get_distance_pure_from_config(type=type)
        return float(distance_fn(self.to_tensor(dtype=dtype), other.to_tensor(dtype=dtype)))   
            
    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
        # to ensure consistent hashing of floating-point arrays, we convert to bytes with a fixed precision
        byte_representation = self.genome.astype(jnp.float32).tobytes()
        return hashlib.sha256(byte_representation).hexdigest()  

    def clone(self, deep: bool = True) -> 'RealGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_shape=self.array_shape,
            min_val=self.min_val,
            max_val=self.max_val,
            random_init=False,
        )
        
        # JAX arrays are immutable, so we can share them
        new_genome.genome = self.genome
        new_genome._is_valid = self._is_valid
        new_genome.jax_random_key = self.jax_random_key
        
        return new_genome

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor."""
        if tensor.shape != self.array_shape:
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with array_shape {self.array_shape}")

        self.genome = tensor
        self.invalidate()
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    def __str__(self) -> str:
        """String representation of the genome."""
        try:
            return f"{self.genome}(array_shape={self.array_shape}, valid={self.is_valid})"
        except Exception:
            return f"{type(self).__name__}(array_shape={self.array_shape}, invalid_state)"

    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        try:
            return (f"{self.__class__.__name__}("
                   f"array_shape={self.array_shape}, "
                   f"min_val={self.min_val}, "
                   f"max_val={self.max_val}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.genome()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_shape={self.array_shape}, invalid_state)"