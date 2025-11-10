"""
Categorical genome implementation for MalthusJAX.

This module provides a simple integer genome representing categorical values,
suitable for categorical optimization problems.
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
class CategoricalGenomeConfig(AbstractGenomeConfig):
    array_shape: Tuple
    num_categories: int
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CategoricalGenomeConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'array_shape': self.array_shape, 'num_categories': self.num_categories}
# Register each concrete config as PyTree
jax.tree_util.register_pytree_node(
    CategoricalGenomeConfig,
    lambda obj: ([], obj.__dict__),
    lambda aux, _: CategoricalGenomeConfig(**aux)
)


class CategoricalGenome(AbstractGenome):
    """
    Categorical genome representation.
    
    Represents candidate solutions as integer arrays with categorical values.
    0 to num_categories - 1.
    """

    def __init__(self,
                 array_shape:Tuple[int, ...],
                 num_categories: int,
                 random_init: bool = False,
                 random_key: Optional[jnp.ndarray] = None,
                 **kwargs: Any):
        """
        Initialize Categorical genome.
        
        Args:
            array_shape: array_shape of the categorical array
            num_categories: Number of categories (integer values from 0 to num_categories - 1
            random_init: Whether to randomly initialize
            random_key: Random seed
            **kwargs: Additional metadata
        """
        self.array_shape = array_shape
        if num_categories < 1:
            raise ValueError(f"num_categories must be greater than 1 it is {num_categories}")
        self.num_categories = num_categories
        
        # Handle random key conversion
        if random_key is None:
            raise ValueError("random_key must be provided for random initialization")
        elif isinstance(random_key, int):
            random_key = jar.PRNGKey(random_key)

        # Store initialization parameters for serialization
        self._init_params = {
            'array_shape': array_shape,
            'num_categories': num_categories,
        }
        
        self.genome_config = CategoricalGenomeConfig(array_shape=array_shape, num_categories=num_categories)
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            **kwargs
        )
                            
    def _validate(self) -> bool:
        """Validate that genome contains only 0s and 1s."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != self.array_shape:
            print(f"{self.genome} = {self.array_shape}")
            return False  
        
        if jnp.any(self.genome < 0) or jnp.any(self.genome >= self.num_categories):
            print(f"jnp.any(self.genome < 0) or jnp.any(self.genome >= self.num_categories): {self.genome}")
            return False

        return self._is_valid
    
    # === JAX JIT Compatibility ===
    
    
    @classmethod
    def get_genome_config_class(cls) -> Any:
        """Get the genome config class associated with this genome type."""
        return CategoricalGenomeConfig
    
    
    
    def get_config(self):
        """Get the genome configuration."""
        return self.genome_config
    
    @classmethod
    def get_random_initialization_pure_from_config(cls, config: CategoricalGenomeConfig) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive a random key and return a tensor."""
        
        def init_fn(random_key:jnp.ndarray , init_config:CategoricalGenomeConfig) -> jnp.ndarray:
            genome =jar.randint(random_key,
                                  shape=init_config.array_shape,
                                  minval=0,
                                  maxval=init_config.num_categories).astype(jnp.int32)
            return genome
        
        return functools.partial(init_fn, init_config=config)
    
    @classmethod
    def get_random_initialization_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization from a config dictionary."""
        config = CategoricalGenomeConfig.from_dict(config_dict)
        return cls.get_random_initialization_pure_from_config(config)

    @classmethod
    def get_autocorrection_pure_from_config(cls, config:CategoricalGenomeConfig = None) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compilable correction function for the solution."""
        def correction_fn(sol: jax.Array, correction_cofig:CategoricalGenomeConfig) -> jax.Array:
            return jnp.clip(sol, 0, correction_cofig.num_categories - 1).astype(jnp.int32)

        return functools.partial(correction_fn, correction_cofig=config)
    

    @classmethod
    def get_validation_pure_from_config(self, validation_config:CategoricalGenomeConfig = None) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable validation function."""
        def validation_fn(sol: jax.Array, validation_config:CategoricalGenomeConfig) -> bool:
            is_within_bounds = jnp.all((sol >= 0) & (sol < validation_config.num_categories))
            return is_within_bounds.item()

        return functools.partial(validation_fn, validation_config=validation_config)

    
    @classmethod
    def get_distance_pure_from_config(cls, config:CategoricalGenomeConfig = None, type:str = 'hamming' ) -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compilable distance function
            you can choose between 'hamming' and 'euclidean' distance types
        """
        if type == 'hamming':
            def distance_fn(sol1: jax.Array, sol2: jax.Array) -> int:
                return jnp.sum(sol1 != sol2).item()
        elif type == 'euclidean':
            def distance_fn(sol1: jax.Array, sol2: jax.Array) -> float:
                return jnp.sqrt(jnp.sum((sol1 - sol2) ** 2)).item()
        else:
            raise ValueError(f"Unsupported distance type: {type}. Supported types are 'hamming' and 'euclidean'.")

        return distance_fn
        

    def to_tensor(self, dtype = jnp.int32) -> jax.Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(dtype)

    def distance(self, other: 'CategoricalGenome', type:str = 'euclidean',dtype = jnp.float32) -> float:
        """Calculate the distance to another genome instance."""
        if not isinstance(other, CategoricalGenome):
            return float('inf')
            
        if self.array_shape != other.array_shape:
            return float('inf')
        
        distance_fn, _ = self.get_distance_pure_from_config(type=type)
        return float(distance_fn(self.to_tensor(dtype=dtype), other.to_tensor(dtype=dtype)))   
            
    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
        # to ensure consistent hashing of JAX arrays, convert to bytes
        genome_bytes = self.genome.tobytes()
        return hashlib.sha256(genome_bytes).hexdigest()
    
    def clone(self, deep: bool = True) -> 'CategoricalGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_shape=self.array_shape,
            num_categories=self.num_categories,
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
                   f"num_categories={self.num_categories}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.genome()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_shape={self.array_shape}, invalid_state)"