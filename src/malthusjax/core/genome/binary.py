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
class BinaryGenomeConfig(AbstractGenomeConfig):
    array_shape: Tuple
    p: float
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BinaryGenomeConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {'array_shape': self.array_shape, 'p': self.p}

# Register each concrete config as PyTree
jax.tree_util.register_pytree_node(
    BinaryGenomeConfig,
    lambda obj: ([], obj.__dict__),
    lambda aux, _: BinaryGenomeConfig(**aux)
)


class BinaryGenome(AbstractGenome):
    """
    Binary genome implementation for binary optimization problems.
    
    Represents candidate solutions as binary arrays (0s and 1s).
    """

    def __init__(self,
                 array_shape:Tuple[int, ...],
                 p: float,
                 random_init: bool = False,
                 random_key: Optional[jnp.ndarray] = None,
                 **kwargs: Any):
        """
        Initialize binary genome.
        
        Args:
            array_shape: array_shape of the binary array
            p: Probability of 1s during random initialization
            random_init: Whether to randomly initialize
            random_key: Random seed
            **kwargs: Additional metadata
        """
        self.array_shape = array_shape
        if 0 >= p or p >= 1:
            raise ValueError (f"p must be between 0 and 1 it is {p}")
        self.p = p

            
        # Store initialization parameters for serialization
        self._init_params = {
            'array_shape': array_shape,
            'p': p
        }
        self.genome_config = BinaryGenomeConfig(array_shape=array_shape, p=p)
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            **kwargs
        )

        '''def _random_init(self) -> None:
        """Randomly initialize the genome as a binary array."""
        # --- FIX: Use self.random_key property ---
        subkey = self.random_key
        if subkey is None:
            raise ValueError("Random key is not set for random_init")
        self.genome = jar.bernoulli(subkey, p=self.p, shape=self.array_shape).astype(jnp.bool_)
        '''
    def _validate(self) -> bool:
        """Validate that genome contains only 0s and 1s."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != self.array_shape:
            print(f"{self.genome} = {self.array_shape}")
            return False
            
        # Check that dtype is integer or boolean
        if not jnp.issubdtype(self.genome.dtype, jnp.integer) and not jnp.issubdtype(self.genome.dtype, jnp.bool_):
            print("problem counting zeroes")
            return False

        self._is_valid = jnp.all(jnp.logical_or(self.genome == 0, self.genome == 1)).item()
        return self._is_valid
    
    # === JAX JIT Compatibility ===
    
    
    @classmethod
    def get_genome_config_class(cls) -> Any:
        """Get the genome config class associated with this genome type."""
        return BinaryGenomeConfig
    
    
    def get_config(self):
        """Get the genome configuration."""
        return self.genome_config
    
    
    @classmethod
    def get_random_initialization_pure_from_config(cls, config: BinaryGenomeConfig) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive a random key and return a tensor."""
        
        def init_fn(random_key:jnp.ndarray, array_shape: Tuple, p: float) -> jnp.ndarray:
            genome = jar.bernoulli(random_key, p=p, shape=array_shape).astype(jnp.bool_)
            return genome
        
        return functools.partial(init_fn, array_shape=config.array_shape, p=config.p)
    
    
    @classmethod
    def get_random_initialization_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization from config dict."""
        config = BinaryGenomeConfig.from_dict(config_dict)
        return cls.get_random_initialization_pure_from_config(config)
    

    @classmethod
    def get_autocorrection_pure_from_config(cls, config:BinaryGenomeConfig = None) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compilable correction function for the solution."""
        def correction_fn(sol: jax.Array) -> jax.Array:
            return jnp.clip(sol, 0, 1)

        return correction_fn
    
    @classmethod
    def get_initialization_pure_from_dict(cls, config: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for genome initialization from config dict."""
        config_obj = BinaryGenomeConfig.from_dict(config)
        return cls.get_random_initialization_pure_from_config(config_obj)
    

    @classmethod
    def get_validation_pure_from_config(self, config:BinaryGenomeConfig = None) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable validation function."""
        def validation_fn(sol: jax.Array) -> jax.Array:
            # Return boolean array indicating validity
            array = jnp.logical_or(sol == 0, sol == 1)
            return jnp.all(array)
        
        return validation_fn
    
    @classmethod
    def get_validation_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable validation function from config dict."""
        config = BinaryGenomeConfig.from_dict(config_dict)
        return cls.get_validation_pure_from_config(config)

    
    @classmethod
    def get_distance_pure_from_config(cls, config:BinaryGenomeConfig = None) -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compilable distance function."""
        def distance_fn(sol1, sol2):
            return jnp.sum(sol1 != sol2).astype(int)
        
        return distance_fn
    
    @classmethod
    def get_distance_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compilable distance function from config dict."""
        config = BinaryGenomeConfig.from_dict(config_dict)
        return cls.get_distance_pure_from_config(config)  
    


    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.int32)  # Use int32 for consistency

    def distance(self, other: 'BinaryGenome') -> float:
        """Calculate Hamming distance between two binary genomes."""
        if not isinstance(other, BinaryGenome):
            return float('inf')
            
        if self.array_shape != other.array_shape:
            return float('inf')
            
        return float(jnp.sum(jnp.abs(self.to_tensor() - other.to_tensor())))

    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
        # Convert binary array to string for consistent hashing
        binary_string = ''.join(str(int(x)) for x in self.genome.tolist())
        return hashlib.md5(binary_string.encode()).hexdigest()

    def clone(self, deep: bool = True) -> 'BinaryGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_shape=self.array_shape,
            p=self.p,
            random_init=False,
            **self._metadata.copy()
        )
        
        # JAX arrays are immutable, so we can share them
        new_genome.genome = self.genome
        new_genome._is_valid = self._is_valid
        new_genome.jax_key = self.jax_key
        
        return new_genome

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor."""
        if tensor.shape != self.array_shape:
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with array_shape {self.array_shape}")

        self.genome = tensor.astype(jnp.bool_)
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
                   f"p={self.p}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.semantic_key()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_shape={self.array_shape}, invalid_state)"