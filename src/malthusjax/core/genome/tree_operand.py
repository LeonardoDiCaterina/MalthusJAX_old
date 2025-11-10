"""
Categorical genome extension for operand genome representation in MalthusJAX trees.

This module provides a simple integer genome representing operand values,
"""

from dataclasses import dataclass
import functools
from typing import Any, Optional, Dict, Tuple, List, Callable

import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenomeConfig, AbstractGenome

@dataclass(frozen=True)
class TreeOperandGenomeConfig(AbstractGenomeConfig):
    n_operands_per_node: int
    maximum_depth: int
    n_features_dataset: int
    
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TreeOperandGenomeConfig':
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_operands_per_node': self.n_operands_per_node,
            'maximum_depth': self.maximum_depth,
            'n_features_dataset': self.n_features_dataset
        }
# Register each concrete config as PyTree
jax.tree_util.register_pytree_node(
    TreeOperandGenomeConfig,
    lambda obj: ([], obj.__dict__),
    lambda aux, _: TreeOperandGenomeConfig(**aux)
)


class TreeOperandGenome(AbstractGenome):
    """
    Binary genome implementation for real optimization problems.
    
    Represents candidate solutions as real-valued arrays.
    """

    def __init__(self,
                    n_operands_per_node: int,
                    maximum_depth: int,
                    n_features_dataset: int,
                 random_init: bool = False,
                 random_key: Optional[int] = None,
                 **kwargs: Any):
        """
        Initialize Categorical genome.
        
        Args:
            array_shape: array_shape of the real array
            p: Probability of 1s during random initialization
            random_init: Whether to randomly initialize
            random_key: Random seed
            **kwargs: Additional metadata
        """

        assert n_operands_per_node > 0, "n_operands_per_node must be greater than 0"
        assert maximum_depth > 0, "maximum_depth must be greater than 0"
        assert n_features_dataset > 0, "n_features_dataset must be greater than 0"
        
        self.n_operands_per_node = n_operands_per_node
        self.maximum_depth = maximum_depth
        self.n_features_dataset = n_features_dataset
        # Handle random key conversion
        if random_key is None:
            self.random = jar.PRNGKey(0)
        elif isinstance(random_key, int):
            random_key = jar.PRNGKey(random_key)

        # Store initialization parameters for serialization
        self._init_params = {
            'n_operands_per_node': n_operands_per_node,
            'maximum_depth': maximum_depth,
            'n_features_dataset': n_features_dataset
        }
        
        self.genome_config = TreeOperandGenomeConfig(
            n_operands_per_node=n_operands_per_node,
            maximum_depth=maximum_depth,
            n_features_dataset=n_features_dataset
        )
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            **kwargs
        )

    def _random_init(self) -> None:
        """Randomly initialize the genome as a real array."""
        init_fn = self.get_random_initialization_pure_from_config(self.genome_config)
                
        self.genome = init_fn(self.random_key)
                            
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
        return TreeOperandGenomeConfig
    
    
    
    def get_config(self):
        """Get the genome configuration."""
        return self.genome_config
    

                  

    @classmethod
    def get_random_initialization_pure_from_config(cls, config: TreeOperandGenomeConfig) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive an ARRAY of random keys and return a tensor."""
        
        def init_atomic_tree(random_key: jnp.ndarray, max_value: jnp.ndarray) -> jnp.ndarray:
            """Initialize a single tree node with random indices"""
            return jar.randint(key=random_key,
                            minval=0,
                            maxval=max_value,
                            shape=(config.n_operands_per_node,),
                            dtype=jnp.int32)
        
        def init_fn(random_key: jnp.ndarray , init_config:TreeOperandGenomeConfig) -> jnp.ndarray:
            # Determine the max value (depth_trees) for each random instruction
            # We assume the length of random_keys is equal to the number of trees (maximum_depth)
            random_keys = jar.split(random_key, num=init_config.maximum_depth)
            num_trees = init_config.maximum_depth
            
            max_value_trees = jnp.arange(init_config.n_features_dataset, 
                                    init_config.n_features_dataset + num_trees)
            
            # Vectorize the initialization over the keys and max values
            # jax.vmap maps over the leading axis of both random_keys and depth_trees.
            return jax.vmap(init_atomic_tree)(random_keys, max_value_trees)
            
        # Return a function that is partially applied with the config and expects the array of keys
        return functools.partial(init_fn, init_config=config)
    
    
    @classmethod
    def get_random_initialization_pure_from_dict(cls, config_dict: Dict[str, Any]) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization from config dict."""
        config = TreeOperandGenomeConfig.from_dict(config_dict)
        return cls.get_random_initialization_pure_from_config(config)
    
    @classmethod
    def get_autocorrection_pure_from_config(cls, config:TreeOperandGenomeConfig = None) -> Callable[[jax.Array], jax.Array]:
        raise NotImplementedError("Autocorrection is not implemented for OperandGenome.")

    @classmethod
    def get_validation_pure_from_config(self, validation_config:TreeOperandGenomeConfig = None) -> Callable[[jax.Array], bool]:
        raise NotImplementedError("Validation is not implemented for OperandGenome.")

    
    @classmethod
    def get_distance_pure_from_config(cls, config:TreeOperandGenomeConfig = None, type:str = 'hamming' ) -> Callable[[jax.Array, jax.Array], int]:
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

        return distance_fn, (0,)
        

    def to_tensor(self, dtype = jnp.int32) -> jax.Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(dtype)

    def distance(self, other: 'TreeOperandGenome', type:str = 'euclidean',dtype = jnp.float32) -> float:
        """Calculate the distance to another genome instance."""
        if not isinstance(other, TreeOperandGenome):
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
    
    def clone(self, deep: bool = True) -> 'TreeOperandGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            n_operands_per_node=self.n_operands_per_node,
            maximum_depth=self.maximum_depth,
            n_features_dataset=self.n_features_dataset,
            random_init=False,
        )
        
        # JAX arrays are immutable, so we can share them
        new_genome.genome = self.genome
        new_genome._is_valid = self._is_valid
        new_genome.jax_random_key = self.jax_random_key
        
        return new_genome

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor."""
        if tensor.shape != (self.maximum_depth, self.n_operands_per_node):
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with array_shape {(self.maximum_depth, self.n_operands_per_node)}")

        self.genome = tensor
        self.invalidate()
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    def __str__(self) -> str:
        """String representation of the genome."""
        try:
            return f"{self.genome}"
        except Exception:
            return f"{type(self).__name__} invalid_state)"

    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        try:
            return (f"{self.__class__.__name__}("
                   f"n_operands_per_node={self.n_operands_per_node}, "
                   f"maximum_depth={self.maximum_depth}, "
                   f"n_features_dataset={self.n_features_dataset}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.genome()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(invalid_state)"