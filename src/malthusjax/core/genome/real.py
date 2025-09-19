"""
Binary genome implementation for MalthusJAX.

This module provides a simple binary string genome representation suitable for
binary optimization problems and as a reference implementation.
"""

from typing import Any, Callable, Optional, Dict, Tuple, List
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenome
from ..base import Compatibility, ProblemTypes, SerializationContext


class RealGenome(AbstractGenome):
    """
    Real-valued genome implementation for real-valued optimization problems.

    Represents candidate solutions as real-valued arrays.
    """

    def __init__(self,
                 array_size: int,
                 minval: float,
                 maxval: float,
                 random_init: bool = False,
                 random_key: Optional[int] = None,
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize real genome.
        
        Args:
            array_size: Length of the real-valued array
            minval: Minimum value for the real numbers
            maxval: Maximum value for the real numbers
            random_init: Whether to randomly initialize
            random_key: Random seed
            compatibility: Compatibility constraints
            **kwargs: Additional metadata
        """
        self.array_size = array_size
        assert array_size > 0, "array_size must be a positive integer"
        assert minval < maxval, "minval must be less than maxval"
        self.minval = minval
        self.maxval = maxval

        # Handle random key conversion
        if random_key is None:
            self.jax_key = jar.PRNGKey(0)
        elif isinstance(random_key, int):
            self.jax_key = jar.PRNGKey(random_key)
        else:
            self.jax_key = random_key
            
        # Set default compatibility for binary problems
        if compatibility is None:
            compatibility = Compatibility(problem_type=ProblemTypes.DISCRETE_OPTIMIZATION)
            
        # Store initialization parameters for serialization
        self._init_params = {
            'array_size': array_size,
            'minval': minval,
            'maxval': maxval
        }
        
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            compatibility=compatibility,
            **kwargs
        )

    def _random_init(self) -> None:
        """Randomly initialize the genome as a binary array."""
        self.jax_key, subkey = jar.split(self.jax_key)
        self.genome = jar.uniform(
            subkey, 
            shape=(self.array_size,), 
            minval=self.minval, 
            maxval=self.maxval
        ).astype(jnp.float32)

    def _validate(self) -> bool:
        """Validate that genome contains only values within the specified range."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != (self.array_size,):
            print(f"{self.genome} = {self.array_size,}")
            return False
            
        # Check that all elements are within the range [minval, maxval]
        if not jnp.all((self.genome >= self.minval) & (self.genome <= self.maxval)):
            print(f"Genome values {self.genome} out of range [{self.minval}, {self.maxval}]")
            return False
        
        self._is_valid = True
        return self._is_valid
    
    
    # === JAX JIT Compatibility ===
    @classmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:  
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        array_size = genome_init_params.get('array_size')
        minval = genome_init_params.get('minval')
        maxval = genome_init_params.get('maxval')
        def init_fn(random_key: Optional[int]) -> jnp.ndarray:
            return jar.uniform(
                random_key, 
                shape=(array_size,), 
                minval=minval, 
                maxval=maxval
            ).astype(jnp.float32)
        return jax.jit(init_fn)  

    @classmethod
    def get_distance_jit(self) -> Callable[[jax.Array, jax.Array], float]:
        """Get JIT-compiled function to compute distance between two genomes."""
        @jax.jit
        def distance_fn(genome1: jax.Array, genome2: jax.Array) -> float:
            return jnp.sqrt(jnp.sum(jnp.square(genome1 - genome2)))
        return distance_fn

    @classmethod
    def get_autocorrection_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        minval = genome_init_params['minval']
        maxval = genome_init_params['maxval']
        array_size = genome_init_params['array_size']
        def validation_fn(sol: jax.Array) -> jax.Array:
            # Clip values to be within the valid range
            return jnp.clip(sol[:array_size], minval, maxval)
        return jax.jit(validation_fn)

    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.float32)  # Ensure tensor is float32

    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Dict[str, Any],
                   **kwargs: Any) -> 'RealGenome':
        """Create a RealGenome from a JAX tensor."""
        # Extract parameters from context if available
        
        # Create instance without random initialization
        new_genome = cls(
            **genome_init_params,  # Unpack the genome initialization parameters 
            random_init=False,
            **kwargs
        )
        # Set the genome tensor
        new_genome.genome = tensor.astype(jnp.float32)  # Ensure tensor is float32
        # Validate
        if not new_genome.is_valid:
            #raise ValueError(f"Genome created from tensor {new_genome.to_tensor() } is not valid")
            print(f"Warning: Genome created from tensor {new_genome.to_tensor() } is not valid")
        return new_genome

    def get_serialization_context(self) -> SerializationContext:
        """Get context needed to reconstruct this genome."""
        return SerializationContext(
            genome_class=type(self),
            genome_init_params=self._init_params,
            compatibility=self.compatibility,
            **self.metadata
        )

    def distance(self, other: 'RealGenome') -> float:
        """Calculate Euclidean distance between two real genomes."""
        if not isinstance(other, RealGenome):
            return float('inf')
            
        if self.array_size != other.array_size:
            return float('inf')

        return float(jnp.sqrt(jnp.sum(jnp.square(self.to_tensor() - other.to_tensor()))))


    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
        # Convert real-valued array to string for consistent hashing
        real_string = ','.join(str(float(x)) for x in self.genome.tolist())
        return hashlib.md5(real_string.encode()).hexdigest()

    def tree_flatten(self) -> Tuple[List[Array], Dict[str, Any]]:
        """JAX tree flattening support."""
        children = [self.genome]
        aux_data = {
            'array_size': self.array_size,
            'minval': self.minval,
            'maxval': self.maxval,
            'metadata': self._metadata,
            'compatibility': self.compatibility
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'RealGenome':
        """JAX tree unflattening support."""
        genome_array = children[0]
        
        new_genome = cls(
            array_size=aux_data['array_size'],
            minval=aux_data['minval'],
            maxval=aux_data['maxval'],
            random_init=False,
            compatibility=aux_data.get('compatibility')
        )
        new_genome.genome = genome_array
        new_genome._metadata = aux_data.get('metadata', {})
        
        return new_genome

    def clone(self, deep: bool = True) -> 'RealGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_size=self.array_size,
            minval=self.minval,
            maxval=self.maxval,
            random_init=False,
            compatibility=self.compatibility,
            **self._metadata.copy()
        )
        
        # JAX arrays are immutable, so we can share them
        new_genome.genome = self.genome
        new_genome._is_valid = self._is_valid
        new_genome.jax_key = self.jax_key
        
        return new_genome

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor."""
        if tensor.shape != (self.array_size,):
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with array_size {self.array_size}")

        self.genome = tensor.astype(jnp.float32)
        self.invalidate()
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    def __str__(self) -> str:
        """String representation of the genome."""
        try:
            return f"{type(self).__name__}(size={self.array_size}, valid={self.is_valid})"
        except Exception:
            return f"{type(self).__name__}(size={self.array_size}, invalid_state)"

    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        try:
            return (f"{self.__class__.__name__}("
                   f"array_size={self.array_size}, "
                   f"p={self.p}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.semantic_key()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_size={self.array_size}, invalid_state)"


# Register with JAX for tree operations
jax.tree_util.register_pytree_node(
    RealGenome,
    RealGenome.tree_flatten,
    RealGenome.tree_unflatten
)