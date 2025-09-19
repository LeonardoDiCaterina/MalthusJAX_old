"""
Permutation Categorical implementation for MalthusJAX.

This module provides a simple permutation categorical representation suitable for
permutation optimization problems and as a reference implementation.
"""

from typing import Any, Callable, Optional, Dict, Tuple, List
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenome
from ..base import Compatibility, ProblemTypes, SerializationContext


class CategoricalGenome(AbstractGenome):
    """
    Categorical genome implementation for categorical optimization problems.

    Represents candidate solutions as categorical distributions.
    """

    def __init__(self,
                 array_size: int = 2,
                 num_categories: int = 2,
                 random_init: bool = False,
                 random_key: Optional[int] = jar.PRNGKey(0),
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize permutation genome.
        Args:
            array_size: Length of the array
            num_categories: Number of categories for the categorical distribution
            random_init: Whether to randomly initialize
            random_key: Random seed
            compatibility: Compatibility constraints
            **kwargs: Additional metadata
        """
        assert array_size > 0, "array_size must be positive"
        assert num_categories > 0, "num_categories must be positive"
        self.num_categories = num_categories
        self.array_size = array_size
        self.random_init = random_init
        self.random_key = random_key
        self.compatibility = compatibility
        self._init_params = kwargs
        super().__init__(random_init=random_init, random_key=random_key, compatibility=compatibility, **kwargs)


        # Handle random key conversion
        if isinstance(random_key, int):
            self.random_key = jar.PRNGKey(random_key)
        else:
            self.random_key = random_key

        # Set default compatibility for permutation problems
        if compatibility is None:
            compatibility = Compatibility(problem_type=ProblemTypes.DISCRETE_OPTIMIZATION)

        # Store initialization parameters for serialization
        self._init_params = {
            'num_categories': num_categories,
            'array_size': array_size
        }
        
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            compatibility=compatibility,
            **kwargs
        )

    def _random_init(self) -> None:
        """Randomly initialize the genome as a permutation array."""
        self.random_key, subkey = jar.split(self.random_key)
        self.genome = jar.randint(subkey, (self.array_size,), 0, self.num_categories, dtype=jnp.int32)
        self.invalidate()
    
    def _validate(self) -> bool:
        """Validate that genome contains only values within the specified range."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != (self.array_size,):
            print(f"{self.genome} != {self.array_size,}")
            return False

        # Check that all elements are within the range [0, num_categories)
        if not jnp.all((self.genome >= 0) & (self.genome < self.num_categories)):
            print(f"Genome values {self.genome} out of range [0, {self.num_categories})")
            return False


        self._is_valid = True
        return self._is_valid
    
    # === JAX JIT Compatibility ===
    @classmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        
        n_categories = genome_init_params.get('num_categories', 2)
        len_array = genome_init_params.get('array_size', 2)

        def init_fn(random_key: Optional[int] = None) -> jnp.ndarray:
    
            return jar.randint(random_key, (len_array,), 0, n_categories, dtype=jnp.int32)
        
        return jax.jit(init_fn)

    @classmethod
    def get_distance_jit(cls) -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compiled distance function between two solutions."""
        def distance_fn(gen1: jax.Array, gen2: jax.Array) -> int:
            diff = gen1 - gen2
            return jnp.count_nonzero(diff)
        return jax.jit(distance_fn)

    @classmethod
    def get_autocorrection_jit(cls, genome_init_params) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        
        n_categories = genome_init_params.get('num_categories', 2)
        len_array = genome_init_params.get('array_size', 2)

        def correction_fn(sol: jax.Array) -> jax.Array:
            # Clip values to be within the valid range [0, n_categories - 1]
            return jnp.clip(sol[:len_array], 0, n_categories - 1)
        return jax.jit(correction_fn)

    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.int32)

    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Dict[str, Any],
                   **kwargs: Any) -> 'CategoricalGenome':
        """Create a CategoricalGenome from a JAX tensor."""
        # Extract parameters from context if available
        
        # Create instance without random initialization
        new_genome = cls(
            **genome_init_params,  # Unpack the genome initialization parameters 
            random_init=False,
            **kwargs
        )
        # Set the genome tensor
        tensor = jnp.clip(tensor, 0, new_genome.num_categories - 1)  # Ensure values are valid
        new_genome.genome = tensor.astype(jnp.int32)  # Ensure tensor is int32
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

    def distance(self, other: 'CategoricalGenome') -> float:
        """Calculate Euclidean distance between two categorical genomes."""
        if not isinstance(other, CategoricalGenome):
            return float('inf')

        if self.array_size != other.array_size or self.num_categories != other.num_categories:
            return float('inf')

        # count number of differing positions
        diff = self.genome - other.genome
        return jnp.count_nonzero(diff)

    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
    
        genome_bytes = self.genome.tobytes()
        hash_obj = hashlib.sha256(genome_bytes)
        return hash_obj.hexdigest()

    def tree_flatten(self) -> Tuple[List[Array], Dict[str, Any]]:
        """JAX tree flattening support."""
        children = [self.genome]
        aux_data = {
            'array_size': self.array_size,
            'num_categories': self.num_categories,
            'metadata': self._metadata,
            'compatibility': self.compatibility
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'CategoricalGenome':
        """JAX tree unflattening support."""
        genome_array = children[0]
        
        new_genome = cls(
            array_size=aux_data['array_size'],
            num_categories=aux_data['num_categories'],
            random_init=False,
            compatibility=aux_data.get('compatibility')
        )
        new_genome.genome = genome_array
        new_genome._metadata = aux_data.get('metadata', {})
        
        return new_genome

    def clone(self, deep: bool = True) -> 'CategoricalGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_size=self.array_size,
            num_categories=self.num_categories,
            random_init=False,
            compatibility=self.compatibility,
            **self._metadata.copy()
        )
        
        # JAX arrays are immutable, so we can share them
        new_genome.genome = self.genome
        new_genome._is_valid = self._is_valid
        new_genome.random_key = self.random_key
        
        return new_genome

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor."""
        if tensor.shape != (self.array_size,):
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with array size {self.array_size}")

        self.genome = tensor.astype(jnp.int32)
        self.invalidate()
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    def __str__(self) -> str:
        """String representation of the genome."""
        try:
            return f"{type(self).__name__}(genome={self.genome}, valid={self.is_valid})"
        except Exception:
            return 'invalid genome'
    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        try:
            return (f"{self.__class__.__name__}("
                   f"array_size={self.array_size}, "
                   f"num_categories={self.num_categories}, "
                   f"valid={self.is_valid}, "
                   f"genome={self.genome}, "
                   f"semantic_key='{self.semantic_key()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_size={self.array_size}, invalid_state)"


# Register with JAX for tree operations
jax.tree_util.register_pytree_node(
    CategoricalGenome,
    CategoricalGenome.tree_flatten,
    CategoricalGenome.tree_unflatten
)