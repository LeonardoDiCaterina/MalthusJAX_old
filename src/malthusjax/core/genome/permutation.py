"""
Permutation genome implementation for MalthusJAX.

This module provides a simple permutation genome representation suitable for
permutation optimization problems and as a reference implementation.
"""

from typing import Any, Callable, Optional, Dict, Tuple, List
import jax.numpy as jnp  # type: ignore
import jax # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenome
from ..base import Compatibility, ProblemTypes, SerializationContext


class PermutationGenome(AbstractGenome):
    """
    Permutation genome implementation for permutation optimization problems.

    Represents candidate solutions as permutations of integers.
    """

    def __init__(self,
                 permutation_start: int = 0,
                 permutation_end: int = 2,
                 random_init: bool = False,
                 random_key: Optional[int] = jar.PRNGKey(0),
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize permutation genome.
        Args:
            array_size: Length of the permutation array
            permutation_start: Start of the integer range for permutation (inclusive)
            permutation_end: End of the integer range for permutation (exclusive)
            random_init: Whether to randomly initialize
            random_key: Random seed
            compatibility: Compatibility constraints
            **kwargs: Additional metadata
        """
        self.permutation_start = permutation_start
        self.permutation_end = permutation_end
        self.random_init = random_init
        self.random_key = random_key
        self.compatibility = compatibility
        self._init_params = kwargs
        super().__init__(random_init=random_init, random_key=random_key, compatibility=compatibility, **kwargs)
        assert permutation_end > permutation_start, "permutation_end must be greater than permutation_start"
        assert permutation_end - permutation_start > 1, "permutation range must be at least 2"

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
            'permutation_start': permutation_start,
            'permutation_end': permutation_end,
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
        self.genome = jax.random.permutation(
            subkey, 
            jnp.arange(self.permutation_start, self.permutation_end, dtype=jnp.int32)
        )
        self.invalidate()
    
    def _validate(self) -> bool:
        """Validate that genome contains only values within the specified range."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != (self.permutation_end - self.permutation_start,):
            print(f"{self.genome} = {self.permutation_end - self.permutation_start,}")
            return False
            
        # Check that all elements are within the range [minval, maxval]
        if not jnp.all((self.genome >= self.permutation_start) & (self.genome < self.permutation_end)):
            print(f"Genome values {self.genome} out of range [{self.permutation_start}, {self.permutation_end}]")
            return False
        
        # does not contain duplicates
        unique_elements = jnp.unique(self.genome)
        if unique_elements.shape[0] != self.genome.shape[0]:
            print(f"Genome contains duplicates: {self.genome}")
            return False

        self._is_valid = True
        return self._is_valid
    
    # === JAX JIT Compatibility ===
    @classmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        permutation_start = genome_init_params.get('permutation_start', 0)
        permutation_end = genome_init_params.get('permutation_end', 2)
        assert permutation_end > permutation_start, "permutation_end must be greater than permutation_start"
        assert permutation_end - permutation_start > 1, "permutation range must be at least 2"
        
        def init_fn(random_key: Optional[int]) -> jnp.ndarray:
            return jax.random.permutation(
                random_key, 
                jnp.arange(permutation_start, permutation_end, dtype=jnp.int32)
            )
        return jax.jit(init_fn)
    

    @classmethod
    def get_distance_jit(cls) -> Callable[[jax.Array, jax.Array], int]:
        """Get JIT-compiled function to compute distance between two genomes."""
        @jax.jit
        def distance_fn(genome1: jax.Array, genome2: jax.Array) -> int:
            diff = genome1 - genome2
            return jnp.count_nonzero(diff)
        return distance_fn
    
    @classmethod
    def get_autocorrection_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        return lambda x: x  # Identity for now, as permutations are tricky to auto-correct
        def autovalidation_fn(genome: jax.Array) -> jax.Array:
            # Create a valid permutation by sorting and removing duplicates
            unique_sorted = jnp.unique(jnp.clip(genome, self.permutation_start, self.permutation_end - 1))
            # If there are missing elements, fill them in
            full_range = jnp.arange(cls.permutation_start, cls.permutation_end)
            missing_elements = jnp.setdiff1d(full_range, unique_sorted, assume_unique=True)
            # Combine and truncate to the correct size
            combined = jnp.concatenate([unique_sorted, missing_elements])
            return combined[:cls.permutation_end - cls.permutation_start]
        return jax.jit(autovalidation_fn)

    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.int32)

    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Dict[str, Any],
                   **kwargs: Any) -> 'PermutationGenome':
        """Create a PermutationGenome from a JAX tensor."""
        # Extract parameters from context if available
        
        # Create instance without random initialization
        new_genome = cls(
            **genome_init_params,  # Unpack the genome initialization parameters 
            random_init=False,
            **kwargs
        )
        # Set the genome tensor
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

    def distance(self, other: 'PermutationGenome') -> float:
        """Calculate Euclidean distance between two permutation genomes."""
        if not isinstance(other, PermutationGenome):
            return float('inf')

        if self.permutation_end - self.permutation_start != other.permutation_end - other.permutation_start:
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
            'permutation_start': self.permutation_start,
            'permutation_end': self.permutation_end,
            'metadata': self._metadata,
            'compatibility': self.compatibility
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'PermutationGenome':
        """JAX tree unflattening support."""
        genome_array = children[0]
        
        new_genome = cls(
            permutation_start=aux_data['permutation_start'],
            permutation_end=aux_data['permutation_end'],
            random_init=False,
            compatibility=aux_data.get('compatibility')
        )
        new_genome.genome = genome_array
        new_genome._metadata = aux_data.get('metadata', {})
        
        return new_genome

    def clone(self, deep: bool = True) -> 'PermutationGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            permutation_start=self.permutation_start,
            permutation_end=self.permutation_end,
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
        if tensor.shape != (self.permutation_end - self.permutation_start,):
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with permutation size {self.permutation_end - self.permutation_start}")

        self.genome = tensor.astype(jnp.int32)
        self.invalidate()
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    def __str__(self) -> str:
        """String representation of the genome."""
        try:
            return f"{type(self).__name__}(permutation_start={self.permutation_start}, permutation_end={self.permutation_end} "
        except Exception:
            return 'invalid genome'
    def __repr__(self) -> str:
        """Detailed representation of the genome."""
        try:
            return (f"{self.__class__.__name__}("
                   f"array_size={self.permutation_end - self.permutation_start}, "
                   f"valid={self.is_valid}, "
                   f"permutation_start={self.permutation_start}, "
                   f"permutation_end={self.permutation_end}, "
                   f"genome={self.genome}, "
                   f"semantic_key='{self.semantic_key()[:10]}...')")
        except Exception:
            return f"{self.__class__.__name__}(array_size={self.permutation_end - self.permutation_start}, invalid_state)"


# Register with JAX for tree operations
jax.tree_util.register_pytree_node(
    PermutationGenome,
    PermutationGenome.tree_flatten,
    PermutationGenome.tree_unflatten
)