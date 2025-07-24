"""
Binary genome implementation for MalthusJAX.

This module provides a simple binary string genome representation suitable for
binary optimization problems and as a reference implementation.
"""

from typing import Any, Optional, Dict, Tuple, List
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore 
from jax import random as jar  # type: ignore
import jax  # type: ignore
import hashlib

from .base import AbstractGenome
from ..base import Compatibility, ProblemTypes, SerializationContext


class BinaryGenome(AbstractGenome):
    """
    Binary genome implementation for binary optimization problems.
    
    Represents candidate solutions as binary arrays (0s and 1s).
    """

    def __init__(self,
                 array_size: int,
                 p: float,
                 random_init: bool = False,
                 random_key: Optional[int] = None,
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize binary genome.
        
        Args:
            array_size: Length of the binary array
            p: Probability of 1s during random initialization
            random_init: Whether to randomly initialize
            random_key: Random seed
            compatibility: Compatibility constraints
            **kwargs: Additional metadata
        """
        self.array_size = array_size
        assert 0 <= p <= 1, "p must be between 0 and 1"
        self.p = p
        
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
            'p': p
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
        self.genome = jar.bernoulli(subkey, p=self.p, shape=(self.array_size,))

    def _validate(self) -> bool:
        """Validate that genome contains only 0s and 1s."""
        if not hasattr(self, 'genome'):
            #print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != (self.array_size,):
            print(f"{self.genome.shape} = {self.array_size,}")
            return False
            
        # Check that all elements are 0 or 1
        if not jnp.issubdtype(self.genome.dtype, jnp.integer) and not jnp.issubdtype(self.genome.dtype, jnp.bool_):
            #print("probem counting zeroes")
            return False
            
        return jnp.all(jnp.isin(self.genome, jnp.array([0, 1], dtype=self.genome.dtype)))

    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.int32)  # Use int32 for consistency

    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Dict[str, Any],
                   **kwargs: Any) -> 'BinaryGenome':
        """Create a BinaryGenome from a JAX tensor."""
        # Extract parameters from context if available
        
        # Create instance without random initialization
        print(genome_init_params)
        new_genome = cls(
            **genome_init_params,  # Unpack the genome initialization parameters 
            random_init=False,
            **kwargs
        )
        # Set the genome tensor
        new_genome.genome = tensor.astype(jnp.bool_)
        # Validate
        if not new_genome.is_valid:
            raise ValueError(f"Genome created from tensor {new_genome.to_tensor() } is not valid")
            
        return new_genome

    def get_serialization_context(self) -> SerializationContext:
        """Get context needed to reconstruct this genome."""
        return SerializationContext(
            genome_class=type(self),
            genome_init_params=self._init_params,
            compatibility=self.compatibility,
            **self.metadata
        )

    def distance(self, other: 'BinaryGenome') -> float:
        """Calculate Hamming distance between two binary genomes."""
        if not isinstance(other, BinaryGenome):
            return float('inf')
            
        if self.array_size != other.array_size:
            return float('inf')
            
        return float(jnp.sum(jnp.abs(self.to_tensor() - other.to_tensor())))

    def semantic_key(self) -> str:
        """Generate a semantic key for the genome."""
        # Convert binary array to string for consistent hashing
        binary_string = ''.join(str(int(x)) for x in self.genome.tolist())
        return hashlib.md5(binary_string.encode()).hexdigest()

    def tree_flatten(self) -> Tuple[List[Array], Dict[str, Any]]:
        """JAX tree flattening support."""
        children = [self.genome]
        aux_data = {
            'array_size': self.array_size,
            'p': self.p,
            'metadata': self._metadata,
            'compatibility': self.compatibility
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'BinaryGenome':
        """JAX tree unflattening support."""
        genome_array = children[0]
        
        new_genome = cls(
            array_size=aux_data['array_size'],
            p=aux_data['p'],
            random_init=False,
            compatibility=aux_data.get('compatibility')
        )
        new_genome.genome = genome_array
        new_genome._metadata = aux_data.get('metadata', {})
        
        return new_genome

    def clone(self, deep: bool = True) -> 'BinaryGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_size=self.array_size,
            p=self.p,
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
            
        self.genome = tensor.astype(jnp.bool_)
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
    BinaryGenome,
    BinaryGenome.tree_flatten,
    BinaryGenome.tree_unflatten
)