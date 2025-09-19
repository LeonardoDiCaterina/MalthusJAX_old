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



@jax.jit
def normal_dist(key, shape, mean=0.0, std=1.0):
    return jar.normal(key, shape) * std + mean

@jax.jit
def t_dist(key, shape, df=10, mean=0.0, std=1.0):
    return jar.t(key, df, shape) * std + mean


class RealGenome_het(AbstractGenome):
    """
    Real-valued genome with heterogeneous distribution for each value of the tensor

    Represents candidate solutions as real-valued arrays.
    """

    def __init__(self,
                 array_shape: Tuple[int, ...] = (10,),
                 distributions: List[Callable[..., Any]] = None,
                 distribution_params: List[Dict[str, Any]] = None,
                 random_init: bool = False,
                 random_key: Optional[int] = None,
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize real genome.
        
        Args:
            array_shape: Shape of the real-valued array
            distributions: List of distribution functions for each element
            distribution_params: List of parameters for each distribution
            random_init: Whether to randomly initialize
            random_key: Random seed
            compatibility: Compatibility constraints
            **kwargs: Additional metadata
        """
        self.array_shape = array_shape
        array_size = 1
        for dim in array_shape:
            array_size *= dim
        self.array_size = array_size
        if distributions is None:
            # it becames automaticlly a list of gaussian distributions
            # multiply all the elements of the shape to get the total size

            distributions = [normal_dist] * self.array_size
        if distribution_params is None:
            # it becames automaticlly a list of gaussian distributions centered in 0 with std 1
            distribution_params = [{'mean': 0.0, 'std': 1.0}] * self.array_size
        assert self.array_size > 0, "array_size must be a positive integer"
        assert len(distributions) == self.array_size, "Length of distributions must match array_size"
        assert len(distribution_params) == self.array_size, "Length of distribution_params must match array_size"
        self.distributions = distributions
        self.distribution_params = distribution_params

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
            'array_shape': array_shape,
            'distributions': distributions,
            'distribution_params': distribution_params,
        }
        super().__init__(
            random_init=random_init, 
            random_key=random_key, 
            compatibility=compatibility,
            **kwargs
        )

    def _random_init(self) -> None:
        """Randomly initialize the genome as a binary array."""
        # split the key for each element
        keys = jar.split(self.jax_key, self.array_size)
        #update the key
        self.jax_key = jar.split(self.jax_key, 1)[0]
        # using jax vmap to vectorize the operation
        def sample_element(key, distribution, params):
            return distribution(key, shape=(1,) ** params).astype(jnp.float32)
        self.genome = jax.vmap(sample_element)(keys, jnp.array(self.distributions), jnp.array(self.distribution_params))
        self.invalidate()  # Mark as needing validation
        
    def _validate(self) -> bool:
        """Validate that genome contains only values within the specified range."""
        if not hasattr(self, 'genome'):
            print(f"hasattr(self, 'genome'){hasattr(self, 'genome')}")
            return False
            
        # Check shape
        if self.genome.shape != (self.array_shape,):
            print(f"{self.genome} != {self.array_shape,}")
            return False
        
        self._is_valid = True
        return self._is_valid
    
    
    # === JAX JIT Compatibility ===
    @classmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:  
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        array_shape = genome_init_params.get('array_shape', (10,))
        distributions = genome_init_params.get('distributions')
        distribution_params = genome_init_params.get('distribution_params')
        def init_fn(random_key: Optional[int] = None) -> jnp.ndarray:
            if random_key is None:
                jax_key = jar.PRNGKey(0)
            elif isinstance(random_key, int):
                jax_key = jar.PRNGKey(random_key)
            else:
                jax_key = random_key
            # split the key for each element
            array_size = 1
            for dim in array_shape:
                array_size *= dim
            keys = jar.split(jax_key, array_size)
            # using jax vmap to vectorize the operation
            def sample_element(key, distribution, params):
                return distribution(key, shape=(1,) ** params).astype(jnp.float32)
            genome = jax.vmap(sample_element)(keys, jnp.array(distributions), jnp.array(distribution_params))
            return genome
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
        pass

    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        return self.genome.astype(jnp.float32)  # Ensure tensor is float32

    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Dict[str, Any],
                   **kwargs: Any) -> 'RealGenome_het':
        """Create a RealGenome_het from a JAX tensor."""
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

    def distance(self, other: 'RealGenome_het') -> float:
        """Calculate Euclidean distance between two real genomes."""
        if not isinstance(other, RealGenome_het):
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
            'distributions': self.distributions,
            'distribution_params': self.distribution_params,
            'metadata': self._metadata,
            'compatibility': self.compatibility
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'RealGenome_het':
        """JAX tree unflattening support."""
        genome_array = children[0]
        
        new_genome = cls(
            array_size=aux_data['array_size'],
            distributions=aux_data['distributions'],
            distribution_params=aux_data['distribution_params'],
            compatibility=aux_data.get('compatibility')
        )
        new_genome.genome = genome_array
        new_genome._metadata = aux_data.get('metadata', {})
        
        return new_genome

    def clone(self, deep: bool = True) -> 'RealGenome_het':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            Note: JAX arrays are immutable, so deep/shallow makes no difference.
        """
        new_genome = self.__class__(
            array_size=self.array_size,
            distributions=self.distributions,
            distribution_params=self.distribution_params,
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
                   f"distributions={self.distributions}, "
                   f"distribution_params={self.distribution_params}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.semantic_key()[:10]}...')")       
            
        except Exception:
            return f"{self.__class__.__name__}(array_size={self.array_size}, invalid_state)"


# Register with JAX for tree operations
jax.tree_util.register_pytree_node(
    RealGenome_het,
    RealGenome_het.tree_flatten,
    RealGenome_het.tree_unflatten
)