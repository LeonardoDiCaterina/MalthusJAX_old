"""
Abstract base classes for genome representations in MalthusJAX.

This module defines the fundamental genome abstractions that encode candidate solutions
for evolutionary algorithms. All genome types must implement JAX tensor interfaces
for efficient batch operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore
import jax  # type: ignore
from ..base import JAXTensorizable, Compatibility, ProblemTypes, SerializationContext

class AbstractGenome(JAXTensorizable, ABC):
    """
    Abstract base class for all genome representations.

    A genome encodes a candidate solution to the optimization problem.
    It serves as a passive data container with validation, distance calculation,
    and semantic key functionality for efficient set operations.

    Genetic operations (mutation, crossover) are handled by Level 2 operators
    that work on batches of genomes for efficiency.
    """

    def __init__(self, 
                 random_init: bool = False, 
                 random_key: Optional[int] = None, 
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize genome with compatibility information.

        Args:
            random_init: Whether to randomly initialize the genome
            random_key: Random seed for initialization
            compatibility: Compatibility object defining problem constraints
            **kwargs: Additional genome-specific metadata
        """
        # Set default compatibility if none provided
        if compatibility is None:
            compatibility = Compatibility(problem_type=ProblemTypes.DISCRETE_OPTIMIZATION)
        
        self._compatibility = compatibility
        self._metadata: Dict[str, Any] = kwargs
        self._is_valid: Optional[bool] = None
        self.random_key = random_key
        
        if random_init:
            self._random_init()
            if not self.is_valid:
                raise ValueError(f"Random initialization produced invalid genome: {self.to_tensor()}")
        self._fitness: Optional[float] = None  # Fitness value, to be set externally

    @property
    def size(self) -> int:
        """Get the size of the genome tensor."""
        return self.to_tensor().size

    @property
    def shape(self) -> tuple:
        """Get the shape of the genome tensor."""
        return self.to_tensor().shape

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get genome metadata."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: Dict[str, Any]) -> None:
        """Set genome metadata."""
        self._metadata = value

    @property
    def compatibility(self) -> Compatibility:
        """Get compatibility information."""
        return self._compatibility

    @compatibility.setter
    def compatibility(self, value: Compatibility) -> None:
        """Set compatibility information."""
        self._compatibility = value

    @property
    def is_valid(self) -> bool:
        """Check if the genome is valid."""
        if self._is_valid is None:
            self._is_valid = self._validate()
        return self._is_valid
    
    @property
    def fitness(self) -> Optional[float]:
        """Get the fitness value of the genome."""
        return self._fitness
    @fitness.setter
    def fitness(self, value: float) -> None:
        """Set the fitness value of the genome."""
        self._fitness = value

    def invalidate(self) -> None:
        """Invalidate cached validation result."""
        self._is_valid = None

    # === Abstract methods that subclasses must implement ===
    
    @classmethod
    @abstractmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        pass

    @abstractmethod
    def _random_init(self) -> None:
        """Initialize genome with random values."""
        pass

    @abstractmethod
    def _validate(self) -> bool:
        """Validate the genome's structure and constraints."""
        pass

    @abstractmethod
    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        pass
    # === JAX JIT Compatibility abstractions ===
    
    @classmethod
    @abstractmethod
    def get_distance_jit(self) -> Callable[[jax.Array, jax.Array], float]:
        """Get JIT-compiled function to compute distance between two genomes."""
        pass
    
    @classmethod
    @abstractmethod
    def get_autocorrection_jit(cls, genome_init_params: Dict[str, Any]) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        pass

    @classmethod
    @abstractmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: Optional[Dict[str, Any]] = None,
                   **kwargs: Any) -> 'AbstractGenome':
        """Create a genome from a JAX tensor with standardized signature."""
        pass

    @abstractmethod
    def get_serialization_context(self) -> SerializationContext:
        """Get context needed to reconstruct this genome."""
        pass

    @abstractmethod
    def distance(self, other: 'AbstractGenome') -> float:
        """Calculate the distance between two genomes."""
        pass

    @abstractmethod
    def semantic_key(self) -> str:
        """Generate a unique key for the genome."""
        pass

    @abstractmethod
    def tree_flatten(self):
        """JAX tree flattening support."""
        pass

    @classmethod
    @abstractmethod
    def tree_unflatten(cls, aux_data, children):
        """JAX tree unflattening support."""
        pass

    @abstractmethod
    def clone(self, deep: bool = True) -> 'AbstractGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            
        Returns:
            A new genome instance that is a copy of this one.
        """
        pass

    # === Concrete methods ===

    def update_from_tensor(self, tensor: Array, validate: bool = False) -> None:
        """Update the genome data in-place from a tensor.
        
        This method allows for efficient updates to the genome's data without
        needing to create a new instance, generally used for mutations.
        
        Note: This method can be problematic when tracking statistics or 
        metadata across generations.
        
        Args:
            tensor: New genome data as a tensor
            validate: Whether to validate after update
        """
        current_shape = self.to_tensor().shape
        if tensor.shape != current_shape:
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with genome shape {current_shape}")
        
        # This is a placeholder - subclasses should override with proper implementation
        self.invalidate()  # Invalidate cached validation result
        
        if validate and not self.is_valid:
            raise ValueError("Updated genome is invalid")

    # === Standard object methods ===
    
    def __len__(self) -> int:
        """Get the length of the genome tensor."""
        return self.size

    def __eq__(self, other: object) -> bool:
        """Check if two genomes are equal."""
        if not isinstance(other, AbstractGenome):
            return False
        return (self.compatibility == other.compatibility and
                self.semantic_key() == other.semantic_key())

    def __ne__(self, other: object) -> bool:
        """Check if two genomes are not equal."""
        return not self.__eq__(other)

    def __sub__(self, other: 'AbstractGenome') -> float:
        """Calculate distance between two genomes."""
        if not isinstance(other, AbstractGenome):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        return self.distance(other)

    def __hash__(self) -> int:
        """Hash based on semantic key."""
        return hash(self.semantic_key())

    def __str__(self) -> str:
        """Safe string representation."""
        try:
            return f"{self.__class__.__name__}(size={self.size}, valid={self.is_valid})"
        except Exception:
            return f"{self.__class__.__name__}(invalid_state)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        try:
            return (f"{self.__class__.__name__}("
                   f"size={self.size}, "
                   f"compatibility={self.compatibility}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.semantic_key()[:20]}...')")
        except Exception:
            return f"{self.__class__.__name__}(invalid_state)"