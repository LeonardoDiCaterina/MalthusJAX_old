"""
Abstract base classes for genome representations in MalthusJAX.

This module defines the fundamental genome abstractions that encode candidate solutions
for evolutionary algorithms. All genome types must implement JAX tensor interfaces
for efficient batch operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional, Dict, Tuple
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore
import jax  # type: ignore
from ..base import JAXTensorizable

import functools

from dataclasses import dataclass

@dataclass(frozen=True)
class AbstractGenomeConfig(ABC):
    """Base config class for all genome types."""
    
    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AbstractGenomeConfig':
        """Create a config instance from a dictionary."""
        pass
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config instance to a dictionary."""
        pass

# Register as PyTree with no children (all static)
jax.tree_util.register_pytree_node(
    AbstractGenomeConfig,
    lambda obj: ([], obj.__dict__),  # no children, all aux
    lambda aux, cls: cls(**aux)  # reconstruct from aux
)
                                                        
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
                 random_key: Optional[jnp.ndarray] = None,
                 **kwargs: Any):
        """
        Initialize genome base class.

        Args:
            random_init: Whether to randomly initialize the genome
            random_key: Random seed for initialization
            **kwargs: Additional genome-specific metadata
        """
        
        self._metadata: Dict[str, Any] = kwargs
        self._is_valid: Optional[bool] = None
        # init the JAXTensorizable
        JAXTensorizable.__init__(self, random_key=random_key)
        self._genome_config = self.get_config()

        if random_init:
            self._random_init()
            #if not self.is_valid:
            #    raise ValueError(f"Random initialization produced invalid genome: {self.to_tensor()}")
        self._fitness: Optional[float] = None  # Fitness value, to be set externally

    @property
    def size(self) -> int:
        """Get the size of the genome tensor."""
        return self.to_tensor().size

    @property
    def shape(self) -> Tuple:
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
        
    @property
    def genome_config(self) -> AbstractGenomeConfig:
        """Get the genome configuration."""
        return self._genome_config
    
    @genome_config.setter
    def genome_config(self, value: Any) -> None:
        """Set the genome configuration."""
        if isinstance(value, Dict):
            value = self.get_config().from_dict(value)
        elif not isinstance(value, AbstractGenomeConfig):
            raise ValueError("genome_config must be an AbstractGenomeConfig instance or a dict")
        self._genome_config = value

    # === JAX JIT Compatibility abstractions ===
    
    @abstractmethod
    def to_tensor(self) -> Array:
        """Convert the genome to a JAX tensor."""
        pass
    
    @abstractmethod
    def get_config(self) -> AbstractGenomeConfig:
        """Get the configuration object for this genome."""
        pass
        
    @classmethod
    @abstractmethod
    def get_random_initialization_pure_from_config(cls, config: AbstractGenomeConfig) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive a random key and return a tensor."""
        pass
    
    def get_random_initialization_pure(self) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compilable function for random genome initialization that will receive a random key and return a tensor."""
        return self.get_random_initialization_pure_from_config(self.genome_config)

    def _random_init(self) -> None:
        """Initialize genome with random values."""
        init_fn = self.get_random_initialization_pure_from_config(self.genome_config)
        random_key = self.random_key # This uses the property which splits the key
        if random_key is None:
            raise ValueError("Random key is not set for random_init")
        tensor = init_fn(random_key)
        self.update_from_tensor(tensor, validate=True)

    
    @classmethod
    @abstractmethod
    def get_validation_pure_from_config(cls, config: AbstractGenomeConfig) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable function that returns a boolean indicating if genome is valid."""
        pass
    
    def get_validation_pure(self) -> Callable[[jax.Array], bool]:
        """Get JIT-compilable function that returns a boolean indicating if genome is valid."""
        return self.get_validation_pure_from_config(self.genome_config)

    @abstractmethod
    def _validate(self) -> bool:
        """Validate the genome's structure and constraints."""
        validation_fn = self.get_validation_pure()
        return bool(validation_fn(self.to_tensor()))
    
    @classmethod
    @abstractmethod
    def get_distance_pure_from_config(self, config: AbstractGenomeConfig) -> Callable[[jax.Array, jax.Array], float]:
        """Get JIT-compilable function to compute distance between two genomes."""
        pass
    
    def get_distance_pure(self) -> Callable[[jax.Array, jax.Array], float]:
        """Get JIT-compilable function to compute distance between two genomes."""
        return self.get_distance_pure_from_config(self.genome_config)
    
    def _distance(self, other: 'AbstractGenome') -> float:
        """Compute distance to another genome."""
        if not isinstance(other, type(self)):
            raise TypeError(f"Distance can only be computed between type {type(self)} instances, got type {type(other)}")
        distance_fn, _ = self.get_distance_pure()
        return float(distance_fn(self.to_tensor(), other.to_tensor()))

    
    @classmethod
    @abstractmethod
    def get_autocorrection_pure_from_config(cls, config: AbstractGenomeConfig) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compilable function that turns invalid genomes into valid ones."""
        pass
    
    def get_autocorrection_pure(self) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compilable function that turns invalid genomes into valid ones."""
        return self.get_autocorrection_pure_from_config(self.genome_config)
    
    def _autocorrect(self) -> None:
        """Autocorrect the genome to make it valid."""
        correction_fn, _ = self.get_autocorrection_pure()
        tensor = correction_fn(self.to_tensor())
        self.update_from_tensor(tensor, validate=True)

    @classmethod
    @abstractmethod
    def get_genome_config_class(cls) -> Any:
        """Get the genome config class associated with this genome type."""
        pass
    
    
    
    @classmethod
    def from_tensor_from_config(cls, 
                tensor: Array,
                config: AbstractGenomeConfig,
                **kwargs: Any) -> 'AbstractGenome':
        """Create a AbstractGenome from a JAX tensor."""
        # Extract parameters from context if available
        if tensor.shape != config.array_shape:
            raise ValueError(f"Tensor shape {tensor.shape} doesn't match config array_shape {config.array_shape}")
        
        # Create instance without random initialization
        new_genome = cls(
            **config.to_dict(),  # Unpack the genome initialization parameters 
            random_init=False,
            **kwargs
        )
        
        new_genome.genome = tensor
        # Set the genome tensor

        # Validate the result
        if not new_genome.is_valid:
            raise ValueError(f"Genome created from tensor is not valid: {new_genome.to_tensor()}")
            
        return new_genome
        
    def from_tensor(self, tensor: Array, **kwargs: Any) -> 'AbstractGenome':
        """Create a AbstractGenome from a JAX tensor."""
        return self.from_tensor_from_config(tensor, self.genome_config, **kwargs)

    @abstractmethod
    def semantic_key(self) -> str:
        """Generate a unique key for the genome."""
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
        if tensor.shape != self.array_shape:
            raise ValueError(f"Tensor shape {tensor.shape} incompatible with genome shape {self.array_shape}")
        
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

    def __ne__(self, other: object) -> bool:
        """Check if two genomes are not equal."""
        return not self.__eq__(other)

    def __sub__(self, other: 'AbstractGenome') -> float:
        """Calculate distance between two genomes."""
        if not isinstance(other, AbstractGenome):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        return self._distance(other)

    def __hash__(self) -> int:
        """Hash based on semantic key."""
        return hash(self.semantic_key())

    def __str__(self) -> str:
        """Safe string representation."""
        try:
            return f"{self.genome}(array_shape={self.array_shape}, valid={self.is_valid})"  
        except Exception:
            return f"{self.__class__.__name__}(invalid_state)"

    def __repr__(self) -> str:
        """Detailed string representation."""
        try:
            return (f"{self.__class__.__name__}("
                   f"size={self.size}, "
                   f"valid={self.is_valid}, "
                   f"semantic_key='{self.semantic_key()[:20]}...')")
        except Exception:
            return f"{self.__class__.__name__}(invalid_state)"