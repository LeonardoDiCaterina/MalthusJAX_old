"""
Core abstract base classes for MalthusJAX.

This module defines the fundamental abstractions that all components inherit from,
including JAX tensor interfaces, compatibility systems, and core evolutionary concepts.
"""

from abc import ABC, abstractmethod
from typing import Any, Set, Final, Protocol, TypeVar, Dict, Optional, Union
from jax import Array  # type: ignore

# Standardized problem type registry
class ProblemTypes:
    """
    Registry of standardized problem types to ensure consistency
    and avoid typos across the framework.
    """

    # Optimization problems
    CONTINUOUS_OPTIMIZATION: Final[str] = "continuous_optimization"
    DISCRETE_OPTIMIZATION: Final[str] = "discrete_optimization"
    COMBINATORIAL_OPTIMIZATION: Final[str] = "combinatorial_optimization"

    # Neural evolution
    NEUROEVOLUTION: Final[str] = "neuroevolution"
    NEAT: Final[str] = "neat"

    # Genetic programming
    SYMBOLIC_REGRESSION: Final[str] = "symbolic_regression"
    GENETIC_PROGRAMMING: Final[str] = "genetic_programming"

    # Game playing
    GAME_PLAYING: Final[str] = "game_playing"
    REINFORCEMENT_LEARNING: Final[str] = "reinforcement_learning"

    # Multi-objective
    MULTI_OBJECTIVE: Final[str] = "multi_objective"

    # Custom/experimental
    CUSTOM: Final[str] = "custom"

    @classmethod
    def get_all_types(cls) -> Set[str]:
        """Get all registered problem types."""
        return {
            value for name, value in cls.__dict__.items()
            if isinstance(value, str) and not name.startswith('_')
        }

    @classmethod
    def is_valid_type(cls, problem_type: str) -> bool:
        """Check if a problem type is in the registry."""
        return problem_type in cls.get_all_types()


class Compatibility:
    """
    Manages compatibility between different evolutionary components.

    This class determines whether genomes, fitness functions, and solutions
    can work together based on problem types and constraints.
    """

    def __init__(self, problem_type: str, **constraints: Any):
        """
        Initialize compatibility checker.

        Args:
            problem_type: String identifier for the problem domain (must be from ProblemTypes)
            **constraints: Additional compatibility constraints

        Raises:
            ValueError: If problem_type is not in the registry
        """
        if not ProblemTypes.is_valid_type(problem_type):
            valid_types = sorted(ProblemTypes.get_all_types())
            raise ValueError(
                f"Invalid problem type '{problem_type}'. "
                f"Must be one of: {valid_types}"
            )

        self.problem_type = problem_type
        self.constraints = constraints

    def is_compatible(self, other: 'Compatibility') -> bool:
        """
        Check if this compatibility is compatible with another.

        Args:
            other: Another compatibility object to check against

        Returns:
            True if compatible, False otherwise
        """
        # Same problem type = compatible for now
        # Additional constraint checking can be implemented here
        return self.problem_type == other.problem_type

    def __eq__(self, other: object) -> bool:
        """Check equality based on problem type and constraints."""
        if not isinstance(other, Compatibility):
            return False
        return (self.problem_type == other.problem_type and
                self.constraints == other.constraints)

    def __ne__(self, other: object) -> bool:
        """Check inequality based on problem type and constraints."""
        return not self.__eq__(other)

    def __str__(self) -> str:
        return f"Compatibility(type={self.problem_type}, constraints={self.constraints})"

    def __repr__(self) -> str:
        return self.__str__()


# Type variable for generic serialization
T = TypeVar('T')


class SerializationContext:
    """Context information needed to reconstruct objects from tensors."""
    
    def __init__(self,
                 genome_class: Optional[type] = None,
                 genome_init_params: Optional[Dict[str, Any]] = None,
                 solution_class: Optional[type] = None,
                 fitness_transform: Optional[callable] = None,
                 compatibility: Optional[Compatibility] = None,
                 **metadata: Any):
        self.genome_class = genome_class
        self.genome_init_params = genome_init_params or {}
        self.solution_class = solution_class
        self.fitness_transform = fitness_transform
        self.compatibility = compatibility
        self.metadata = metadata
    
    def for_genome(self) -> Dict[str, Any]:
        """Get parameters for genome reconstruction."""
        return {
            **self.genome_init_params,
            'compatibility': self.compatibility,
            **self.metadata
        }
    
    def for_solution(self) -> Dict[str, Any]:
        """Get parameters for solution reconstruction."""
        return {
            'genome_class': self.genome_class,
            'fitness_transform': self.fitness_transform,
            **self.metadata
        }


class TensorSerializable(Protocol):
    """Standard protocol for tensor serialization."""
    
    def to_tensor(self) -> Array:
        """Convert object to JAX tensor."""
        pass
    
    @classmethod
    def from_tensor(cls: type[T], 
                   tensor: Array,
                   context: Optional[SerializationContext] = None,
                   **kwargs: Any) -> T:
        """Create object from tensor with context."""
        pass
    
    def get_serialization_context(self) -> SerializationContext:
        """Get context needed for reconstruction."""
        pass


class JAXTensorizable(TensorSerializable, ABC):
    """
    Abstract base class for objects that can be converted to/from JAX tensors.

    This enables efficient batch operations and JIT compilation across all
    MalthusJAX components. Inherits from TensorSerializable protocol.
    """

    @abstractmethod
    def to_tensor(self) -> Array:
        """Convert this object to a JAX tensor representation."""
        pass

    @classmethod
    @abstractmethod
    def from_tensor(cls, 
                   tensor: Array,
                   context: Optional[SerializationContext] = None,
                   **kwargs: Any) -> 'JAXTensorizable':
        """Create an instance from a JAX tensor representation."""
        pass

    @abstractmethod
    def get_serialization_context(self) -> SerializationContext:
        """Get context needed for reconstruction."""
        pass