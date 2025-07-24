"""
Abstract base classes for solution representations in MalthusJAX.

This module defines the Solution class that wraps genomes with fitness evaluation,
lazy computation, transformation pipelines, and efficient comparison operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Callable, Dict, Union
import jax.numpy as jnp  # type: ignore
import jax.random as jar  # type: ignore
from jax import Array  # type: ignore
import jax  # type: ignore
import copy

from ..base import JAXTensorizable, Compatibility, SerializationContext
from ..genome.base import AbstractGenome


class FitnessTransforms:
    """
    Collection of common fitness transformation functions.
    
    These transforms allow converting between maximization/minimization problems,
    applying normalization, handling constraints, and other fitness modifications.
    """
    
    @staticmethod
    def identity(x: float) -> float:
        """No transformation (default)."""
        return x
    
    @staticmethod
    def minimize(x: float) -> float:
        """Convert maximization to minimization problem."""
        return -x
    
    @staticmethod
    def reciprocal(x: float, epsilon: float = 1e-10) -> float:
        """1/x transform (good for error minimization)."""
        return 1.0 / (x + epsilon)
    
    @staticmethod
    def sigmoid(x: float, steepness: float = 1.0) -> float:
        """Sigmoid normalization to [0,1] range."""
        return 1.0 / (1.0 + jnp.exp(-steepness * x))
    
    @staticmethod  
    def linear_scale(min_val: float, max_val: float) -> Callable[[float], float]:
        """Create linear scaling transform to [0,1] range."""
        def transform(x: float) -> float:
            return (x - min_val) / (max_val - min_val)
        return transform
    
    @staticmethod
    def clamp(min_val: float, max_val: float) -> Callable[[float], float]:
        """Clamp fitness values to specified range."""
        def transform(x: float) -> float:
            return float(jnp.clip(x, min_val, max_val))
        return transform
    
    @staticmethod
    def power_transform(exponent: float) -> Callable[[float], float]:
        """Apply power transformation (x^exponent)."""
        def transform(x: float) -> float:
            return float(jnp.power(jnp.abs(x), exponent)) * jnp.sign(x)
        return transform


class AbstractSolution(JAXTensorizable):
    """
    Abstract base class for solutions in evolutionary algorithms.
    
    A solution wraps a genome with fitness evaluation, providing lazy fitness
    computation, transformation pipelines, and efficient comparison operations.
    Solutions are compared by fitness but identified by genome semantic keys.
    """
    
    def __init__(self, 
                 genome: Optional[AbstractGenome] = None,
                 genome_cls: Optional[type] = None,
                 fitness_transform: Optional[Callable[[float], float]] = None,
                 random_init: bool = False,
                 random_key: Optional[Union[int, Array]] = None,
                 genome_init_params: Optional[Dict[str, Any]] = None,
                 **kwargs: Any):
        """
        Initialize solution with genome and fitness components.
        
        Args:
            genome: Existing genome instance (preferred)
            genome_cls: Genome class for creating new genome
            fitness_transform: Function to transform raw fitness (identity by default)
            random_init: Whether to randomly initialize genome if needed
            random_key: Random seed for genome initialization
            genome_init_params: Parameters for genome initialization
            **kwargs: Additional solution metadata
        """
        self.genome_init_params = genome_init_params or {}

        # Handle genome creation
        if genome is not None:
            self.genome = genome
            self.genome_cls = type(genome)
            
        elif genome_cls is not None:
            self.genome_cls = genome_cls
            self.genome = genome_cls(
                random_init=random_init,
                random_key=random_key,
                **self.genome_init_params
            )
        else:
            raise ValueError("Must provide either 'genome' or 'genome_cls'")
        
        # Fitness components
        self._raw_fitness: Optional[float] = None
        self.fitness_transform = fitness_transform or FitnessTransforms.identity
        self._transformed_fitness: Optional[float] = None
        
        # Metadata and state
        self._metadata: Dict[str, Any] = kwargs
        self.random_key = self._ensure_jax_key(random_key)
    
    def _ensure_jax_key(self, key: Optional[Union[int, Array]]) -> Array:
        """Ensure we have a valid JAX PRNGKey."""
        if key is None:
            return jar.PRNGKey(0)
        elif isinstance(key, int):
            return jar.PRNGKey(key)
        else:
            return key
    
    # === Fitness Properties ===
    
    @property
    def raw_fitness(self) -> Optional[float]:
        """Get the raw (untransformed) fitness value."""
        return self._raw_fitness    

    @raw_fitness.setter
    def raw_fitness(self, value: Optional[float]) -> None:
        """Set the raw fitness value and invalidate transformed cache."""
        self._raw_fitness = value
        self._transformed_fitness = None
    
    @property
    def fitness(self) -> float:
        """
        Get the transformed fitness value.
        
        Returns:
            Transformed fitness value (raises error if no fitness set)
        """
        if self._raw_fitness is None:
            raise ValueError("Raw fitness not set - cannot compute transformed fitness")
        
        if self._transformed_fitness is None:
            self._transformed_fitness = self.fitness_transform(self._raw_fitness)
        return self._transformed_fitness
    
    @property
    def has_fitness(self) -> bool:
        """Check if fitness has been computed."""
        return self._raw_fitness is not None
    
    def invalidate_fitness(self) -> None:
        """Invalidate cached fitness values."""
        self._raw_fitness = None
        self._transformed_fitness = None
    
    # === Genome Properties ===
    
    @property
    def compatibility(self) -> Compatibility:
        """Get compatibility from genome."""
        return self.genome.compatibility
    
    @property
    def is_valid(self) -> bool:
        """Check if genome is valid."""
        return self.genome.is_valid
    
    @property
    def shape(self) -> tuple:
        """Get genome shape."""
        return self.genome.shape
    
    @property
    def size(self) -> int:
        """Get genome size."""
        return self.genome.size
    
    # === Core Methods ===
    
    def is_compatible(self, other: 'AbstractSolution') -> bool:
        """Check if this solution is compatible with another."""
        if not isinstance(other, AbstractSolution):
            return False
        return self.genome.compatibility.is_compatible(other.genome.compatibility)
    
    def distance(self, other: 'AbstractSolution') -> float:
        """Calculate distance between solutions (delegates to genome)."""
        if not self.is_compatible(other):
            raise ValueError("Solutions are not compatible for distance calculation")
        return self.genome.distance(other.genome)
    
    def semantic_key(self) -> str:
        """Return semantic key (delegates to genome)."""
        return self.genome.semantic_key()
    
    # === Metadata Methods ===
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value by key."""
        return self._metadata.get(key, default)
    
    def set_metadata(self, key: str, value: Any) -> None:
        """Set metadata value by key."""
        self._metadata[key] = value
    
    # === JAX Tensor Interface ===
    
    def to_tensor(self) -> Array:
        """Convert solution to tensor (delegates to genome)."""
        return self.genome.to_tensor()
    
    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   context: Optional[SerializationContext] = None,
                   **kwargs: Any) -> 'AbstractSolution':
        """Create solution from tensor representation."""
        if context is None:
            raise ValueError("SerializationContext required for solution reconstruction")
        
        # Extract solution parameters from context
        solution_params = context.for_solution()
        genome_cls = solution_params.get('genome_class')
        
        if genome_cls is None:
            raise ValueError("genome_class not found in serialization context")
        
        # Create genome from tensor
        genome_context = SerializationContext(
            genome_class=genome_cls,
            genome_init_params=context.genome_init_params,
            compatibility=context.compatibility,
            **context.metadata
        )
        
        genome = genome_cls.from_tensor(tensor, genome_init_params=context.genome_init_params)
        
        # Create solution with genome
        return cls(
            genome=genome,
            fitness_transform=solution_params.get('fitness_transform'),
            **kwargs
        )
    
    def get_serialization_context(self) -> SerializationContext:
        """Get context needed to reconstruct this solution."""
        return SerializationContext(
            genome_class=self.genome_cls,
            genome_init_params=getattr(self.genome, '_init_params', {}),
            solution_class=type(self),
            fitness_transform=self.fitness_transform,
            compatibility=self.genome.compatibility,
            **self._metadata
        )
    
    # === Factory Methods ===
    
    def from_genome(self, 
                   genome: AbstractGenome, 
                   random_key: Optional[Array] = None,
                   **solution_kwargs: Any) -> 'AbstractSolution':
        """Create new solution from existing genome."""
        if random_key is None:
            random_key = jar.PRNGKey(hash(str(genome)) % 2**32)
        
        # Create new solution copying relevant attributes
        new_solution = type(self)(
            genome=genome,
            fitness_transform=self.fitness_transform,
            random_key=random_key,
            **solution_kwargs
        )
        
        return new_solution
    
    def clone(self, deep: bool = True, exclude_attrs: Optional[set] = None) -> 'AbstractSolution':
        """
        Clone the solution with generic support for any genome class.
        
        This method works by storing genome_init_params during initialization
        and using them to recreate the solution during cloning.
        
        Args:
            deep: Whether to perform deep copy of mutable attributes
            exclude_attrs: Set of attribute names to exclude from cloning
            
        Returns:
            Cloned solution instance of the same type
        """
        if exclude_attrs is None:
            exclude_attrs = set()
        
        # Clone the genome
        genome_copy = self.genome.clone(deep=deep)
        
        # Create new solution using the same class and stored parameters
        cloned_solution = self.__class__(
            genome_init_params=self.genome_init_params,
            random_init=False, # We already have a genome
            random_key=self.random_key,
            fitness_transform=self.fitness_transform
        )
        
        # Replace the genome with our cloned version
        cloned_solution.genome = genome_copy
        
        # Copy fitness if it exists
        if self.has_fitness:
            cloned_solution._raw_fitness = self._raw_fitness
            if not deep:
                cloned_solution._transformed_fitness = self._transformed_fitness
        
        # Copy other metadata attributes (excluding genome_init_params since we already used it)
        for attr_name, attr_value in self._metadata.items():
            if attr_name not in exclude_attrs and attr_name != 'genome_init_params':
                if deep:
                    setattr(cloned_solution, attr_name, copy.deepcopy(attr_value))
                else:
                    setattr(cloned_solution, attr_name, attr_value)
        
        return cloned_solution
    
    # === Comparison Methods ===
    
    # Comparison based on FITNESS (transformed)
    def __lt__(self, other: 'AbstractSolution') -> bool:
        """Less than comparison based on transformed fitness."""
        if not self.has_fitness or not other.has_fitness:
            raise ValueError("Cannot compare solutions without fitness values")
        return self.fitness < other.fitness
    
    def __le__(self, other: 'AbstractSolution') -> bool:
        """Less than or equal comparison based on transformed fitness."""
        if not self.has_fitness or not other.has_fitness:
            raise ValueError("Cannot compare solutions without fitness values")
        return self.fitness <= other.fitness
    
    def __gt__(self, other: 'AbstractSolution') -> bool:
        """Greater than comparison based on transformed fitness."""
        if not self.has_fitness or not other.has_fitness:
            raise ValueError("Cannot compare solutions without fitness values")
        return self.fitness > other.fitness
    
    def __ge__(self, other: 'AbstractSolution') -> bool:
        """Greater than or equal comparison based on transformed fitness."""
        if not self.has_fitness or not other.has_fitness:
            raise ValueError("Cannot compare solutions without fitness values")
        return self.fitness >= other.fitness
    
    # Equality and hashing based on GENOME semantic key
    def __eq__(self, other: object) -> bool:
        """Equality comparison based on genome semantic key."""
        if not isinstance(other, AbstractSolution):
            return False
        if not self.is_compatible(other):
            return False
        return self.genome == other.genome
    
    def __ne__(self, other: object) -> bool:
        """Inequality comparison based on genome semantic key."""
        return not self.__eq__(other)
    
    def __hash__(self) -> int:
        """Hash based on genome semantic key."""
        return hash(self.genome)
    
    # === String Representations ===
    
    def __str__(self) -> str:
        fitness_str = f"{self.fitness:.4f}" if self.has_fitness else "Not computed"
        return f"Solution(fitness={fitness_str}, genome={self.genome})"
    
    def __repr__(self) -> str:
        return (f"Solution(genome={repr(self.genome)}, "
                f"raw_fitness={self._raw_fitness}, "
                f"transformed_fitness={self._transformed_fitness})")