"""
Binary solution implementation that simplifies usage of AbstractSolution for binary problems.
"""

from typing import Any, Optional, Dict, Tuple, List, Callable
import jax  # type: ignore
import jax.random as jar  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore

from ..base import Compatibility, ProblemTypes, SerializationContext
from ..genome.binary import BinaryGenome
from .base import AbstractSolution


class BinarySolution(AbstractSolution):
    """
    Convenience wrapper for binary optimization problems.
    
    Simplifies usage by handling context automatically and providing
    streamlined tensor operations without exposing serialization complexity.
    """
    def __init__(self, 
                genome_init_params: Optional[Dict] = None,
                random_init: bool = False,
                random_key: Optional[jar.PRNGKey] = None,
                fitness_transform: Optional[Callable[[float], float]] = None):
        """
        Initialize BinarySolution.
        
        Args:
            genome_init_params: Parameters for genome initialization (must include 'array_size')
            random_init: Whether to randomly initialize the genome
            random_key: JAX random key for initialization
            fitness_transform: Optional fitness transformation function
        """
        if genome_init_params is None:
            raise ValueError("genome_init_params must be provided")
        
        # Call parent constructor
        super().__init__(
            genome=BinaryGenome(**genome_init_params,random_init=random_init,random_key=random_key),
            random_key=random_key,
            fitness_transform=fitness_transform,
            random_init=random_init,
            genome_init_params=genome_init_params
        )
    
    # === Convenience Properties ===
    
    @property
    def array_size(self) -> int:
        """Get the binary array size."""
        return self.genome_init_params.get('array_size')

    @array_size.setter
    def array_size(self, value: int):
        """Set the binary array size."""
        self.genome_init_params['array_size'] = value
    
    @property
    def p(self) -> float:
        """Get the probability parameter."""
        return self.genome_init_params.get('p', self._p)
    @p.setter
    def p(self, value: float):
        """Set the probability parameter."""
        self.genome_init_params['p'] = value
        
    @property
    def binary_array(self) -> Array:
        """Get the binary array directly."""
        return self.genome.genome
    
    # === Simplified Factory Methods ===
    
    def from_genome(self, genome: BinaryGenome, random_key: Optional[int] = None) -> 'BinarySolution':
        """
        Create a new solution from an existing binary genome.
        
        Args:
            genome: Binary genome instance
            random_key: Optional random key for the new solution
            
        Returns:
            New BinarySolution with the provided genome
        """
        if not isinstance(genome, BinaryGenome):
            raise TypeError("genome must be a BinaryGenome instance")
        
        # Create new solution with same parameters
        new_solution = BinarySolution(
            random_init=False,
            random_key=random_key,
            fitness_transform=self.fitness_transform,
            genome_init_params=self.genome_init_params
        )
        
        # Set the genome directly
        new_solution.genome = genome
        
        return new_solution
    
    @classmethod
    def from_binary_array(cls, 
                         binary_array: Array,
                         genome_init_params: Optional[Dict[str, Any]],
                         fitness_transform = None,
                         **kwargs: Any) -> 'BinarySolution':
        """
        Create solution directly from a binary array.
        
        Args:
            binary_array: JAX array of 0s and 1s
            p: Probability parameter for the solution
            fitness_transform: Optional fitness transformation
            **kwargs: Additional metadata
            
        Returns:
            New BinarySolution with the binary array
        """
        array_size = binary_array.shape[0]
        if genome_init_params.get('array_size') != array_size:
            raise ValueError(f"Binary array size {array_size} does not match genome_init_params['array_size'] {genome_init_params.get('array_size')}")
        
        # Create solution without random initialization
        solution = cls(
            genome_init_params = genome_init_params,
            random_init=False,
            fitness_transform=fitness_transform,
            **kwargs
        )
        
        # Set the genome array directly
        solution.genome.genome = binary_array.astype(jnp.bool_)
        
        return solution
    
    # === Simplified Tensor Operations (Context is Implicit) ===
    
    @classmethod
    def from_tensor(cls, 
                   tensor: Array,
                   genome_init_params: [Dict[str, Any]],
                   context: Optional[SerializationContext] = None,
                   fitness_transform = None,
                   **kwargs: Any) -> 'BinarySolution':
        """
        Create BinarySolution from tensor with automatic context handling.
        
        Context is built automatically - users don't need to manage it.
        
        Args:
            tensor: JAX tensor representing the binary genome
            genome_init_params: Parameters for initializing the genome
            context: Optional context (usually None for convenience)
            fitness_transform: Optional fitness transformation
            **kwargs: Additional metadata
            
        Returns:
            New BinarySolution reconstructed from tensor
        """

        
        # Infer array_size from tensor if not provided
      
        
        if tensor.shape[0] != genome_init_params.get('array_size'):
            raise ValueError(f"Tensor shape {tensor.shape} does not match expected array size {genome_init_params.get('array_size')}")
        
        # Create the solution
        solution = cls(
            genome_init_params=genome_init_params,
            random_init=False,
            fitness_transform=fitness_transform,
            **kwargs
        )
        
        # Use the parent's from_tensor with auto-generated context
        if context is None:
            context = solution.get_serialization_context(genome_init_params =genome_init_params)
        
        # Create genome from tensor
        solution.genome = BinaryGenome.from_tensor(tensor, genome_init_params = genome_init_params)
        
        return solution
    
    def get_serialization_context(self,genome_init_params) -> SerializationContext:
        """Get context with binary-specific parameters automatically included."""
        return SerializationContext(
            genome_class=BinaryGenome,
            genome_init_params=genome_init_params,
            solution_class=BinarySolution,
            fitness_transform=self.fitness_transform,
            **self._metadata
        )
    
    # === JAX PyTree Support (Simplified) ===
    
    def tree_flatten(self) -> Tuple[List[Array], Dict[str, Any]]:
        """
        Flatten for JAX operations with simplified structure.
        """
        # Children: genome tensor and fitness
        children = [
            self.genome.to_tensor(),
            jnp.array([0.0 if not self.has_fitness else self.raw_fitness])
        ]
        
        # Aux data: everything needed to reconstruct
        aux_data = {
            'array_size': self.genome_init_params['array_size'],
            'p': self.genome_init_params['p'],
            'fitness_transform': self.fitness_transform,
            'metadata': self._metadata
        }
        
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data: Dict[str, Any], children: List[Array]) -> 'BinarySolution':
        """
        Reconstruct from flattened representation.
        """
        genome_tensor, fitness_array = children
        
        # Create solution
        solution = cls(
            array_size=aux_data['array_size'],
            p=aux_data['p'],
            random_init=False,
            fitness_transform=aux_data.get('fitness_transform')
        )
        
        # Set genome from tensor (context handled automatically)
        solution.genome = BinaryGenome.from_tensor(
            genome_tensor, 
            context=solution.get_serialization_context()
        )
        
        # Set fitness if it was saved
        fitness_val = float(fitness_array[0])
        if fitness_val != 0.0:  # Assuming 0.0 means no fitness
            solution._raw_fitness = fitness_val
        
        # Set metadata
        solution._metadata.update(aux_data.get('metadata', {}))
        
        return solution
    
    # === Convenience Methods ===
    
    def flip_bit(self, index: int) -> 'BinarySolution':
        """
        Create new solution with a single bit flipped.
        
        Args:
            index: Index of bit to flip
            
        Returns:
            New BinarySolution with flipped bit
        """
        if not 0 <= index < self.genome_init_params['array_size']:
            raise ValueError(f"Index {index} out of bounds for array size {self.genome_init_params['array_size']}")
        
        # Create new array with flipped bit
        new_array = self.binary_array.at[index].set(1 - self.binary_array[index])
        
        return self.from_binary_array(
            new_array,
            genome_init_params=self.genome_init_params,
            fitness_transform=self.fitness_transform
        )
    
    def hamming_distance(self, other: 'BinarySolution') -> int:
        """
        Calculate Hamming distance to another binary solution.
        
        Args:
            other: Another BinarySolution
            
        Returns:
            Hamming distance as integer
        """
        if not isinstance(other, BinarySolution):
            raise TypeError("other must be a BinarySolution")
        if other.array_size != self.array_size:
            raise ValueError("Solutions must have same array size")
        
        return int(jnp.sum(jnp.abs(self.binary_array - other.binary_array)))
    
    def to_binary_string(self) -> str:
        """Convert binary array to string representation."""
        return ''.join(str(int(x)) for x in self.binary_array.tolist())
    
    def __str__(self) -> str:
        """Simplified string representation."""
        fitness_str = f"{self.fitness:.4f}" if self.has_fitness else "Not computed"
        binary_str = self.to_binary_string()
        if len(binary_str) > 20:
            binary_str = binary_str[:17] + "..."
        return f"BinarySolution(fitness={fitness_str}, binary={binary_str})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return (f"BinarySolution(array_size={self.genome_init_params['array_size']}, "
                f"p={self._p}, "
                f"fitness={self.raw_fitness if self.has_fitness else None}, "
                f"binary={self.to_binary_string()})")


# Register with JAX for tree operations
jax.tree_util.register_pytree_node(
    BinarySolution,
    BinarySolution.tree_flatten,
    BinarySolution.tree_unflatten
)