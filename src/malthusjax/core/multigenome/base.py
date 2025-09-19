"""
Abstract base classes for multiple genome representations in MalthusJAX.

This module defines the fundamental genome abstractions that encode candidate solutions
for evolutionary algorithms with multiple genome types. All genome types must implement
JAX tensor interfaces for efficient batch operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Dict, Tuple, Type, TypeVar
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore
import jax  # type: ignore
from ..base import JAXTensorizable, Compatibility, ProblemTypes, SerializationContext
from ..genome.base import AbstractGenome

T = TypeVar('T', bound=AbstractGenome)

class AbstractMultiGenome(JAXTensorizable, ABC):
    """
    Abstract base class for all multi-genome representations.

    A multi-genome encodes a candidate solution to the optimization problem.
    It serves as a passive data container with validation, distance calculation,
    and semantic key functionality for efficient set operations.

    Genetic operations (mutation, crossover) are handled by Level 2 operators
    that work on batches of multi-genomes for efficiency.
    """

    def __init__(self, 
                 genome_init_params: Optional[Dict[str, Dict[str, Any]]] = None,
                 genome_types_dict: Optional[Dict[str, Type]] = None,
                 random_init: bool = False,
                 random_key: Optional[int] = None, 
                 compatibility: Optional[Compatibility] = None,
                 **kwargs: Any):
        """
        Initialize multi-genome with compatibility information.

        Args:
            random_init: Whether to randomly initialize the multi-genome
            genome_init_params: Parameters for random initialization,
            genome_types_list: List of genome types contained in this multi-genome
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

        self.genome_init_params_dict = genome_init_params if genome_init_params is not None else {}
        self.genome_types_dict = genome_types_dict if genome_types_dict is not None else {}
        

        self._genome_list: List[T] = []
        if random_init:
            self._random_init()
    @property
    def size(self) -> tuple:
        """Get tuple of sizes of the genome tensors."""
        return tuple(genome.size for genome in self._genome_list)
 
    @property
    def shape(self) -> tuple:
        """Get tuple of shapes of the genome tensors."""
        return tuple(genome.shape for genome in self._genome_list)

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
    def fitness(self) -> tuple:
        """Get the fitness value of the genome."""
        return tuple(genome.fitness for genome in self._genome_list)
    
    @fitness.setter
    def fitness(self, value: float, index: int = None) -> None:
        """Set the fitness value of the genome."""
        if index is not None:
            self._genome_list[index].fitness = value
        else:
            for genome in self._genome_list:
                genome.fitness = value

    def invalidate(self, index: int = None) -> None:
        """Invalidate cached validation result."""
        if index is not None:
            self._genome_list[index]._is_valid = None
        else:
            self._is_valid = None
            for genome in self._genome_list:
                genome._is_valid = None

    # === Abstract methods that subclasses must implement ===
    
    @classmethod
    def get_random_initialization_jit(cls, genome_init_params: Dict[str, Any], ) -> Callable[[Optional[int]], jnp.ndarray]:
        """Get JIT-compiled function for random genome initialization that will receive a random key and return a tensor."""
        init_fn_list = [
            genome_cls.get_random_initialization_jit(genome_init_params.get(type, {})) for type, genome_cls in cls.genome_types_dict.items()
            ]
        return cls.create_batch_executor(init_fn_list, as_tuple=True)

    def _random_init(self) -> None:
        """Initialize genome with random values."""
        for type in self.genome_types_dict:
            init_params = self.genome_init_params_dict.get(type, {})
            genome_cls = self.genome_types_dict[type]
            genome = genome_cls(**init_params, random_init=True, random_key=self.random_key)
            self.random_key, _ = jax.random.split(self.random_key)
            self._genome_list.append(genome)

    def _validate(self) -> bool:
        """Validate all the genomes in the multi-genome."""
        return all(genome.is_valid for genome in self._genome_list)

    def to_tensors(self, as_tuple: bool = False) -> tuple:
        """Convert the genome to a JAX tensor."""
        if as_tuple:
            return tuple(genome.to_tensor() for genome in self._genome_list)
        return [genome.to_tensor() for genome in self._genome_list]
    # === JAX JIT Compatibility abstractions ===

    def get_distance_jit(self, as_tuple: bool = False) -> Callable[[jax.Array, jax.Array], float]:
        """
        Get JIT-compiled function to compute distance between two multi-genomes.

        Args:
            as_tuple (bool, optional): Whether to return the result as a tuple. Defaults to False.

        Returns:
            Callable[[jax.Array, jax.Array], float]: A JIT-compiled function to compute distance between two multi-genomes.
        """
        distance_fn_list = [genome.get_distance_jit() for genome in self._genome_list]
        return self.create_batch_executor(distance_fn_list, as_tuple=True)

    def get_autocorrection_jit(self, as_tuple: bool = False) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        autocorrection_fn_list = [
            genome.get_autocorrection_jit() for genome in self._genome_list
            ]
        return self.create_batch_executor(autocorrection_fn_list, as_tuple=as_tuple)
    
    # === JIT methods made easy ===
    
    def auto_correct(self, index: int = None) -> None:
        """Auto-correct invalid genomes in-place."""
        if index is not None:
            if abs(index) >= len(self._genome_list):
                raise IndexError(f"Index {index} out of range for genome list of length {len(self._genome_list)}")
            if type(index) is int:
                genome_init_params = list(self.genome_init_params_dict)[index]
            if type(index) is str:
                genome_init_params = self.genome_init_params_dict.get(index, {})
            autocorrection_fn = self._genome_list[index].get_autocorrection_jit(genome_init_params)
            corrected_tensor = autocorrection_fn(self._genome_list[index].to_tensor())
            self._genome_list[index].update_from_tensor(corrected_tensor, validate=True)
            return 
                
        auto_correction_fn_list = [
            self.genome_types_dict.get(genome_name).get_autocorrection_jit(self.genome_init_params_dict[genome_name])for genome_name, _ in self.genome_init_params_dict.items()
        ]
        ''''''

        batch_executor = self.create_batch_executor(auto_correction_fn_list, as_tuple=True)
        batch_auto_correction_inputs = [ (self.to_tensors(as_tuple=True)[i],) for i in range(len(self._genome_list)) ]
        return batch_executor(batch_auto_correction_inputs)

    def from_tensor(self, tensor: Array, **kwargs: Any) -> 'AbstractGenome':
        pass
    
    
    def get_serialization_context(self) -> SerializationContext:
        """Get serialization context for the genome."""
        return SerializationContext(
            genome_init_params=self.genome_init_params,
            genome_types_list=self.genome_types_list,
            compatibility=self.compatibility,
            metadata=self.metadata
        )
    
    def to_tensor(self) -> Array:
        """Convert the multi-genome to a list of JAX tensors."""
        return [genome.to_tensor() for genome in self._genome_list]
    @classmethod
    def from_tensors(cls, 
                   tensors: List[Array],
                   genome_init_params: Optional[Dict[str, Any]] = None,
                   **kwargs: Any) -> 'AbstractGenome':
        """Create a multi-genome instance from a list of tensors.
        Args:
            tensors: List of genome data as tensors
            genome_init_params: Parameters for genome initialization
            **kwargs: Additional genome-specific metadata
        Returns:
            A new multi-genome instance
        """
        if genome_init_params is None:
            genome_init_params = {}
        if len(tensors) != len(cls.genome_types_list):
            raise ValueError(f"Number of tensors {len(tensors)} does not match number of genome types {len(cls.genome_types_list)}")

        new_genomes_list = [genome_type.from_tensor(tensor, genome_init_params=genome_init_params, **kwargs) for tensor, genome_type in zip(tensors, cls.genome_types_list)]
        new_instance = cls(genome_init_params=genome_init_params, genome_types_list=cls.genome_types_list, **kwargs)
        new_instance._genome_list = new_genomes_list
        return new_instance

    def distance(self, other: 'AbstractGenome') -> float:
        """Returns a vector of distances between corresponding genomes in two multi-genomes.
        Args:
            other: Another multi-genome to compute distance to
        Returns:
            A vector of distances between corresponding genomes in two multi-genomes.
        """
        if not isinstance(other, AbstractMultiGenome):
            raise TypeError(f"Distance can only be computed between two AbstractMultiGenome instances, got {type(other)}")
        if len(self._genome_list) != len(other._genome_list):
            raise ValueError("Cannot compute distance between multi-genomes with different number of genomes")

        distance_fn = self.get_distance_jit(as_tuple=True)
        batch_distance_inputs = [
            (self.to_tensors(as_tuple=True)[i], other.to_tensors(as_tuple=True)[i]) for i in range(len(self._genome_list))
        ]
        batch_exec = distance_fn(batch_distance_inputs)
        return batch_exec
    
    def semantic_key(self) -> tuple:
        """Generate a unique key for the genome."""
        semantic_parts = tuple(genome.semantic_key() for genome in self._genome_list)
        return semantic_parts

    ''' @abstractmethod
        def tree_flatten(self):
            """JAX tree flattening support."""
            pass

        @classmethod
        @abstractmethod
        def tree_unflatten(cls, aux_data, children):
            """JAX tree unflattening support."""
            pass'''

    def clone(self, deep: bool = True) -> 'AbstractMultiGenome':
        """Create a copy of the genome.
        
        Args:
            deep: If True, create a deep copy. If False, create a shallow copy.
            
        Returns:
            A new genome instance that is a copy of this one.
        """
        new_instance = self.__class__(
            genome_init_params=self.genome_init_params_dict,
            genome_types_dict=self.genome_types_dict,
            compatibility=self.compatibility,
            **self.metadata
        )
        if deep:
            new_instance._genome_list = [genome.clone(deep=True) for genome in self._genome_list]
        else:
            new_instance._genome_list = self._genome_list.copy()
        return new_instance

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
        
    def __sub__(self, other: 'AbstractMultiGenome') -> float:
        """Calculate distance between two multi-genomes."""
        if not isinstance(other, AbstractMultiGenome):
            raise TypeError(f"Unsupported operand type(s) for -: '{type(self).__name__}' and '{type(other).__name__}'")
        return self.distance(other)
        
    def __hash__(self) -> int:
        """Hash based on semantic key."""
        return hash(self.semantic_key())
    
    def __getitem__(self, index: int) -> AbstractGenome:
        """Get genome at index."""
        return self._genome_list[index]
    
    def __setitem__(self, index: int, value: AbstractGenome) -> None:
        """Set genome at index."""
        if not isinstance(value, AbstractGenome):
            raise TypeError(f"Value must be an instance of AbstractGenome, got {type(value)}")
        self._genome_list[index] = value
        self.invalidate(index=index)
        
    @staticmethod
    def create_batch_executor(jit_functions: List[Callable], as_tuple: bool = True) -> Callable:
        """
        Creates a function that executes multiple JIT-compiled functions with their inputs.
        
        Args:
            jit_functions: List of JIT-compiled functions
            
        Returns:
            A function that takes a list of input tuples and executes all functions
        """
        if as_tuple:
            @jax.jit
            def batch_execute(inputs_list: List[Tuple]) -> List[Any]:
                """
                Execute all functions with their respective inputs.
                
                Args:
                    inputs_list: List of tuples, where each tuple contains the arguments
                                for the corresponding function in jit_functions
                                
                Returns:
                    List of results from each function execution
                """
                results = []
                
                for func, inputs in zip(jit_functions, inputs_list):
                    if isinstance(inputs, tuple):
                        # Unpack tuple arguments
                        result = func(*inputs)
                    else:
                        # Single argument
                        result = func(inputs)
                    results.append(result)

                return tuple(results)

            return batch_execute


        @jax.jit
        def batch_execute(inputs_list: List[Tuple]) -> List[Any]:
            """
            Execute all functions with their respective inputs.
            
            Args:
                inputs_list: List of tuples, where each tuple contains the arguments
                            for the corresponding function in jit_functions
                            
            Returns:
                List of results from each function execution
            """
            results = []
            
            for func, inputs in zip(jit_functions, inputs_list):
                if isinstance(inputs, tuple):
                    # Unpack tuple arguments
                    result = func(*inputs)
                else:
                    # Single argument
                    result = func(inputs)
                results.append(result)
                
            return results

        return batch_execute
   

        
        
