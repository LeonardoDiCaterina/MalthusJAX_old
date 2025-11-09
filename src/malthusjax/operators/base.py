"""
Abstract base classes for genetic layers in MalthusJAX.

This module defines the layer abstractions that enable Keras-like composition
of genetic operations for evolutionary algorithms.
"""
# Path: src/malthusjax/operators/base.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Union

from malthusjax.core.base import JAXTensorizable
"""
Abstract base classes for genetic layers in MalthusJAX.

This module defines the layer abstractions that enable Keras-like composition
of genetic operations for evolutionary algorithms.
"""
from abc import ABC, abstractmethod
from typing import Callable
from malthusjax.core.base import JAXTensorizable

class AbstractGeneticOperator(JAXTensorizable, ABC):
    """Abstract base class for genetic operators in MalthusJAX.
    
    All genetic operators follow a "factory" pattern, providing a
    JIT-compilable function that operates on raw JAX arrays.
    """
    
    def __init__(self) -> None:
        """Initialize the genetic operator."""
        super().__init__()

    @abstractmethod    
    def get_pure_function(self) -> Callable:
        """
        Get the pure, a Pure JAX function implementing the genetic operation.
        
        This function will have static arguments (like mutation_rate)
        partially applied and is ready for `jax.jit`.
        
        Returns:
            A callable JAX function.
        """
        pass
    
    @abstractmethod
    def verify_signature(self, *args  ) -> bool:
        """
        Verify that the compiled function follows the correct signature.
        
        Args:
            test_genome_shape: Shape of test genome for validation.
            
        Returns:
            True if signature is correct, False otherwise.
            
        Raises:
            Exception with details if the signature test fails.
        """
        pass
    
    
    def from_tensor():
        """Genetic operators do not require tensor deserialization."""
        raise NotImplementedError("Genetic operators do not support from_tensor().")
    
    def to_tensor(self) -> Dict[str, Any]:
        """Serialize the genetic operator to a tensor dictionary."""
        raise NotImplementedError("Genetic operators do not support to_tensor().")