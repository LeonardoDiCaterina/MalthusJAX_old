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
    def get_compiled_function(self) -> Callable:
        """
        Get the pure, JIT-compilable function for this operator.
        
        This function will have static arguments (like mutation_rate)
        partially applied and is ready for `jax.jit`.
        
        Returns:
            A callable JAX function.
        """
        pass
    
    def from_tensor():
        """Genetic operators do not require tensor deserialization."""
        raise NotImplementedError("Genetic operators do not support from_tensor().")
    
    def to_tensor(self) -> Dict[str, Any]:
        """Serialize the genetic operator to a tensor dictionary."""
        raise NotImplementedError("Genetic operators do not support to_tensor().")