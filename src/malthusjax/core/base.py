"""
Core abstract base classes for MalthusJAX.

This module defines the fundamental abstractions that all components inherit from,
including JAX tensor interfaces, compatibility systems, and core evolutionary concepts.
"""

from abc import ABC, abstractmethod
from typing import Any,TypeVar, Optional
from jax import Array  # type: ignore
import jax.random as jar  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore


# Type variable for generic JAXTensorizable objects to handle random keys properly
T = TypeVar('T', bound='JAXTensorizable')
class JAXTensorizable(ABC):
    """
    Abstract base class for objects that can be converted to/from JAX tensors.

    This enables efficient batch operations and JIT compilation across all
    MalthusJAX components. Inherits from TensorSerializable protocol.
    """
    def __init__(self, random_key: Optional[jnp.ndarray] = None):
        self.jax_random_key: Optional[jnp.ndarray] = random_key
        
    @property
    def random_key(self) -> Optional[jnp.ndarray]:
        """Get the JAX random key after folding it to avoid reuse."""
        if self.jax_random_key is not None:
            self.jax_random_key, subkey = jar.split(self.jax_random_key)
            return subkey
        return self.jax_random_key
    
    #setter for random_key
    @random_key.setter 
    def random_key(self, value: Optional[jnp.ndarray]) -> None:
        """Set the JAX random key."""
        self.jax_random_key = value

    @abstractmethod
    def to_tensor(self) -> Array:
        """Convert this object to a JAX tensor representation."""
        pass

    @classmethod
    @abstractmethod
    def from_tensor(cls, 
                   tensor: Array,
                   **kwargs: Any) -> 'JAXTensorizable':
        """Create an instance from a JAX tensor representation."""
        pass