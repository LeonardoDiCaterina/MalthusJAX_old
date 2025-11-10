"""
Base classes for mutation operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, TypeVar, Generic, Callable, Tuple

import jax  # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from functools import partial # type: ignore
import functools

from malthusjax.operators.base import AbstractGeneticOperator

DEFAULT_RANDOM_SEED = 0

class AbstractMutation(AbstractGeneticOperator, ABC):
    """Abstract base class for mutation operators.
    
    Mutation operators return a pure function with the signature:
    (key: PRNGKey, genome: jax.Array) -> mutated_genome: jax.Array
    """

    def __init__(self, mutation_rate: float) -> None:
        """
        Initialize mutation operator.
        
        Args:
            mutation_rate: Probability of mutation (behavior depends on operator).
        """
        super().__init__()
        if not (0.0 <= mutation_rate <= 1.0):
            raise ValueError("Mutation rate must be between 0.0 and 1.0")
        self.mutation_rate = mutation_rate
        
    @abstractmethod
    def get_pure_function(self) -> Callable:
        """
        Returns a pure function implementing the mutation logic.
        
        The function will have the signature:
        (key: jax.Array, genome: jax.Array) -> mutated_genome: jax.Array
        """
        pass
    
    def verify_signature(self, test_genome_shape: Tuple[int, ...] = (5,)) -> bool:
        """
        Verify that the compiled function follows the correct signature.
        
        Args:
            test_genome_shape: Shape of test genome for validation.
            
        Returns:
            True if signature is correct, False otherwise.
            
        Raises:
            Exception with details if the signature test fails.
        """
        try:
            # Create test data
            test_key = jar.PRNGKey(DEFAULT_RANDOM_SEED)
            test_genome = jnp.ones(test_genome_shape, dtype=jnp.float32)
            
            # Get the compiled function
            mutation_fn = self.get_pure_function()
            
            # Test correct signature: (key, genome)
            try:
                result = mutation_fn(test_key, test_genome)
                
                # Verify result has expected properties
                if not isinstance(result, jax.Array):
                    raise ValueError(f"Expected jax.Array output, got {type(result)}")
                
                # For most mutations, output should have same shape as input
                # (some operators might change this, but this is the common case)
                expected_shape = test_genome.shape
                if result.shape != expected_shape:
                    print(f"Warning: Output shape {result.shape} != input shape {expected_shape}")
                
                print(f"✅ {self.__class__.__name__} signature verified: (key, genome) -> result")
                return True
                
            except Exception as e:
                # Try wrong signature: (genome, key) to give helpful error
                try:
                    wrong_result = mutation_fn(test_genome, test_key)
                    raise ValueError(
                        f"❌ {self.__class__.__name__} uses WRONG signature (genome, key). "
                        f"Should be (key, genome). Error with correct signature: {e}"
                    )
                except:
                    # Both failed, show the original error
                    raise ValueError(
                        f"❌ {self.__class__.__name__} signature test failed. "
                        f"Expected (key, genome) -> result. Error: {e}"
                    )
                    
        except Exception as e:
            print(f"❌ Signature verification failed for {self.__class__.__name__}: {e}")
            return False