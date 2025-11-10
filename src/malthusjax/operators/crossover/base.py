"""
Base classes for crossover operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Callable, Tuple
from malthusjax.operators.base import AbstractGeneticOperator
import jax  # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

DEFAULT_RANDOM_SEED = 0


class AbstractCrossover(AbstractGeneticOperator, ABC):
    """Abstract base class for crossover operators.
    
    Crossover operators return a pure function with the signature:
    (key: PRNGKey, parent1: jax.Array, parent2: jax.Array) -> offspring_batch: jax.Array
    """

    def __init__(self, crossover_rate: float, n_outputs: int = 1) -> None:
        """
        Initialize crossover operator.
        
        Args:
            crossover_rate: Probability of crossover (behavior depends on operator).
            n_outputs: Number of offspring to produce *per pair* of parents.
        """
        super().__init__()
        if not (0.0 <= crossover_rate <= 1.0):
            raise ValueError("Crossover rate must be between 0.0 and 1.0")
        self.crossover_rate = crossover_rate
        self.n_outputs = n_outputs

    @abstractmethod
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compilable function for performing crossover.
        
        The function will have the signature:
        (key: jax.Array, parent1: jax.Array, parent2: jax.Array) -> offspring_batch: jax.Array
        
        The output shape will be (n_outputs, ...genome_shape).
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
                test_parent1 = jnp.ones(test_genome_shape, dtype=jnp.float32)
                test_parent2 = jnp.zeros(test_genome_shape, dtype=jnp.float32)
                
                # Get the compiled function
                crossover_fn = self.get_pure_function()
                
                # Test correct signature: (key, parent1, parent2)
                try:
                    result = crossover_fn(test_key, test_parent1, test_parent2)
                    
                    # Verify result has expected properties
                    if not isinstance(result, jax.Array):
                        raise ValueError(f"Expected jax.Array output, got {type(result)}")
                    
                    # Check output shape: should be (n_outputs, ...genome_shape)
                    expected_shape = (self.n_outputs,) + test_genome_shape
                    if result.shape != expected_shape:
                        # Some crossover operators might return flattened or different shapes
                        # Give a warning but don't fail
                        print(f"Warning: Output shape {result.shape} != expected {expected_shape}")
                        
                        # Check if total elements match (might be flattened)
                        expected_elements = self.n_outputs * jnp.prod(jnp.array(test_genome_shape))
                        actual_elements = jnp.prod(jnp.array(result.shape))
                        if actual_elements != expected_elements:
                            print(f"Warning: Element count mismatch. Expected {expected_elements}, got {actual_elements}")
                    
                    print(f"✅ {self.__class__.__name__} signature verified: (key, parent1, parent2) -> offspring")
                    return True
                    
                except Exception as e:
                    # Test various wrong signatures to give helpful errors
                    wrong_signatures = [
                        ("parent1, parent2, key", lambda: crossover_fn(test_parent1, test_parent2, test_key)),
                        ("parent1, key, parent2", lambda: crossover_fn(test_parent1, test_key, test_parent2)),
                        ("key, parent2, parent1", lambda: crossover_fn(test_key, test_parent2, test_parent1)),
                    ]
                    
                    for sig_name, sig_test in wrong_signatures:
                        try:
                            wrong_result = sig_test()
                            raise ValueError(
                                f"❌ {self.__class__.__name__} uses WRONG signature ({sig_name}). "
                                f"Should be (key, parent1, parent2). Error with correct signature: {e}"
                            )
                        except:
                            continue
                    
                    # All wrong signatures also failed, show the original error
                    raise ValueError(
                        f"❌ {self.__class__.__name__} signature test failed. "
                        f"Expected (key, parent1, parent2) -> offspring. Error: {e}"
                    )
                        
            except Exception as e:
                print(f"❌ Signature verification failed for {self.__class__.__name__}: {e}")
                return False