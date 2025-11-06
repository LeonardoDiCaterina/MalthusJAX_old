"""
Base classes for crossover operators with optimized JAX JIT support.
"""
from abc import ABC, abstractmethod
from typing import Callable
from malthusjax.operators.base import AbstractGeneticOperator

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
    def get_compiled_function(self) -> Callable:
        """
        Returns a JIT-compiled function for performing crossover.
        
        The function will have the signature:
        (key: jax.Array, parent1: jax.Array, parent2: jax.Array) -> offspring_batch: jax.Array
        
        The output shape will be (n_outputs, ...genome_shape).
        """
        pass