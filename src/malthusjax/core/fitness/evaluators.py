"""
Modern fitness evaluator abstractions with JAX-native design.

Provides BaseEvaluator for generic fitness evaluation and specialized
evaluators for different problem types with automatic vectorization.
"""

from typing import TypeVar, Generic, Tuple
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.core.base import BaseGenome, BasePopulation


# Type variables for generic evaluator components
G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C")  # Config type
D = TypeVar("D")  # Data type


@struct.dataclass
class BaseEvaluator(Generic[G, C, D]):
    """
    Abstract base class for fitness evaluators.
    
    Provides automatic vectorization over populations and clean
    separation between single-genome evaluation and batch operations.
    """
    config: C

    def evaluate(self, genome: G, data: D) -> chex.Array:
        """
        Evaluate a single genome on given data.
        
        Args:
            genome: Single genome to evaluate
            data: Evaluation data (e.g., training set)
            
        Returns:
            Fitness score(s) - can be scalar or array for multi-objective
        """
        raise NotImplementedError

    def evaluate_population(self, population: BasePopulation[G], data: D) -> BasePopulation[G]:
        """
        Evaluate entire population with automatic vectorization.
        
        Args:
            population: Population to evaluate
            data: Evaluation data
            
        Returns:
            Population with updated fitness values
        """
        # Vectorize over genes (axis 0), keep self/data constant
        fitness_scores = jax.vmap(self.evaluate, in_axes=(0, None))(population.genes, data)
        return population.replace(fitness=fitness_scores)


# Type alias for regression data
RegressionData = Tuple[chex.Array, chex.Array]  # (X, y)