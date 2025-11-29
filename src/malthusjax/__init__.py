"""
MalthusJAX: A JAX-based framework for evolutionary computation.

A comprehensive 6-level genetic programming framework built on JAX, designed for 
tensorized parallelization, research-grade experimentation, and open-source extensibility.
"""

from . import core
from . import operators
from . import compat

# Expose key new components at top level
from .core.base import BaseGenome, BasePopulation, DistanceMetric
from .core.genome.linear import LinearGenome, LinearPopulation, LinearGenomeConfig
from .core.genome.binary_genome import BinaryGenome, BinaryPopulation, BinaryGenomeConfig
from .core.genome.real_genome import RealGenome, RealPopulation, RealGenomeConfig
from .core.genome.categorical_genome import CategoricalGenome, CategoricalPopulation, CategoricalGenomeConfig
from .core.fitness.evaluators import BaseEvaluator, RegressionData
from .core.fitness.linear_gp_evaluator import LinearGPEvaluator, OP_FUNCTIONS, OP_NAMES
from .core.fitness.binary_evaluators import (
    BinarySumEvaluator, BinarySumConfig, KnapsackEvaluator, KnapsackConfig
)
from .core.fitness.real_evaluators import (
    SphereEvaluator, SphereConfig, GriewankEvaluator, GriewankConfig,
    BoxEvaluator, BoxConfig
)
from .operators.base import BaseMutation, BaseCrossover, BaseSelection
from .operators.linear_operators import LinearMutation, LinearCrossover, LinearPointMutation

__version__ = "0.2.0"  # Bumped for new architecture
__all__ = [
    "core", "operators", "compat",
    # Core abstractions
    "BaseGenome", "BasePopulation", "DistanceMetric",
    # Genome types
    "LinearGenome", "LinearPopulation", "LinearGenomeConfig",
    "BinaryGenome", "BinaryPopulation", "BinaryGenomeConfig",
    "RealGenome", "RealPopulation", "RealGenomeConfig", 
    "CategoricalGenome", "CategoricalPopulation", "CategoricalGenomeConfig",
    # Evaluators & GP
    "LinearGPEvaluator", "OP_FUNCTIONS", "OP_NAMES",
    "BaseEvaluator", "RegressionData",
    # Binary fitness functions
    "BinarySumEvaluator", "BinarySumConfig",
    "KnapsackEvaluator", "KnapsackConfig",
    # Real fitness functions  
    "SphereEvaluator", "SphereConfig",
    "GriewankEvaluator", "GriewankConfig",
    "BoxEvaluator", "BoxConfig",
    # NEW Genetic operators
    "BaseMutation", "BaseCrossover", "BaseSelection",
    "LinearMutation", "LinearCrossover", "LinearPointMutation",
]