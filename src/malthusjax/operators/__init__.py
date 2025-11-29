"""
Operators module for MalthusJAX - Level 2 Architecture.

Provides genetic operation layers for evolutionary algorithms with NEW paradigm.
Uses @struct.dataclass operators with JAX JIT compilation support.
"""

# Import NEW paradigm base classes
from .base import BaseMutation, BaseCrossover, BaseSelection

# Import concrete NEW operators
from .selection import TournamentSelection, RouletteWheelSelection
from .crossover import BinaryUniformCrossover, BinarySinglePointCrossover, BlendCrossover, SimulatedBinaryCrossover
from .mutation import BitFlipMutation, CategoricalFlipMutation, BallMutation, SwapMutation, ScrambleMutation

# Additional operators
try:
    from .linear_operators import LinearMutation, LinearCrossover
except ImportError:
    # Linear operators not available
    pass


__all__ = [
    # Base classes
    "BaseMutation",
    "BaseCrossover", 
    "BaseSelection",
    # Selection operators
    "TournamentSelection",
    "RouletteWheelSelection",
    # Crossover operators
    "BinaryUniformCrossover",
    "BinarySinglePointCrossover", 
    "BlendCrossover",
    "SimulatedBinaryCrossover",
    # Mutation operators
    "BitFlipMutation",
    "CategoricalFlipMutation",
    "BallMutation",
    "SwapMutation",
    "ScrambleMutation",
]

__version__ = "0.1.0"
__security_checked__ = True