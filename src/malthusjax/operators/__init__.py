"""
Operators module for MalthusJAX - Level 2 Architecture.

Provides genetic operation layers for Keras-like evolutionary algorithm composition.
Security: Uses explicit imports only, no dynamic code execution.
"""

from .base import AbstractGeneticOperator
from .selection.base import AbstractSelectionOperator
from .crossover.base import AbstractCrossover
from .mutation.base import AbstractMutation

from .selection import TournamentSelection, RouletteSelection
from .crossover import SinglePointCrossover, UniformCrossover, AverageCrossover
from .mutation import BitFlipMutation, CategoricalFlipMutation, BallMutation, SwapMutation, ScrambleMutation


__all__ = [
    "AbstractGeneticOperator",
    "AbstractSelectionOperator",
    "AbstractCrossover",
    "AbstractMutation",
    "TournamentSelection",
    "RouletteSelection",
    "SinglePointCrossover",
    "UniformCrossover",
    "AverageCrossover",
    "BitFlipMutation",
    "CategoricalFlipMutation",
    "BallMutation",
    "SwapMutation",
    "ScrambleMutation",
]

__version__ = "0.1.0"
__security_checked__ = True