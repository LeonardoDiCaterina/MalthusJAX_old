"""
Crossover operators for MalthusJAX.

Crossover operators generate offspring from parent pairs, typically increasing
population size. All crossover operators inherit from CrossoverLayer.
"""

from .base import AbstractCrossover
from .binary import UniformCrossover, SinglePointCrossover
from .categorical import UniformCrossover, SinglePointCrossover
from .real import UniformCrossover, AverageCrossover

__all__ = [
    "AbstractCrossover",
    "OnePointCrossover",
    "SinglePointCrossover",
    "UniformCrossover",
    "AverageCrossover",
]
