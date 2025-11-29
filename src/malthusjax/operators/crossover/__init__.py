"""
Crossover operators for MalthusJAX.

Crossover operators generate offspring from parent pairs using the new batch-first paradigm.
All crossover operators inherit from BaseCrossover and return (num_offspring, genome_shape).
"""

from .binary import UniformCrossover as BinaryUniformCrossover, SinglePointCrossover as BinarySinglePointCrossover
from .real import BlendCrossover, SimulatedBinaryCrossover

__all__ = [
    "BinaryUniformCrossover", 
    "BinarySinglePointCrossover",
    "BlendCrossover",
    "SimulatedBinaryCrossover",
]
