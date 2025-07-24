"""
Fitness module for MalthusJAX.

This module provides abstract base classes and concrete implementations for
fitness functions in evolutionary algorithms, with emphasis on efficient
batch evaluation using JAX.
"""

from .base import AbstractFitnessEvaluator
from .binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator

__all__ = [
    "AbstractFitnessEvaluator",
    "BinarySumFitnessEvaluator",
    "KnapsackFitnessEvaluator",
]
