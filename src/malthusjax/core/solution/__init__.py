"""
Solution module for MalthusJAX.

This module provides the Solution class that wraps genomes with fitness evaluation,
lazy computation, and transformation pipelines for evolutionary algorithms.
"""

from .base import AbstractSolution, FitnessTransforms

__all__ = [
    "AbstractSolution",
    "FitnessTransforms",
    "BinarySolution",
]