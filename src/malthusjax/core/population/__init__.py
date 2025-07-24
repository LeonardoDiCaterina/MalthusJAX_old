"""
Population module for MalthusJAX.

This module provides the Population class that manages a collection of genomes,
allowing for operations like adding solutions, updating fitness, and retrieving
statistics. It serves as a concrete implementation of the AbstractPopulation interface.
"""

from .base import AbstractPopulation
from .population import Population

__all__ = [
    "AbstractPopulation",
    "Population",
]