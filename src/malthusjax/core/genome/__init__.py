"""
Genome module for MalthusJAX.

This module provides abstract base classes and concrete implementations for
genome representations in evolutionary algorithms.
"""

from .base import AbstractGenome
from .binary import BinaryGenome
from .real import RealGenome
from .categorical import CategoricalGenome
#from .permutation import PermutationGenome

__all__ = [
    "AbstractGenome",
    "BinaryGenome",
    "RealGenome",
    "CategoricalGenome",
    #"PermutationGenome"
]