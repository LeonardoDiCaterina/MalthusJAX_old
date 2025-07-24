"""
Genome module for MalthusJAX.

This module provides abstract base classes and concrete implementations for
genome representations in evolutionary algorithms.
"""

from .base import AbstractGenome
from .binary import BinaryGenome

__all__ = [
    "AbstractGenome",
    "BinaryGenome",    
]