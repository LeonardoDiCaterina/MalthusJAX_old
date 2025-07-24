"""
MalthusJAX: A JAX-based framework for evolutionary computation.

A comprehensive 6-level genetic programming framework built on JAX, designed for 
tensorized parallelization, research-grade experimentation, and open-source extensibility.
"""

from . import core
from . import operators

__version__ = "0.1.0"
__all__ = ["core", "operators","models"]