"""
Core module for MalthusJAX.

This module contains the foundational Level 1 components:
- Base abstractions and compatibility system
- Genome representations  
- Fitness evaluation functions
- Solution wrappers with lazy evaluation
"""

from . import base
from . import genome  
from . import fitness

__all__ = ["base", "genome", "fitness", "population"] 