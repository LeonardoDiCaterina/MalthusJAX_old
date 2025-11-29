"""
Legacy compatibility for transition to NEW paradigm.

This module provides backward compatibility for old fitness evaluator patterns
during the transition to the NEW @struct.dataclass paradigm.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable

class JAXTensorizable:
    """Legacy base class - deprecated in favor of @struct.dataclass."""
    
    def __init__(self):
        """Initialize legacy tensorizable object."""
        pass
    
    @property 
    def random_key(self):
        """Legacy random key property - deprecated."""
        return None

# Re-export for backward compatibility
__all__ = ["JAXTensorizable"]