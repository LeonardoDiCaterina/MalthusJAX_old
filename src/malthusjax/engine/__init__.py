"""
Level 3: The JIT-compiled Genetic Algorithm Engine.

This package contains the core `MalthusState` Pytree and the 
`MalthusEngine` orchestrator that uses `jax.lax.scan`
to run the evolutionary loop.
"""

from .state import MalthusState

__all__ = ["MalthusState"]