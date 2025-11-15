"""
State definitions for MalthusJAX engines.

This module provides backward compatibility and centralizes state definitions.
For new code, prefer importing states directly from their respective engines:
- ProductionState from malthusjax.engine.ProductionEngine
- ResearchState from malthusjax.engine.ResearchEngine
"""

import flax.struct # type: ignore
from jax import Array # type: ignore
from jax.random import PRNGKey # type: ignore

from malthusjax.engine.base import AbstractState


@flax.struct.dataclass
class MalthusState(AbstractState):
    """
    Legacy state class for backward compatibility.
    
    This is now just an alias to AbstractState for existing code.
    New engines should define their own state classes that inherit from AbstractState.
    """
    pass