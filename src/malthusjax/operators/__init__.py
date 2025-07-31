"""
Operators module for MalthusJAX - Level 2 Architecture.

Provides genetic operation layers for Keras-like evolutionary algorithm composition.
Security: Uses explicit imports only, no dynamic code execution.
"""

from .base import AbstractGeneticOperator
from .selection.base import AbstractSelectionOperator
from .crossover.base import AbstractCrossover
from .mutation.base import AbstractMutation

# Import specific implementations
from .selection.roulette import RouletteSelection
from .selection.tournament import TournamentSelection
from .crossover.binary import CycleCrossover, UniformCrossover
from .mutation.binary import BitFlipMutation, ScrambleMutation

__all__ = [
    # Base classes
    "AbstractGeneticOperator",
    "AbstractSelectionOperator",
    "AbstractCrossover",
    "AbstractMutation",
    # Selection operators
    "RouletteSelection",
    "TournamentSelection",
    # Specific implementations
    "UniformCrossover",
    "CycleCrossover",   
    "BitFlipMutation",
    "ScrambleMutation", 
]

__version__ = "0.1.0"
__security_checked__ = True