"""
Selection operators for MalthusJAX.

Selection operators reduce population size by choosing individuals based on
fitness values or other criteria. All selection operators inherit from SelectionLayer.
"""

from .base import AbstractSelectionOperator
from .tournament import TournamentSelection
from .roulette import RouletteSelection

__all__ = [
    "AbstractSelectionOperator ",
    "TournamentSelection",
    "RouletteSelection",
]