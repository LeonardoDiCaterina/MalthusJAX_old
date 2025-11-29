"""
Selection operators for MalthusJAX.

Selection operators reduce population size by choosing individuals based on
fitness values or other criteria. All selection operators inherit from BaseSelection.
"""

from .tournament import TournamentSelection
from .roulette import RouletteWheelSelection

__all__ = [
    "TournamentSelection",
    "RouletteWheelSelection",
]