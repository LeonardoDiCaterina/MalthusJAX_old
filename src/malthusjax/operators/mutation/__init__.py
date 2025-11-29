"""
Mutation operators for MalthusJAX.

Mutation operators introduce variation into the population while maintaining
population size. All mutation operators inherit from BaseMutation.
"""
from .binary import BitFlipMutation, ScrambleMutation, SwapMutation
from .permutation import ScrambleMutation, SwapMutation
from .categorical import CategoricalFlipMutation
from .real import BallMutation

__all__ = [
    "BitFlipMutation",
    "ScrambleMutation",
    "SwapMutation",
    "CategoricalFlipMutation",
    "BallMutation",
]