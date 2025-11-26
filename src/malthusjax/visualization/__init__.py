"""
MalthusJAX Level 4: Visualization Module

This module provides stateful visualization classes for analyzing evolution results.
All classes follow MalthusJAX design patterns with proper initialization and caching.
"""

from .base import AbstractVisualizer, VisualizationConfig
from .single_run import EvolutionVisualizer, GeneticAlgorithmVisualizer
from .multi_run import EngineComparator, FunctionalDataAnalyzer

__all__ = [
    'AbstractVisualizer',
    'EvolutionVisualizer', 
    'GeneticAlgorithmVisualizer',
    'EngineComparator',
    'FunctionalDataAnalyzer',
    'VisualizationConfig'
]