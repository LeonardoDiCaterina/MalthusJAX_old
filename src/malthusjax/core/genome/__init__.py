"""
Genome module for MalthusJAX.

This module provides NEW paradigm genome implementations using @struct.dataclass
for evolutionary algorithms with JAX JIT compilation support.
"""

# NEW architecture components
from .linear import LinearGenome, LinearGenomeConfig, LinearPopulation
from .binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .real_genome import RealGenome, RealGenomeConfig, RealPopulation
from .categorical_genome import CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation

__all__ = [
    # NEW architecture genomes
    "LinearGenome",
    "LinearGenomeConfig",
    "LinearPopulation",
    "BinaryGenome",
    "BinaryGenomeConfig", 
    "BinaryPopulation",
    "RealGenome",
    "RealGenomeConfig",
    "RealPopulation", 
    "CategoricalGenome",
    "CategoricalGenomeConfig",
    "CategoricalPopulation",
]