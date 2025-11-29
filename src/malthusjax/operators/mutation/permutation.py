"""
Permutation mutation operators using the new paradigm.

This module provides mutation operators for permutation/array-based genomes using 
the new @struct.dataclass factory pattern for JIT compilation and vectorization.
These operators work on any array-like genome (Binary, Real, Categorical).
"""

from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation
from typing import TypeVar, Generic

# Generic genome type for permutation operations
G = TypeVar("G")
C = TypeVar("C")


@struct.dataclass
class ScrambleMutation(BaseMutation[G, C]):
    """
    Scramble Mutation using the new paradigm.
    
    The `mutation_rate` is the probability that the genome is scrambled *at all*.
    Works on any array-like genome structure.
    """
    # --- DYNAMIC PARAMS (Runtime configurable) ---
    mutation_rate: float = 0.1

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Apply scramble mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Conditionally apply scramble based on mutation rate
        def scramble_genome():
            # Generate random permutation indices
            indices = jar.permutation(k2, jnp.arange(genome.genome.shape[-1]))
            return genome.genome[indices]
        
        def keep_genome():
            return genome.genome
        
        # Decide whether to scramble
        should_mutate = jar.bernoulli(k1, self.mutation_rate)
        mutated_data = jax.lax.cond(should_mutate, scramble_genome, keep_genome)
        
        # Create new genome using clone
        try:
            return genome.replace(genome=mutated_data)
        except AttributeError:
            new_genome = genome.clone()
            new_genome.genome = mutated_data
            return new_genome


@struct.dataclass
class SwapMutation(BaseMutation[G, C]):
    """
    Swap Mutation using the new paradigm.
    
    Swaps two random genes in the genome.
    The `mutation_rate` is the probability that a swap occurs *at all*.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Apply swap mutation to a single genome."""
        k1, k2, k3 = jar.split(key, 3)
        
        # Conditionally apply swap based on mutation rate
        def swap_genes():
            # Pick two random positions
            genome_size = genome.genome.shape[-1]
            pos1 = jar.randint(k2, (), 0, genome_size)
            pos2 = jar.randint(k3, (), 0, genome_size)
            
            # Perform swap
            val1 = genome.genome[pos1]
            val2 = genome.genome[pos2]
            
            return genome.genome.at[pos1].set(val2).at[pos2].set(val1)
        
        def keep_genome():
            return genome.genome
        
        # Decide whether to swap
        should_mutate = jar.bernoulli(k1, self.mutation_rate)
        mutated_data = jax.lax.cond(should_mutate, swap_genes, keep_genome)
        
        # Create new genome using clone
        try:
            return genome.replace(genome=mutated_data)
        except AttributeError:
            new_genome = genome.clone()
            new_genome.genome = mutated_data
            return new_genome


__all__ = [
    "ScrambleMutation",
    "SwapMutation"
]