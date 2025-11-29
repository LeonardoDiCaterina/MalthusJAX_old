"""
Binary mutation operators using the new paradigm.

This module provides mutation operators for BinaryGenome using the new 
@struct.dataclass factory pattern for JIT compilation and vectorization.
"""

from typing import Callable
import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from malthusjax.operators.base import BaseMutation
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig


@struct.dataclass
class BitFlipMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig]):
    """
    Bit flip mutation for binary genomes using the new paradigm.
    
    Flips each bit with probability mutation_rate.
    Supports automatic vectorization for multiple offspring.
    """
    # --- DYNAMIC PARAMS (Runtime configurable) ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Apply bit flip mutation to a single genome."""
        # Generate mutation mask
        mutation_mask = jar.bernoulli(key, self.mutation_rate, genome.bits.shape)
        
        # Apply XOR to flip bits where mask is True
        mutated_bits = jnp.logical_xor(genome.bits, mutation_mask)
        
        # Create new genome using replace
        return genome.replace(bits=mutated_bits)


@struct.dataclass
class ScrambleMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig]):
    """
    Scramble Mutation for binary genomes using the new paradigm.
    
    The `mutation_rate` is the probability that the genome is scrambled *at all*.
    Supports automatic vectorization for multiple offspring.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Apply scramble mutation to a single genome."""
        k1, k2 = jar.split(key)
        
        # Conditionally apply scramble based on mutation rate
        def scramble_genome():
            # Generate random permutation indices
            indices = jar.permutation(k2, jnp.arange(genome.bits.shape[-1]))
            return genome.bits[indices]
        
        def keep_genome():
            return genome.bits
        
        # Decide whether to scramble
        should_mutate = jar.bernoulli(k1, self.mutation_rate)
        mutated_bits = jax.lax.cond(should_mutate, scramble_genome, keep_genome)
        
        # Create new genome using replace
        return genome.replace(bits=mutated_bits)


@struct.dataclass
class SwapMutation(BaseMutation[BinaryGenome, BinaryGenomeConfig]):
    """
    Swap Mutation for binary genomes using the new paradigm.
    
    Swaps two random genes in the genome.
    The `mutation_rate` is the probability that a swap occurs *at all*.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: BinaryGenome, config: BinaryGenomeConfig) -> BinaryGenome:
        """Apply swap mutation to a single genome."""
        k1, k2, k3 = jar.split(key, 3)
        
        # Conditionally apply swap based on mutation rate
        def swap_genes():
            # Pick two random positions
            genome_size = genome.bits.shape[-1]
            pos1 = jar.randint(k2, (), 0, genome_size)
            pos2 = jar.randint(k3, (), 0, genome_size)
            
            # Perform swap
            val1 = genome.bits[pos1]
            val2 = genome.bits[pos2]
            
            return genome.bits.at[pos1].set(val2).at[pos2].set(val1)
        
        def keep_genome():
            return genome.bits
        
        # Decide whether to swap
        should_mutate = jar.bernoulli(k1, self.mutation_rate)
        mutated_bits = jax.lax.cond(should_mutate, swap_genes, keep_genome)
        
        # Create new genome using replace
        return genome.replace(bits=mutated_bits)


__all__ = [
    "BitFlipMutation",
    "ScrambleMutation", 
    "SwapMutation"
]