"""
Genetic operators using the new MalthusJAX paradigm.

This module implements mutation and crossover operators following the new design:
- @struct.dataclass for immutable, JIT-compatible operators
- Factory pattern with static/dynamic parameters  
- Pure JAX functions with automatic vectorization
- Generic type support for all genome types
"""

import jax
import jax.numpy as jnp
import jax.random as jar
from flax import struct
import chex
from typing import TypeVar, Generic

from .base import BaseMutation, BaseCrossover

# Import genome types and configs
G = TypeVar("G", bound="BaseGenome")
C = TypeVar("C")

# ==========================================
# BINARY GENOME OPERATORS
# ==========================================

@struct.dataclass
class BitFlipMutation(BaseMutation[G, C]):
    """
    Bit Flip Mutation for binary genomes using new paradigm.
    
    Flips each bit with specified probability.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.01
    
    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Mutate one binary genome."""
        # Generate mutation mask
        mutation_mask = jar.bernoulli(key, p=self.mutation_rate, shape=genome.bits.shape)
        
        # Apply bit flips
        new_bits = jnp.where(mutation_mask, ~genome.bits.astype(bool), genome.bits).astype(genome.bits.dtype)
        
        return genome.replace(bits=new_bits)


@struct.dataclass
class UniformCrossover(BaseCrossover[G, C]):
    """
    Uniform Crossover for binary genomes using new paradigm.
    
    Each gene has crossover_rate probability of coming from parent 1.
    """
    # --- DYNAMIC PARAMS ---
    crossover_rate: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        """Create one child from two binary parents."""
        # Generate crossover mask
        mask = jar.bernoulli(key, p=self.crossover_rate, shape=p1.bits.shape)
        
        # Create offspring
        child_bits = jnp.where(mask, p1.bits, p2.bits)
        
        return p1.replace(bits=child_bits)


# ==========================================
# REAL GENOME OPERATORS  
# ==========================================

@struct.dataclass
class GaussianMutation(BaseMutation[G, C]):
    """
    Gaussian Mutation for real genomes using new paradigm.
    
    Adds Gaussian noise to each gene with specified probability.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    mutation_strength: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Mutate one real genome."""
        key_noise, key_mask = jar.split(key)
        
        # Generate Gaussian noise
        noise = jar.normal(key_noise, shape=genome.values.shape) * self.mutation_strength
        
        # Generate mutation mask
        mutation_mask = jar.bernoulli(key_mask, p=self.mutation_rate, shape=genome.values.shape)
        
        # Apply mutation
        new_values = jnp.where(mutation_mask, genome.values + noise, genome.values)
        
        # Apply bounds if config has them
        if hasattr(config, 'bounds'):
            min_bound, max_bound = config.bounds
            new_values = jnp.clip(new_values, min_bound, max_bound)
        
        return genome.replace(values=new_values)


@struct.dataclass
class BlendCrossover(BaseCrossover[G, C]):
    """
    Blend Crossover (BLX-α) for real genomes using new paradigm.
    
    Creates offspring by sampling from extended intervals around parents.
    """
    # --- DYNAMIC PARAMS ---
    crossover_rate: float = 0.9
    alpha: float = 0.5
    
    def _cross_one(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        """Create one child from two real parents."""
        key_uniform, key_mask = jar.split(key)
        
        # Create crossover mask
        crossover_mask = jar.bernoulli(key_mask, p=self.crossover_rate, shape=p1.values.shape)
        
        # Calculate interval bounds
        min_vals = jnp.minimum(p1.values, p2.values)
        max_vals = jnp.maximum(p1.values, p2.values)
        interval_size = max_vals - min_vals
        
        # Extend interval by alpha
        extended_min = min_vals - self.alpha * interval_size
        extended_max = max_vals + self.alpha * interval_size
        
        # Sample offspring values
        child_values = jar.uniform(
            key_uniform, 
            shape=p1.values.shape,
            minval=extended_min, 
            maxval=extended_max
        )
        
        # Apply crossover mask (use p1 values where crossover doesn't occur)
        child_values = jnp.where(crossover_mask, child_values, p1.values)
        
        # Apply bounds if config has them
        if hasattr(config, 'bounds'):
            min_bound, max_bound = config.bounds
            child_values = jnp.clip(child_values, min_bound, max_bound)
        
        return p1.replace(values=child_values)


# ==========================================
# LINEAR GENOME OPERATORS (Your Design!)
# ==========================================

@struct.dataclass
class LinearMutation(BaseMutation[G, C]):
    """
    Linear GP Mutation using your paradigm from the notebook.
    
    Mutates opcodes and arguments with separate rates.
    """
    # --- DYNAMIC PARAMS ---
    op_rate: float = 0.1   # Prob to flip opcode
    arg_rate: float = 0.1  # Prob to flip argument

    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Mutate one linear genome."""
        k_op, k_arg, k_noise = jar.split(key, 3)

        # 1. Generate boolean masks
        mask_ops = jar.bernoulli(k_op, self.op_rate, genome.ops.shape)
        mask_args = jar.bernoulli(k_arg, self.arg_rate, genome.args.shape)

        # 2. Generate random noise
        noise_ops = jar.randint(k_noise, genome.ops.shape, 0, config.num_ops)
        
        # For args, allow references up to current position + inputs
        max_mem = config.num_inputs + config.length
        k_noise_args = jar.split(k_noise)[0]
        noise_args = jar.randint(k_noise_args, genome.args.shape, 0, max_mem)

        # 3. Apply mutations
        new_ops = jnp.where(mask_ops, noise_ops, genome.ops)
        new_args = jnp.where(mask_args, noise_args, genome.args)

        # 4. Construct and repair (autocorrect for topological validity)
        return genome.replace(ops=new_ops, args=new_args).autocorrect(config)


@struct.dataclass
class LinearCrossover(BaseCrossover[G, C]):
    """
    Linear GP Uniform Crossover using your paradigm from the notebook.
    
    Mixes opcodes and arguments with specified probability.
    """
    # --- DYNAMIC PARAMS ---
    mixing_ratio: float = 0.5  # 0.5 = balanced mix

    def _cross_one(self, key: chex.PRNGKey, p1: G, p2: G, config: C) -> G:
        """Create one child from two linear parents."""
        # Generate mixing mask
        mask = jar.bernoulli(key, self.mixing_ratio, p1.ops.shape)

        # Mix opcodes
        child_ops = jnp.where(mask, p1.ops, p2.ops)

        # Mix arguments (broadcast mask for coupling)
        mask_expanded = mask[:, None]
        child_args = jnp.where(mask_expanded, p1.args, p2.args)

        # Return child (no autocorrect needed for uniform crossover between valid parents)
        return p1.replace(ops=child_ops, args=child_args)


# ==========================================
# CATEGORICAL GENOME OPERATORS
# ==========================================

@struct.dataclass
class CategoryFlipMutation(BaseMutation[G, C]):
    """
    Category Flip Mutation for categorical genomes using new paradigm.
    
    Replaces categories with new random values.
    """
    # --- DYNAMIC PARAMS ---
    mutation_rate: float = 0.1
    
    def _mutate_one(self, key: chex.PRNGKey, genome: G, config: C) -> G:
        """Mutate one categorical genome."""
        key_mask, key_vals = jar.split(key)
        
        # Create mutation mask
        mutation_mask = jar.bernoulli(key_mask, p=self.mutation_rate, shape=genome.categories.shape)
        
        # Generate new categories
        new_categories = jar.randint(
            key_vals,
            shape=genome.categories.shape,
            minval=0,
            maxval=config.num_categories
        )
        
        # Apply mutations
        mutated_categories = jnp.where(mutation_mask, new_categories, genome.categories)
        
        return genome.replace(categories=mutated_categories)


__all__ = [
    "BitFlipMutation",
    "UniformCrossover", 
    "GaussianMutation",
    "BlendCrossover",
    "LinearMutation",
    "LinearCrossover",
    "CategoryFlipMutation"
]