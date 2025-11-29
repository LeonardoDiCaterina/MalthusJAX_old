"""
Tests for Level 2 genetic operators (NEW PARADIGM).
"""
import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.mutation.real import GaussianMutation
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.selection.tournament import TournamentSelection

from malthusjax.core.genome.binary_genome import BinaryGenome
from malthusjax.core.genome.real_genome import RealGenome

class TestBitFlipMutation:
    def test_basic_functionality(self, rng_key, binary_genome_config, binary_genome):
        mutator = BitFlipMutation(mutation_rate=0.5)
        # Direct call
        mutated = mutator(rng_key, binary_genome, binary_genome_config)
        # Output is batch (1, L)
        assert mutated.bits.shape == (1, binary_genome_config.length)

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, binary_genome_config, binary_genome):
        mutator = BitFlipMutation(mutation_rate=0.3)
        # JIT the operator directly
        jit_fn = jax.jit(mutator)
        mutated = jit_fn(rng_key, binary_genome, binary_genome_config)
        assert mutated.bits.shape == (1, binary_genome_config.length)

class TestGaussianMutation:
    def test_basic_functionality(self, rng_key, real_genome_config, real_genome):
        mutator = GaussianMutation(mutation_rate=0.3)
        mutated = mutator(rng_key, real_genome, real_genome_config)
        assert mutated.values.shape == (1, real_genome_config.length)

class TestOperatorIntegration:
    def test_complete_generation_cycle(self, rng_key, binary_genome_config):
        """Test complete cycle with correct batch handling."""
        pop_size = 10
        k1, k2, k3 = jr.split(rng_key, 3)
        
        # Init
        population = BinaryGenome.create_population(k1, binary_genome_config, pop_size)
        
        # Selection (Mock - just select all for this operator test)
        # We don't need to test selection logic here, just data flow
        indices = jnp.arange(pop_size)
        
        # Crossover
        crossover = UniformCrossover(num_offspring=1, crossover_rate=0.5)
        # Pair up 0-1, 2-3... (Split population into two halves of parents)
        # Use tree_map to slice the Pytree safely
        p1 = jax.tree_util.tree_map(lambda x: x[0::2], population)
        p2 = jax.tree_util.tree_map(lambda x: x[1::2], population)
        
        # Returns (1, Half_Pop, L)
        offspring = crossover(k2, p1, p2, binary_genome_config)
        
        # Flatten -> (Half_Pop, L)
        offspring_flat = jax.tree_util.tree_map(lambda x: x.squeeze(0), offspring)
        
        # Mutation
        mutator = BitFlipMutation(num_offspring=2, mutation_rate=0.1)
        # Input: (Half_Pop, L) -> Output: (2, Half_Pop, L)
        mutants = mutator(k3, offspring_flat, binary_genome_config)
        
        # Reshape to (Pop_Size, L)
        # (2, 5, 10) -> (10, 10)
        final_bits = mutants.bits.reshape(pop_size, binary_genome_config.length)
        
        # This replaces the variable 'final_population' that was missing
        assert final_bits.shape == (pop_size, binary_genome_config.length)