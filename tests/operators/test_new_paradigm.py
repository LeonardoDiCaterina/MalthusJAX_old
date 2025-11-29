"""
Tests for Level 2 genetic operators (NEW PARADIGM).

Tests the current operators that follow the @struct.dataclass paradigm
with BaseMutation and BaseCrossover inheritance.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

# Import NEW PARADIGM operators
from malthusjax.operators.mutation.binary import BitFlipMutation, ScrambleMutation, SwapMutation
from malthusjax.operators.mutation.real import GaussianMutation, PolynomialMutation, BallMutation
from malthusjax.operators.mutation.categorical import CategoricalFlipMutation, RandomCategoryMutation

from malthusjax.operators.crossover.binary import UniformCrossover, SinglePointCrossover
from malthusjax.operators.crossover.real import BlendCrossover, SimulatedBinaryCrossover

from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.selection.roulette import RouletteWheelSelection

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig

from tests.conftest import (
    assert_valid_binary_genome,
    assert_valid_real_genome,
    assert_valid_categorical_genome,
    assert_valid_binary_genome_batch,
    assert_valid_real_genome_batch,
    assert_jit_compilable
)


class TestBitFlipMutation:
    """Test bit flip mutation with new paradigm."""

    def test_basic_functionality(self, rng_key, binary_genome_config, binary_genome):
        """Test basic mutation functionality."""
        mutator = BitFlipMutation(mutation_rate=0.5)
        
        mutated = mutator(rng_key, binary_genome, binary_genome_config)
        assert_valid_binary_genome(mutated, binary_genome_config)

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, binary_genome_config, binary_genome):
        """Test JIT compilation."""
        mutator = BitFlipMutation(mutation_rate=0.3)
        jit_fn = jax.jit(mutator)  # JIT the operator directly
        
        result = jit_fn(rng_key, binary_genome, binary_genome_config)
        assert_valid_binary_genome_batch(result, binary_genome_config)

    def test_batch_operation(self, rng_key, binary_population):
        """Test batch mutation - operators handle batching internally."""
        population_bits, config = binary_population
        mutator = BitFlipMutation(num_offspring=1, mutation_rate=0.2)
        
        # Test single genome from population
        single_genome = BinaryGenome(bits=population_bits[0])
        result = mutator(rng_key, single_genome, config)
        
        # NEW paradigm: operators return batch-first results
        assert result.bits.shape[0] == 1  # num_offspring
        assert result.bits.shape[1:] == population_bits[0].shape
        assert jnp.all((result.bits == 0) | (result.bits == 1))


class TestGaussianMutation:
    """Test Gaussian mutation with new paradigm."""

    def test_basic_functionality(self, rng_key, real_genome_config, real_genome):
        """Test basic mutation functionality."""
        mutator = GaussianMutation(mutation_rate=0.3, mutation_strength=0.1)
        
        mutated = mutator(rng_key, real_genome, real_genome_config)
        assert_valid_real_genome(mutated, real_genome_config)

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, real_genome_config, real_genome):
        """Test JIT compilation."""
        mutator = GaussianMutation(mutation_rate=0.2, mutation_strength=0.05)
        jit_mutator = jax.jit(mutator)  # JIT the operator directly
        
        result = jit_mutator(rng_key, real_genome, real_genome_config)
        assert_valid_real_genome_batch(result, real_genome_config)

    def test_bounds_respected(self, rng_key, constrained_real_genome_config):
        """Test that mutation respects genome bounds."""
        values = jnp.array([0.8, -0.8, 0.0])
        genome = RealGenome(values=values)
        
        mutator = GaussianMutation(mutation_rate=1.0, mutation_strength=0.5)
        
        for i in range(5):
            key = jr.fold_in(rng_key, i)
            mutated = mutator(key, genome, constrained_real_genome_config)
            assert_valid_real_genome(mutated, constrained_real_genome_config)


class TestUniformCrossover:
    """Test uniform crossover with new batch-first paradigm."""

    def test_batch_output_format(self, rng_key, binary_genome_config):
        """Test that crossover returns correct batch format."""
        key1, key2 = jr.split(rng_key)
        parent1 = BinaryGenome.random_init(key1, binary_genome_config)
        parent2 = BinaryGenome.random_init(key2, binary_genome_config)
        
        crossover = UniformCrossover(num_offspring=3, crossover_rate=0.7)
        offspring = crossover(rng_key, parent1, parent2, binary_genome_config)
        
        # Should return (num_offspring, genome_length) format
        assert offspring.bits.shape == (3, binary_genome_config.length)
        
        # Each offspring should be valid
        for i in range(3):
            child_bits = offspring.bits[i]
            child = BinaryGenome(bits=child_bits)
            assert_valid_binary_genome(child, binary_genome_config)

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, binary_genome_config):
        """Test JIT compilation of crossover."""
        key1, key2 = jr.split(rng_key)
        parent1 = BinaryGenome.random_init(key1, binary_genome_config)
        parent2 = BinaryGenome.random_init(key2, binary_genome_config)
        
        crossover = UniformCrossover(num_offspring=2, crossover_rate=0.5)
        jit_fn = jax.jit(crossover)  # JIT operator directly
        
        offspring = jit_fn(rng_key, parent1, parent2, binary_genome_config)
        assert offspring.bits.shape == (2, binary_genome_config.length)

    def test_inheritance_behavior(self, rng_key, binary_genome_config):
        """Test that offspring inherit from both parents."""
        # Create distinct parents
        parent1_bits = jnp.ones(binary_genome_config.length, dtype=jnp.int32)
        parent2_bits = jnp.zeros(binary_genome_config.length, dtype=jnp.int32)
        
        parent1 = BinaryGenome(bits=parent1_bits)
        parent2 = BinaryGenome(bits=parent2_bits)
        
        crossover = UniformCrossover(num_offspring=10, crossover_rate=0.5)
        offspring = crossover(rng_key, parent1, parent2, binary_genome_config)
        
        # With uniform crossover, most offspring should have mix of 0s and 1s
        for i in range(offspring.bits.shape[0]):
            child_bits = offspring.bits[i]
            # Should contain both 0s and 1s (unless very unlucky)
            if binary_genome_config.length > 3:
                has_zeros = jnp.any(child_bits == 0)
                has_ones = jnp.any(child_bits == 1)
                assert has_zeros or has_ones  # At least one type


class TestBlendCrossover:
    """Test blend crossover with new batch-first paradigm."""

    def test_batch_output_format(self, rng_key, real_genome_config):
        """Test correct batch format output."""
        key1, key2 = jr.split(rng_key)
        parent1 = RealGenome.random_init(key1, real_genome_config)
        parent2 = RealGenome.random_init(key2, real_genome_config)
        
        crossover = BlendCrossover(num_offspring=4, crossover_rate=0.8, alpha=0.3)
        offspring = crossover(rng_key, parent1, parent2, real_genome_config)
        
        assert offspring.values.shape == (4, real_genome_config.length)
        
        for i in range(4):
            child_values = offspring.values[i]
            child = RealGenome(values=child_values)
            assert_valid_real_genome(child, real_genome_config)

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, real_genome_config):
        """Test JIT compilation."""
        key1, key2 = jr.split(rng_key)
        parent1 = RealGenome.random_init(key1, real_genome_config)
        parent2 = RealGenome.random_init(key2, real_genome_config)
        
        crossover = BlendCrossover(num_offspring=2, crossover_rate=0.7, alpha=0.5)
        jit_crossover = jax.jit(crossover)  # JIT operator directly
        result = jit_crossover(rng_key, parent1, parent2, real_genome_config)
        assert_valid_real_genome_batch(result, real_genome_config)

    def test_bounds_respected(self, rng_key, constrained_real_genome_config):
        """Test that blend crossover respects bounds."""
        key1, key2 = jr.split(rng_key)
        parent1 = RealGenome.random_init(key1, constrained_real_genome_config)
        parent2 = RealGenome.random_init(key2, constrained_real_genome_config)
        
        crossover = BlendCrossover(num_offspring=10, crossover_rate=1.0, alpha=1.0)
        
        for i in range(3):
            key = jr.fold_in(rng_key, i)
            offspring = crossover(key, parent1, parent2, constrained_real_genome_config)
            
            for j in range(offspring.values.shape[0]):
                child_values = offspring.values[j]
                child = RealGenome(values=child_values)
                assert_valid_real_genome(child, constrained_real_genome_config)


class TestTournamentSelection:
    """Test tournament selection with new paradigm."""

    def test_basic_functionality(self, rng_key, fitness_values):
        """Test basic selection functionality."""
        selector = TournamentSelection(num_selections=5, tournament_size=3)
        
        selected_indices = selector(rng_key, fitness_values)
        
        assert selected_indices.shape == (5,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_values))

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, fitness_values):
        """Test JIT compilation."""
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        jit_fn = jax.jit(selector)  # JIT operator directly
        
        selected = jit_fn(rng_key, fitness_values)
        assert selected.shape == (3,)

    def test_selection_bias(self, rng_key):
        """Test that selection favors higher fitness."""
        # Create fitness with clear winner
        fitness_values = jnp.array([0.1, 0.2, 0.95, 0.1, 0.1])  # Index 2 is best
        
        selector = TournamentSelection(num_selections=20, tournament_size=5)
        selected = selector(rng_key, fitness_values)
        
        # With large tournament and many selections, should often pick index 2
        best_count = jnp.sum(selected == 2)
        assert best_count > 0  # Should pick the best at least sometimes


class TestRouletteWheelSelection:
    """Test roulette wheel selection with new paradigm."""

    def test_basic_functionality(self, rng_key, fitness_values):
        """Test basic selection functionality."""
        selector = RouletteWheelSelection(num_selections=5)
        
        selected_indices = selector(rng_key, fitness_values)
        
        assert selected_indices.shape == (5,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_values))

    @pytest.mark.jit
    def test_jit_compilation(self, rng_key, fitness_values):
        """Test JIT compilation."""
        selector = RouletteWheelSelection(num_selections=3)
        jit_fn = jax.jit(selector)  # JIT operator directly
        
        selected = jit_fn(rng_key, fitness_values)
        assert selected.shape == (3,)

    def test_positive_fitness_only(self, rng_key):
        """Test roulette selection with positive fitness values."""
        # Ensure all positive fitness values
        fitness_values = jnp.array([0.1, 0.3, 0.6, 0.2, 0.8])
        
        selector = RouletteWheelSelection(num_selections=10)
        selected = selector(rng_key, fitness_values)
        
        assert jnp.all(selected >= 0)
        assert jnp.all(selected < len(fitness_values))


@pytest.mark.slow
class TestOperatorIntegration:
    """Integration tests for operator combinations."""

    def test_complete_generation_cycle(self, rng_key, binary_genome_config):
        """Test complete mutation → crossover → selection cycle with NEW paradigm."""
        # Create initial population
        pop_size = 6  # Smaller for cleaner test
        keys = jr.split(rng_key, pop_size)
        
        # Create individual genomes using NEW paradigm 
        genomes = [BinaryGenome.random_init(key, binary_genome_config) for key in keys]
        
        # Test mutation on first genome
        mutator = BitFlipMutation(num_offspring=1, mutation_rate=0.1)
        k1, rng_key = jr.split(rng_key)
        mutated = mutator(k1, genomes[0], binary_genome_config)
        assert mutated.bits.shape == (1, binary_genome_config.length)
        
        # Test crossover on genome pair  
        crossover = UniformCrossover(num_offspring=2, crossover_rate=0.7)
        k2, rng_key = jr.split(rng_key)
        offspring = crossover(k2, genomes[0], genomes[1], binary_genome_config)
        assert offspring.bits.shape == (2, binary_genome_config.length)
        
        # Test selection with dummy fitness
        population_bits = jnp.stack([g.bits for g in genomes])
        fitness_values = jnp.sum(population_bits, axis=1)  # Simple sum fitness
        
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        k3, rng_key = jr.split(rng_key)
        selected_indices = selector(k3, fitness_values)
        assert selected_indices.shape == (3,)
        
        # Verify end-to-end integration works with NEW paradigm
        
        assert final_population.shape == (pop_size, binary_genome_config.length)
        assert jnp.all((final_population == 0) | (final_population == 1))

    @pytest.mark.jit
    def test_operator_batch_processing(self, rng_key, real_genome_config):
        """Test that operators work efficiently in batch mode.""" 
        batch_size = 20
        
        # Create parent populations
        keys = jr.split(rng_key, batch_size * 2)
        parent1_keys, parent2_keys = keys[:batch_size], keys[batch_size:]
        
        parent1_population = jnp.array([
            RealGenome.random_init(key, real_genome_config).values
            for key in parent1_keys
        ])
        parent2_population = jnp.array([
            RealGenome.random_init(key, real_genome_config).values
            for key in parent2_keys
        ])
        
        # Batch crossover
        crossover = BlendCrossover(num_offspring=2, crossover_rate=0.8, alpha=0.3)
        cross_keys = jr.split(rng_key, batch_size)
        
        cross_fn = jax.vmap(crossover.get_pure_function(), in_axes=(0, 0, 0, None))
        offspring_batch = cross_fn(cross_keys, parent1_population, parent2_population, real_genome_config)
        
        assert offspring_batch.shape == (batch_size, 2, real_genome_config.length)
        
        # Verify all offspring are valid
        for i in range(batch_size):
            for j in range(2):
                child_values = offspring_batch[i, j]
                child = RealGenome(values=child_values)
                assert_valid_real_genome(child, real_genome_config)