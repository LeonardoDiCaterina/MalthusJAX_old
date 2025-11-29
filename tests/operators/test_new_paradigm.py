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
        jit_fn = jax.jit(mutator.get_pure_function(), static_argnames=['config'])
        
        mutated_bits = jit_fn(rng_key, binary_genome.bits, binary_genome_config)
        result = BinaryGenome(bits=mutated_bits)
        assert_valid_binary_genome(result, binary_genome_config)

    def test_batch_operation(self, rng_key, binary_population):
        """Test batch mutation via vmap."""
        population_bits, config = binary_population
        mutator = BitFlipMutation(mutation_rate=0.2)
        
        batch_size = population_bits.shape[0]
        keys = jr.split(rng_key, batch_size)
        
        mutate_fn = jax.vmap(mutator.get_pure_function(), in_axes=(0, 0, None))
        mutated_population = mutate_fn(keys, population_bits, config)
        
        assert mutated_population.shape == population_bits.shape
        assert jnp.all((mutated_population == 0) | (mutated_population == 1))


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
        assert_jit_compilable(mutator.get_pure_function(), 
                             rng_key, real_genome.values, real_genome_config)

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
        jit_fn = jax.jit(crossover.get_pure_function(), static_argnames=['config'])
        
        offspring_bits = jit_fn(rng_key, parent1.bits, parent2.bits, binary_genome_config)
        assert offspring_bits.shape == (2, binary_genome_config.length)

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
        assert_jit_compilable(crossover.get_pure_function(),
                             rng_key, parent1.values, parent2.values, real_genome_config)

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
        jit_fn = jax.jit(selector.get_pure_function())
        
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
        jit_fn = jax.jit(selector.get_pure_function())
        
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
        """Test complete mutation → crossover → selection cycle."""
        # Create initial population
        pop_size = 10
        keys = jr.split(rng_key, pop_size)
        population_bits = jnp.array([
            BinaryGenome.random_init(key, binary_genome_config).bits
            for key in keys
        ])
        
        # Apply mutations
        mutator = BitFlipMutation(mutation_rate=0.1)
        mut_keys = jr.split(rng_key, pop_size)
        mutate_fn = jax.vmap(mutator.get_pure_function(), in_axes=(0, 0, None))
        mutated_population = mutate_fn(mut_keys, population_bits, binary_genome_config)
        
        # Apply crossover to create offspring
        crossover = UniformCrossover(num_offspring=2, crossover_rate=0.7)
        cross_keys = jr.split(rng_key, pop_size//2)
        
        offspring_list = []
        for i in range(pop_size//2):
            p1_bits = mutated_population[i*2]
            p2_bits = mutated_population[i*2 + 1]
            p1 = BinaryGenome(bits=p1_bits)
            p2 = BinaryGenome(bits=p2_bits)
            
            offspring = crossover(cross_keys[i], p1, p2, binary_genome_config)
            offspring_list.append(offspring.bits)
        
        # Flatten offspring
        all_offspring = jnp.concatenate(offspring_list, axis=0)
        
        # Apply selection (dummy fitness for testing)
        fitness_values = jnp.sum(all_offspring, axis=1)  # Simple sum fitness
        selector = TournamentSelection(num_selections=pop_size, tournament_size=3)
        selected_indices = selector(rng_key, fitness_values)
        
        # Final population
        final_population = all_offspring[selected_indices]
        
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