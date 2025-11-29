"""
Tests for NEW paradigm operators using @struct.dataclass.

Tests the actual current architecture with proper operator signatures.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.mutation.real import BallMutation
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.crossover.real import BlendCrossover
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.selection.roulette import RouletteWheelSelection


class TestNewMutationOperators:
    """Test NEW paradigm mutation operators."""

    def test_bitflip_mutation_basic(self, rng_key, binary_genome_config):
        """Test basic BitFlipMutation functionality."""
        genome = BinaryGenome.random_init(rng_key, binary_genome_config)
        mutator = BitFlipMutation(mutation_rate=0.5, num_offspring=1)
        
        # Test basic call
        mutated = mutator(rng_key, genome, binary_genome_config)
        
        # Should return BinaryGenome with shape (num_offspring, genome_shape)
        assert hasattr(mutated, 'bits')
        assert mutated.bits.shape == (1, binary_genome_config.length)
        
        # Values should be binary
        assert jnp.all((mutated.bits == 0) | (mutated.bits == 1))

    def test_bitflip_mutation_multiple_offspring(self, rng_key, binary_genome_config):
        """Test BitFlipMutation with multiple offspring."""
        genome = BinaryGenome.random_init(rng_key, binary_genome_config)
        mutator = BitFlipMutation(mutation_rate=0.2, num_offspring=3)
        
        mutated = mutator(rng_key, genome, binary_genome_config)
        
        # Should produce 3 offspring
        assert mutated.bits.shape == (3, binary_genome_config.length)
        assert jnp.all((mutated.bits == 0) | (mutated.bits == 1))

    def test_ball_mutation_basic(self, rng_key, real_genome_config):
        """Test basic BallMutation functionality.""" 
        genome = RealGenome.random_init(rng_key, real_genome_config)
        mutator = BallMutation(mutation_strength=0.1, num_offspring=1)
        
        mutated = mutator(rng_key, genome, real_genome_config)
        
        # Should return RealGenome with shape (num_offspring, genome_shape)
        assert hasattr(mutated, 'values')
        assert mutated.values.shape == (1, real_genome_config.length)
        
        # Values should be within bounds
        assert jnp.all(mutated.values >= real_genome_config.low)
        assert jnp.all(mutated.values <= real_genome_config.high)


class TestNewCrossoverOperators:
    """Test NEW paradigm crossover operators."""

    def test_uniform_crossover_basic(self, rng_key, binary_genome_config):
        """Test basic UniformCrossover functionality."""
        key1, key2 = jr.split(rng_key)
        parent1 = BinaryGenome.random_init(key1, binary_genome_config)
        parent2 = BinaryGenome.random_init(key2, binary_genome_config)
        
        crossover = UniformCrossover(crossover_rate=0.5, num_offspring=2)
        
        offspring = crossover(rng_key, parent1, parent2, binary_genome_config)
        
        # Should return offspring with shape (num_offspring, genome_shape)
        assert hasattr(offspring, 'bits')
        assert offspring.bits.shape == (2, binary_genome_config.length)
        assert jnp.all((offspring.bits == 0) | (offspring.bits == 1))

    def test_blend_crossover_basic(self, rng_key, real_genome_config):
        """Test basic BlendCrossover functionality."""
        key1, key2 = jr.split(rng_key)
        parent1 = RealGenome.random_init(key1, real_genome_config)
        parent2 = RealGenome.random_init(key2, real_genome_config)
        
        crossover = BlendCrossover(alpha=0.1, num_offspring=2)
        
        offspring = crossover(rng_key, parent1, parent2, real_genome_config)
        
        # Should return offspring with shape (num_offspring, genome_shape)
        assert hasattr(offspring, 'values')
        assert offspring.values.shape == (2, real_genome_config.length)
        
        # Values should be within bounds
        assert jnp.all(offspring.values >= real_genome_config.low)
        assert jnp.all(offspring.values <= real_genome_config.high)


class TestNewSelectionOperators:
    """Test NEW paradigm selection operators."""

    def test_tournament_selection_basic(self, rng_key):
        """Test basic TournamentSelection functionality."""
        fitness_values = jnp.array([1.0, 3.0, 2.0, 5.0, 0.5])
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        
        selected_indices = selector(rng_key, fitness_values)
        
        # Should return correct number of indices
        assert selected_indices.shape == (3,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_values))

    def test_roulette_wheel_selection_basic(self, rng_key):
        """Test basic RouletteWheelSelection functionality."""
        # Use positive fitness values for roulette wheel
        fitness_values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        selector = RouletteWheelSelection(num_selections=4)
        
        selected_indices = selector(rng_key, fitness_values)
        
        # Should return correct number of indices
        assert selected_indices.shape == (4,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_values))


class TestOperatorIntegration:
    """Test integration between different NEW operators."""

    def test_binary_evolution_step(self, rng_key, binary_genome_config):
        """Test a complete binary evolution step using NEW operators."""
        pop_size = 6
        
        # Create population
        keys = jr.split(rng_key, pop_size)
        population = [BinaryGenome.random_init(key, binary_genome_config) for key in keys]
        
        # Evaluate fitness (simple sum of bits)
        fitness_values = jnp.array([jnp.sum(genome.bits) for genome in population])
        
        # Selection
        key1, key2, key3 = jr.split(rng_key, 3)
        selector = TournamentSelection(num_selections=pop_size, tournament_size=2)
        selected_indices = selector(key1, fitness_values)
        
        # Crossover (pairwise)
        crossover = UniformCrossover(crossover_rate=0.8, num_offspring=1)
        new_population = []
        
        for i in range(0, pop_size - 1, 2):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[i + 1]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            offspring = crossover(key2, parent1, parent2, binary_genome_config)
            # Take first offspring from batch
            child = BinaryGenome(bits=offspring.bits[0])
            new_population.append(child)
        
        # If odd population, add one more
        if len(new_population) < pop_size:
            new_population.append(population[selected_indices[-1]])
            
        assert len(new_population) == pop_size
        
        # Mutation
        mutator = BitFlipMutation(mutation_rate=0.05, num_offspring=1)
        mut_keys = jr.split(key3, pop_size)
        
        final_population = []
        for i, genome in enumerate(new_population):
            mutated = mutator(mut_keys[i], genome, binary_genome_config)
            # Take first offspring from batch  
            final_genome = BinaryGenome(bits=mutated.bits[0])
            final_population.append(final_genome)
        
        assert len(final_population) == pop_size
        for genome in final_population:
            assert genome.bits.shape == (binary_genome_config.length,)
            assert jnp.all((genome.bits == 0) | (genome.bits == 1))


class TestDataclassProperties:
    """Test that operators behave as proper dataclasses."""

    def test_operator_immutability(self):
        """Test that operators are immutable."""
        mutator = BitFlipMutation(mutation_rate=0.1, num_offspring=1)
        
        # Should not be able to modify
        with pytest.raises((AttributeError, TypeError)):
            mutator.mutation_rate = 0.5
            
    def test_operator_equality(self):
        """Test operator equality and hashing."""
        mutator1 = BitFlipMutation(mutation_rate=0.1, num_offspring=2)
        mutator2 = BitFlipMutation(mutation_rate=0.1, num_offspring=2)
        mutator3 = BitFlipMutation(mutation_rate=0.2, num_offspring=2)
        
        assert mutator1 == mutator2
        assert mutator1 != mutator3
        assert hash(mutator1) == hash(mutator2)

    def test_operator_repr(self):
        """Test operator string representation."""
        mutator = BitFlipMutation(mutation_rate=0.15, num_offspring=3)
        repr_str = repr(mutator)
        
        assert "BitFlipMutation" in repr_str
        assert "mutation_rate=0.15" in repr_str
        assert "num_offspring=3" in repr_str


@pytest.mark.jit
class TestJITCompilation:
    """Test JAX JIT compilation with NEW operators."""
    
    def test_mutation_jit_compilation(self, rng_key, binary_genome_config):
        """Test that NEW mutation operators can be JIT compiled."""
        genome = BinaryGenome.random_init(rng_key, binary_genome_config)
        mutator = BitFlipMutation(mutation_rate=0.1, num_offspring=1)
        
        # JIT compile the operator call  
        @jax.jit
        def mutate_jit(key, genome, config):
            return mutator(key, genome, config)
        
        mutated = mutate_jit(rng_key, genome, binary_genome_config)
        
        assert mutated.bits.shape == (1, binary_genome_config.length)
        assert jnp.all((mutated.bits == 0) | (mutated.bits == 1))

    def test_selection_jit_compilation(self, rng_key):
        """Test that NEW selection operators can be JIT compiled."""
        fitness_values = jnp.array([1.0, 3.0, 2.0, 5.0, 0.5])
        selector = TournamentSelection(num_selections=3, tournament_size=2)
        
        # JIT compile the operator call
        @jax.jit  
        def select_jit(key, fitness):
            return selector(key, fitness)
        
        selected_indices = select_jit(rng_key, fitness_values)
        
        assert selected_indices.shape == (3,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_values))