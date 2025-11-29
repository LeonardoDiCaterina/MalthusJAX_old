"""
Tests for fitness evaluators (Level 1).

Tests binary ones evaluator which is the main working evaluator.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator as BinarySumFitnessEvaluator  
from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig

from tests.conftest import assert_jit_compilable


class TestBinarySumFitnessEvaluator:
    """Test binary sum fitness evaluator."""

    def test_single_evaluation(self):
        """Test evaluation of a single binary genome."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        genome = BinaryGenome(bits=jnp.array([1, 0, 1, 1, 0]))  # Sum = 3
        
        fitness = evaluator.evaluate_single(genome)
        assert fitness == 3.0

    def test_batch_evaluation(self, rng_key):
        """Test batch evaluation of binary genomes."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Create batch of binary genomes
        batch_size = 5
        length = 10
        genome_config = BinaryGenomeConfig(length=length)
        
        # Create population using NEW paradigm
        population = BinaryPopulation.init_random(rng_key, genome_config, size=batch_size)
        
        fitness_values = evaluator.evaluate_batch(population)
        
        assert fitness_values.shape == (batch_size,)
        assert jnp.all(fitness_values >= 0)
        assert jnp.all(fitness_values <= length)

    def test_known_values(self):
        """Test evaluator on known binary patterns."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        
        # All zeros
        all_zeros = BinaryGenome(bits=jnp.zeros(5, dtype=jnp.int32))
        assert evaluator.evaluate_single(all_zeros) == 0.0
        
        # All ones
        all_ones = BinaryGenome(bits=jnp.ones(5, dtype=jnp.int32))
        assert evaluator.evaluate_single(all_ones) == 5.0
        
        # Mixed pattern
        mixed = BinaryGenome(bits=jnp.array([1, 0, 1, 0, 1]))
        assert evaluator.evaluate_single(mixed) == 3.0

    @pytest.mark.jit
    def test_jit_compatibility(self):
        """Test that evaluator functions are JIT compilable."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Test single evaluation JIT
        genome = BinaryGenome(bits=jnp.array([1, 0, 1, 1, 0]))
        jit_single = jax.jit(evaluator.evaluate_single)
        fitness = jit_single(genome)
        assert fitness == 3.0
        
        # Test batch evaluation JIT with population
        genome_config = BinaryGenomeConfig(length=3)
        population = BinaryPopulation.init_random(jax.random.PRNGKey(42), genome_config, size=3)
        # Override with known values for testing
        population = population.replace(
            genes=population.genes.replace(
                bits=jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
            )
        )
        jit_batch = jax.jit(evaluator.evaluate_batch)
        fitness_values = jit_batch(population)
        expected = jnp.array([2.0, 1.0, 3.0])
        assert jnp.allclose(fitness_values, expected)

    def test_pure_function_interface(self):
        """Test the NEW paradigm batch evaluation interface."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Create a small population for batch evaluation
        genome_config = BinaryGenomeConfig(length=5)
        population = BinaryPopulation.init_random(jax.random.PRNGKey(42), genome_config, size=2)
        # Override with known values
        population = population.replace(
            genes=population.genes.replace(
                bits=jnp.array([[1, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
            )
        )
        
        fitness_values = evaluator.evaluate_batch(population)
        expected = jnp.array([3.0, 2.0])
        assert jnp.allclose(fitness_values, expected)
        
        # Should be JIT compilable
        jit_batch = jax.jit(evaluator.evaluate_batch)
        fitness_jit = jit_batch(population)
        assert jnp.allclose(fitness_values, fitness_jit)


@pytest.mark.slow
class TestFitnessPerformance:
    """Performance tests for fitness evaluators."""

    @pytest.mark.jit
    def test_large_batch_evaluation(self, rng_key):
        """Test evaluation on large batches."""
        batch_size = 100  # Smaller for faster tests
        length = 20
        
        # Binary sum evaluator
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        binary_evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Create proper population
        genome_config = BinaryGenomeConfig(length=length)
        population = BinaryPopulation.init_random(rng_key, genome_config, size=batch_size)
        
        binary_fitness = binary_evaluator.evaluate_batch(population)
        assert binary_fitness.shape == (batch_size,)
        assert jnp.all(binary_fitness >= 0)
        assert jnp.all(binary_fitness <= length)

    @pytest.mark.jit
    def test_vmap_compatibility(self, rng_key):
        """Test that evaluators work with vmap."""
        # Test binary evaluator with vmap
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        binary_evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Create population
        genome_config = BinaryGenomeConfig(length=10)
        population = BinaryPopulation.init_random(rng_key, genome_config, size=5)
        
        # Manual batch vs vmap should give same results
        manual_fitness = binary_evaluator.evaluate_batch(population)
        vmap_fitness = jax.vmap(binary_evaluator.evaluate_single)(population.genes)
        
        assert jnp.allclose(manual_fitness, vmap_fitness)

    def test_evaluator_consistency(self, rng_key):
        """Test that different evaluation methods give consistent results."""
        from malthusjax.core.fitness.binary_evaluators import BinarySumConfig
        from malthusjax.core.genome.binary_genome import BinaryPopulation
        config = BinarySumConfig(maximize=True)
        evaluator = BinarySumFitnessEvaluator(config=config)
        
        # Create test data
        genome_config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config=genome_config)
        
        # Single evaluation
        fitness1 = evaluator.evaluate_single(genome)
        
        # Batch evaluation with single item
        population = BinaryPopulation.init_random(rng_key, genome_config, size=1)
        population = population.replace(genes=population.genes.replace(bits=genome.bits.reshape(1, -1)))
        fitness2 = evaluator.evaluate_batch(population)[0]
        
        assert jnp.isclose(fitness1, fitness2)