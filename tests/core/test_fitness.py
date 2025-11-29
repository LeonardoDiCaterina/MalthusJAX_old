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
        evaluator = BinarySumFitnessEvaluator()
        bits = jnp.array([1, 0, 1, 1, 0])  # Sum = 3
        
        fitness = evaluator.evaluate_single(bits)
        assert fitness == 3.0

    def test_batch_evaluation(self, rng_key):
        """Test batch evaluation of binary genomes."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Create batch of binary genomes
        batch_size = 5
        length = 10
        keys = jr.split(rng_key, batch_size)
        config = BinaryGenomeConfig(length=length)
        
        population = jnp.array([
            BinaryGenome.random_init(key, config).bits 
            for key in keys
        ])
        
        fitness_values = evaluator.evaluate_batch(population)
        
        assert fitness_values.shape == (batch_size,)
        assert jnp.all(fitness_values >= 0)
        assert jnp.all(fitness_values <= length)

    def test_known_values(self):
        """Test evaluator on known binary patterns."""
        evaluator = BinarySumFitnessEvaluator()
        
        # All zeros
        all_zeros = jnp.zeros(5, dtype=jnp.int32)
        assert evaluator.evaluate_single(all_zeros) == 0.0
        
        # All ones
        all_ones = jnp.ones(5, dtype=jnp.int32)
        assert evaluator.evaluate_single(all_ones) == 5.0
        
        # Mixed pattern
        mixed = jnp.array([1, 0, 1, 0, 1])
        assert evaluator.evaluate_single(mixed) == 3.0

    @pytest.mark.jit
    def test_jit_compatibility(self):
        """Test that evaluator functions are JIT compilable."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Test single evaluation JIT
        bits = jnp.array([1, 0, 1, 1, 0])
        jit_single = jax.jit(evaluator.evaluate_single)
        fitness = jit_single(bits)
        assert fitness == 3.0
        
        # Test batch evaluation JIT
        population = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        jit_batch = jax.jit(evaluator.evaluate_batch)
        fitness_values = jit_batch(population)
        expected = jnp.array([2.0, 1.0, 3.0])
        assert jnp.allclose(fitness_values, expected)

    def test_pure_function_interface(self):
        """Test the pure function interface."""
        evaluator = BinarySumFitnessEvaluator()
        pure_fn = evaluator.get_tensor_fitness_function()
        
        # Test on single genome
        bits = jnp.array([1, 0, 1, 1, 0])
        fitness = pure_fn(bits)
        assert fitness == 3.0
        
        # Should be JIT compilable
        jit_pure_fn = jax.jit(pure_fn)
        fitness_jit = jit_pure_fn(bits)
        assert fitness == fitness_jit


@pytest.mark.slow
class TestFitnessPerformance:
    """Performance tests for fitness evaluators."""

    @pytest.mark.jit
    def test_large_batch_evaluation(self, rng_key):
        """Test evaluation on large batches."""
        batch_size = 1000
        
        # Binary sum evaluator
        binary_evaluator = BinarySumFitnessEvaluator()
        binary_population = jr.bernoulli(rng_key, 0.5, (batch_size, 50))
        binary_fitness = binary_evaluator.evaluate_batch(binary_population)
        assert binary_fitness.shape == (batch_size,)

    @pytest.mark.jit  
    def test_vmap_compatibility(self, rng_key):
        """Test that evaluators work with vmap."""
        # Test binary evaluator with vmap
        binary_evaluator = BinarySumFitnessEvaluator()
        population = jr.bernoulli(rng_key, 0.5, (10, 20))
        
        # Manual batch vs vmap should give same results
        manual_fitness = binary_evaluator.evaluate_batch(population)
        vmap_fitness = jax.vmap(binary_evaluator.evaluate_single)(population)
        
        assert jnp.allclose(manual_fitness, vmap_fitness)

    def test_evaluator_consistency(self, rng_key):
        """Test that different evaluation methods give consistent results."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Create test data
        config = BinaryGenomeConfig(length=10)
        genome = BinaryGenome.random_init(rng_key, config)
        
        # Single evaluation
        fitness1 = evaluator.evaluate_single(genome.bits)
        
        # Batch evaluation with single item
        fitness2 = evaluator.evaluate_batch(genome.bits.reshape(1, -1))[0]
        
        # Pure function evaluation
        pure_fn = evaluator.get_tensor_fitness_function()
        fitness3 = pure_fn(genome.bits)
        
        assert jnp.isclose(fitness1, fitness2)
        assert jnp.isclose(fitness1, fitness3)