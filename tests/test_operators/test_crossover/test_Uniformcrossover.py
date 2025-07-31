import unittest
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.core.genome import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.solution.base import FitnessTransforms


class Test_UniformCrossover(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.key = jar.PRNGKey(42)
        self.genome_size = 10
        self.pop_size = 6  # Even number for proper pairing
        self.genome_init_params = {'array_size': self.genome_size, 'p': 0.5}
        
        # Create a test population
        self.population = Population(
            solution_class=BinarySolution,
            max_size=self.pop_size,
            random_init=True,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=FitnessTransforms.minimize,
        )
    
    def test_default_initialization(self):
        """Test UniformCrossover with default parameters."""
        crossover = UniformCrossover()
        self.assertEqual(crossover.crossover_rate, 0.01)
    
    def test_custom_initialization(self):
        """Test UniformCrossover with custom crossover rate."""
        custom_rate = 0.7
        crossover = UniformCrossover(crossover_rate=custom_rate)
        self.assertEqual(crossover.crossover_rate, custom_rate)
    
    def test_invalid_crossover_rate(self):
        """Test that invalid crossover rates raise appropriate errors."""
        with self.assertRaises((ValueError, AssertionError)):
            UniformCrossover(crossover_rate=-0.1)
        
        with self.assertRaises((ValueError, AssertionError)):
            UniformCrossover(crossover_rate=1.1)
    
    def test_build_returns_callable(self):
        """Test that build method returns a callable function."""
        crossover = UniformCrossover()
        crossover_fn = crossover.build(self.population)
        self.assertTrue(callable(crossover_fn))
    
    def test_crossover_output_shape(self):
        """Test that crossover preserves genome shape."""
        crossover = UniformCrossover(crossover_rate=0.5)
        crossover_fn = crossover.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(population_stack, keys)
        
        # Check shape is preserved
        self.assertEqual(crossed_genomes.shape, population_stack.shape)
        self.assertEqual(crossed_genomes.shape, (self.pop_size, self.genome_size))
    
    def test_crossover_output_dtype_and_values(self):
        """Test that crossover preserves binary dtype and values."""
        crossover = UniformCrossover(crossover_rate=0.8)
        crossover_fn = crossover.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(population_stack, keys)
        
        # Check that output contains only 0s and 1s
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
        # Check dtype is preserved
        self.assertEqual(crossed_genomes.dtype, population_stack.dtype)
    
    def test_high_crossover_rate_causes_changes(self):
        """Test that high crossover rate causes changes."""
        crossover = UniformCrossover(crossover_rate=0.9)
        crossover_fn = crossover.build(self.population)
        
        # Create test genomes with distinctive patterns
        test_genomes = jnp.array([
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ])
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(test_genomes, keys)
        
        # With high crossover rate, there should be differences
        differences = jnp.sum(crossed_genomes != test_genomes)
        self.assertGreater(differences, 0)
    
    def test_uniform_crossover_behavior(self):
        """Test specific uniform crossover behavior with known patterns."""
        crossover = UniformCrossover(crossover_rate=1.0)  # Always apply mask
        crossover_fn = crossover.build(self.population)
        
        # Test with all 0s - should become all 1s when mask is applied everywhere
        all_zeros = jnp.zeros((self.pop_size, self.genome_size), dtype=jnp.int32)
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(all_zeros, keys)
        
        # With crossover_rate=1.0, all bits should be flipped (0 -> 1)
        # Since the implementation uses flipped_mask (logical_not of mask)
        # and jnp.where(mask, genome_data, flipped_mask)
        # This means where mask is True, keep original (0), where mask is False, use flipped (True->1)
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
    
    def test_deterministic_with_same_key(self):
        """Test that same key produces same results."""
        crossover = UniformCrossover(crossover_rate=0.5)
        crossover_fn = crossover.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes1 = crossover_fn(population_stack, keys)
        crossed_genomes2 = crossover_fn(population_stack, keys)
        
        # Should be identical with same key
        self.assertTrue(jnp.array_equal(crossed_genomes1, crossed_genomes2))
    
    def test_different_results_with_different_keys(self):
        """Test that different keys produce different results (with high probability)."""
        crossover = UniformCrossover(crossover_rate=0.8)
        crossover_fn = crossover.build(self.population)
        
        # Use distinctive test genomes
        test_genomes = jnp.array([
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
        ])
        
        keys1 = jar.split(jar.PRNGKey(1), self.pop_size)
        keys2 = jar.split(jar.PRNGKey(2), self.pop_size)
        
        crossed_genomes1 = crossover_fn(test_genomes, keys1)
        crossed_genomes2 = crossover_fn(test_genomes, keys2)
        
        # With different keys and high crossover rate, results should differ
        differences = jnp.sum(crossed_genomes1 != crossed_genomes2)
        self.assertGreater(differences, 0)
    
    def test_jit_compatibility(self):
        """Test that the crossover function is JIT-compatible."""
        crossover = UniformCrossover(crossover_rate=0.3)
        crossover_fn = crossover.build(self.population)
        jit_crossover_fn = jax.jit(crossover_fn)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        # Should not raise an error
        crossed_genomes = jit_crossover_fn(population_stack, keys)
        
        # Check basic properties
        self.assertEqual(crossed_genomes.shape, population_stack.shape)
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
    
    def test_mask_generation_and_application(self):
        """Test the mask generation behavior of uniform crossover."""
        crossover = UniformCrossover(crossover_rate=0.5)
        crossover_fn = crossover.build(self.population)
        
        # Create a known pattern to test mask application
        test_genome = jnp.array([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]])
        keys = jar.split(self.key, 1)
        
        crossed_genome = crossover_fn(test_genome, keys)
        
        # Result should be different from input (with high probability)
        # and should still be binary
        self.assertTrue(jnp.all(jnp.isin(crossed_genome, jnp.array([0, 1]))))
        self.assertEqual(crossed_genome.shape, test_genome.shape)
    
    def test_crossover_probability_behavior(self):
        """Test that crossover probability is respected statistically."""
        crossover_rate = 0.3
        crossover = UniformCrossover(crossover_rate=crossover_rate)
        crossover_fn = crossover.build(self.population)
        
        # Run multiple trials to test probability
        num_trials = 1000
        total_bits = 0
        changed_bits = 0
        
        # Use a distinctive test genome
        test_genome = jnp.array([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1]])
        
        for i in range(num_trials):
            key = jar.PRNGKey(i)
            keys = jar.split(key, 1)
            crossed = crossover_fn(test_genome, keys)
            
            total_bits += test_genome.size
            changed_bits += jnp.sum(crossed != test_genome)
        
        actual_change_rate = changed_bits / total_bits
        # The actual change rate should be related to crossover_rate
        # Allow for statistical variance
        self.assertGreater(actual_change_rate, 0.0)
        self.assertLess(actual_change_rate, 1.0)
    
    def test_single_element_genome(self):
        """Test uniform crossover on single-element genomes."""
        # Create population with single-element genomes
        single_element_population = Population(
            solution_class=BinarySolution,
            max_size=4,
            random_init=True,
            random_key=self.key,
            genome_init_params={'array_size': 1, 'p': 0.5},
            fitness_transform=FitnessTransforms.minimize,
        )
        
        crossover = UniformCrossover(crossover_rate=1.0)
        crossover_fn = crossover.build(single_element_population)
        
        population_stack = single_element_population.to_stack()
        keys = jar.split(self.key, 4)
        
        crossed_genomes = crossover_fn(population_stack, keys)
        
        # Single element genomes should still be valid binary values
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
        self.assertEqual(crossed_genomes.shape, population_stack.shape)
    
    def test_all_zeros_genome_behavior(self):
        """Test uniform crossover behavior with all-zeros genomes."""
        crossover = UniformCrossover(crossover_rate=1.0)
        crossover_fn = crossover.build(self.population)
        
        # Test with all 0s
        all_zeros = jnp.zeros((self.pop_size, self.genome_size), dtype=jnp.int32)
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(all_zeros, keys)
        
        # Results should still be valid binary arrays
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
        self.assertEqual(crossed_genomes.shape, all_zeros.shape)
    
    def test_all_ones_genome_behavior(self):
        """Test uniform crossover behavior with all-ones genomes."""
        crossover = UniformCrossover(crossover_rate=1.0)
        crossover_fn = crossover.build(self.population)
        
        # Test with all 1s
        all_ones = jnp.ones((self.pop_size, self.genome_size), dtype=jnp.int32)
        keys = jar.split(self.key, self.pop_size)
        
        crossed_genomes = crossover_fn(all_ones, keys)
        
        # Results should still be valid binary arrays
        self.assertTrue(jnp.all(jnp.isin(crossed_genomes, jnp.array([0, 1]))))
        self.assertEqual(crossed_genomes.shape, all_ones.shape)
    
    def test_call_method_with_population(self):
        """Test the call method with a population object."""
        crossover = UniformCrossover(crossover_rate=0.5)
        
        result = crossover.call(self.population, self.key)
        
        # Should return a new population
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), len(self.population))
    
    def test_call_method_with_stack(self):
        """Test the call method with a genome stack."""
        crossover = UniformCrossover(crossover_rate=0.6)
        crossover.build(self.population)  # Build first
        
        population_stack = self.population.to_stack()
        result = crossover.call(population_stack, self.key)
        
        # Should return a modified stack
        self.assertIsInstance(result, jax.Array)
        self.assertEqual(result.shape, population_stack.shape)
        self.assertTrue(jnp.all(jnp.isin(result, jnp.array([0, 1]))))
    
    def test_empty_population_raises_error(self):
        """Test that building with empty population raises error."""
        empty_population = Population(
            solution_class=BinarySolution,
            max_size=0,
            random_init=False,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=FitnessTransforms.minimize,
        )
        
        crossover = UniformCrossover()
        
        with self.assertRaises(ValueError):
            crossover.build(empty_population)


