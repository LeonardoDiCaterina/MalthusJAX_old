import unittest
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.mutation.binary import ScrambleMutation
from malthusjax.core.genome import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.solution.base import FitnessTransforms


class Test_ScrambleMutation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.key = jar.PRNGKey(42)
        self.genome_size = 10
        self.pop_size = 5
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
        """Test ScrambleMutation with default parameters."""
        mutation = ScrambleMutation()
        self.assertEqual(mutation.mutation_rate, 0.01)
    
    def test_custom_initialization(self):
        """Test ScrambleMutation with custom mutation rate."""
        custom_rate = 0.05
        mutation = ScrambleMutation(mutation_rate=custom_rate)
        self.assertEqual(mutation.mutation_rate, custom_rate)
    
    def test_invalid_mutation_rate(self):
        """Test that invalid mutation rates raise appropriate errors."""
        with self.assertRaises((ValueError, AssertionError)):
            ScrambleMutation(mutation_rate=-0.1)
        
        with self.assertRaises((ValueError, AssertionError)):
            ScrambleMutation(mutation_rate=1.1)
    
    def test_build_returns_callable(self):
        """Test that build method returns a callable function."""
        mutation = ScrambleMutation()
        mutation_fn = mutation.build(self.population)
        self.assertTrue(callable(mutation_fn))
    
    def test_mutation_output_shape(self):
        """Test that mutation preserves genome shape."""
        mutation = ScrambleMutation(mutation_rate=0.5)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Check shape is preserved
        self.assertEqual(mutated_genomes.shape, population_stack.shape)
        self.assertEqual(mutated_genomes.shape, (self.pop_size, self.genome_size))
    
    def test_mutation_output_dtype_and_values(self):
        """Test that mutation preserves binary dtype and values."""
        mutation = ScrambleMutation(mutation_rate=1.0)  # High rate to ensure scrambling
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Check that output contains only 0s and 1s
        self.assertTrue(jnp.all(jnp.isin(mutated_genomes, jnp.array([0, 1]))))
        # Check dtype is preserved
        self.assertEqual(mutated_genomes.dtype, population_stack.dtype)
    
    def test_zero_mutation_rate(self):
        """Test that zero mutation rate produces no changes."""
        mutation = ScrambleMutation(mutation_rate=0.0)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Should be identical to original
        self.assertTrue(jnp.array_equal(mutated_genomes, population_stack))
    
    def test_scramble_preserves_element_counts(self):
        """Test that scrambling preserves the count of each element (permutation property)."""
        mutation = ScrambleMutation(mutation_rate=1.0)  # Always scramble
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # For each genome, check that element counts are preserved
        for i in range(self.pop_size):
            original_counts = jnp.bincount(population_stack[i], length=2)
            mutated_counts = jnp.bincount(mutated_genomes[i], length=2)
            self.assertTrue(jnp.array_equal(original_counts, mutated_counts))
    
    def test_high_mutation_rate_causes_changes(self):
        """Test that high mutation rate causes changes (with high probability)."""
        mutation = ScrambleMutation(mutation_rate=0.8)
        mutation_fn = mutation.build(self.population)
        
        # Use a population with non-uniform distribution to make changes more visible
        test_genome = jnp.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        test_population_stack = jnp.tile(test_genome, (self.pop_size, 1))
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(test_population_stack, keys)
        
        # With high mutation rate, there should be differences
        differences = jnp.sum(mutated_genomes != test_population_stack)
        self.assertGreater(differences, 0)
    
    def test_deterministic_with_same_key(self):
        """Test that same key produces same results."""
        mutation = ScrambleMutation(mutation_rate=0.5)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes1 = mutation_fn(population_stack, keys)
        mutated_genomes2 = mutation_fn(population_stack, keys)
        
        # Should be identical with same key
        self.assertTrue(jnp.array_equal(mutated_genomes1, mutated_genomes2))
    
    def test_different_results_with_different_keys(self):
        """Test that different keys produce different results (with high probability)."""
        mutation = ScrambleMutation(mutation_rate=1.0)  # Always scramble
        mutation_fn = mutation.build(self.population)
        
        # Use a non-uniform genome to make differences more likely
        test_genome = jnp.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
        test_population_stack = jnp.tile(test_genome, (self.pop_size, 1))
        
        keys1 = jar.split(jar.PRNGKey(1), self.pop_size)
        keys2 = jar.split(jar.PRNGKey(2), self.pop_size)
        
        mutated_genomes1 = mutation_fn(test_population_stack, keys1)
        mutated_genomes2 = mutation_fn(test_population_stack, keys2)
        
        # With different keys and always scrambling, results should differ
        differences = jnp.sum(mutated_genomes1 != mutated_genomes2)
        self.assertGreater(differences, 0)
    
    def test_jit_compatibility(self):
        """Test that the mutation function is JIT-compatible."""
        mutation = ScrambleMutation(mutation_rate=0.3)
        mutation_fn = mutation.build(self.population)
        jit_mutation_fn = jax.jit(mutation_fn)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        # Should not raise an error
        mutated_genomes = jit_mutation_fn(population_stack, keys)
        
        # Check basic properties
        self.assertEqual(mutated_genomes.shape, population_stack.shape)
        self.assertTrue(jnp.all(jnp.isin(mutated_genomes, jnp.array([0, 1]))))
    
    def test_scramble_behavior_with_uniform_genome(self):
        """Test scramble behavior with uniform genomes (all 0s or all 1s)."""
        mutation = ScrambleMutation(mutation_rate=1.0)
        mutation_fn = mutation.build(self.population)
        
        # Test with all 0s
        all_zeros = jnp.zeros((self.pop_size, self.genome_size), dtype=jnp.int32)
        keys = jar.split(self.key, self.pop_size)
        mutated_zeros = mutation_fn(all_zeros, keys)
        
        # Scrambling all zeros should still be all zeros
        self.assertTrue(jnp.array_equal(mutated_zeros, all_zeros))
        
        # Test with all 1s
        all_ones = jnp.ones((self.pop_size, self.genome_size), dtype=jnp.int32)
        mutated_ones = mutation_fn(all_ones, keys)
        
        # Scrambling all ones should still be all ones
        self.assertTrue(jnp.array_equal(mutated_ones, all_ones))
    
    def test_mutation_probability_behavior(self):
        """Test that mutation probability is respected statistically."""
        mutation_rate = 0.3
        mutation = ScrambleMutation(mutation_rate=mutation_rate)
        mutation_fn = mutation.build(self.population)
        
        # Run multiple trials to test probability
        num_trials = 1000
        mutations_occurred = 0
        
        # Use a distinctive genome pattern
        test_genome = jnp.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        test_population_stack = jnp.tile(test_genome, (1, 1))  # Single genome
        
        for i in range(num_trials):
            key = jar.PRNGKey(i)
            keys = jar.split(key, 1)
            mutated = mutation_fn(test_population_stack, keys)
            
            if not jnp.array_equal(mutated, test_population_stack):
                mutations_occurred += 1
        
        actual_rate = mutations_occurred / num_trials
        # Allow for some statistical variance
        self.assertAlmostEqual(actual_rate, mutation_rate, delta=0.05)
    
    def test_single_element_genome(self):
        """Test scramble mutation on single-element genomes."""
        # Create population with single-element genomes
        single_element_population = Population(
            solution_class=BinarySolution,
            max_size=3,
            random_init=True,
            random_key=self.key,
            genome_init_params={'array_size': 1, 'p': 0.5},
            fitness_transform=FitnessTransforms.minimize,
        )
        
        mutation = ScrambleMutation(mutation_rate=1.0)
        mutation_fn = mutation.build(single_element_population)
        
        population_stack = single_element_population.to_stack()
        keys = jar.split(self.key, 3)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Single element should remain unchanged even when scrambled
        self.assertTrue(jnp.array_equal(mutated_genomes, population_stack))
