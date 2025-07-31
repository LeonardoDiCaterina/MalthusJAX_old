import unittest
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.core.genome import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.solution.base import FitnessTransforms


class Test_BitFlipMutation(unittest.TestCase):
    
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
        """Test BitFlipMutation with default parameters."""
        mutation = BitFlipMutation()
        self.assertEqual(mutation.mutation_rate, 0.01)
    
    def test_custom_initialization(self):
        """Test BitFlipMutation with custom mutation rate."""
        custom_rate = 0.05
        mutation = BitFlipMutation(mutation_rate=custom_rate)
        self.assertEqual(mutation.mutation_rate, custom_rate)
    
    def test_invalid_mutation_rate(self):
        """Test that invalid mutation rates raise appropriate errors."""
        with self.assertRaises((ValueError, AssertionError)):
            BitFlipMutation(mutation_rate=-0.1)
        
        with self.assertRaises((ValueError, AssertionError)):
            BitFlipMutation(mutation_rate=1.1)
    
    def test_build_returns_callable(self):
        """Test that build method returns a callable function."""
        mutation = BitFlipMutation()
        mutation_fn = mutation.build(self.population)
        self.assertTrue(callable(mutation_fn))
    
    def test_mutation_output_shape(self):
        """Test that mutation preserves genome shape."""
        mutation = BitFlipMutation(mutation_rate=0.5)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Check shape is preserved
        self.assertEqual(mutated_genomes.shape, population_stack.shape)
        self.assertEqual(mutated_genomes.shape, (self.pop_size, self.genome_size))
    
    def test_mutation_output_dtype(self):
        """Test that mutation preserves binary dtype."""
        mutation = BitFlipMutation(mutation_rate=0.5)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Check that output contains only 0s and 1s
        self.assertTrue(jnp.all(jnp.isin(mutated_genomes, jnp.array([0, 1]))))
    
    def test_zero_mutation_rate(self):
        """Test that zero mutation rate produces no changes."""
        mutation = BitFlipMutation(mutation_rate=0.0)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Should be identical to original
        self.assertTrue(jnp.array_equal(mutated_genomes, population_stack))
    
    def test_high_mutation_rate_causes_changes(self):
        """Test that high mutation rate causes significant changes."""
        mutation = BitFlipMutation(mutation_rate=0.8)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # With high mutation rate, there should be differences
        # (though not guaranteed for every individual)
        differences = jnp.sum(mutated_genomes != population_stack)
        self.assertGreater(differences, 0)
    
    def test_deterministic_with_same_key(self):
        """Test that same key produces same results."""
        mutation = BitFlipMutation(mutation_rate=0.3)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes1 = mutation_fn(population_stack, keys)
        mutated_genomes2 = mutation_fn(population_stack, keys)
        
        # Should be identical with same key
        self.assertTrue(jnp.array_equal(mutated_genomes1, mutated_genomes2))
    
    def test_different_results_with_different_keys(self):
        """Test that different keys produce different results (with high probability)."""
        mutation = BitFlipMutation(mutation_rate=0.5)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys1 = jar.split(jar.PRNGKey(1), self.pop_size)
        keys2 = jar.split(jar.PRNGKey(2), self.pop_size)
        
        mutated_genomes1 = mutation_fn(population_stack, keys1)
        mutated_genomes2 = mutation_fn(population_stack, keys2)
        
        # With high mutation rate, results should likely differ
        # (though not guaranteed due to randomness)
        differences = jnp.sum(mutated_genomes1 != mutated_genomes2)
        # With 50% mutation rate on 50 bits, we expect some differences
        self.assertGreater(differences, 0)
    
    def test_jit_compatibility(self):
        """Test that the mutation function is JIT-compatible."""
        mutation = BitFlipMutation(mutation_rate=0.2)
        mutation_fn = mutation.build(self.population)
        jit_mutation_fn = jax.jit(mutation_fn)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        # Should not raise an error
        mutated_genomes = jit_mutation_fn(population_stack, keys)
        
        # Check basic properties
        self.assertEqual(mutated_genomes.shape, population_stack.shape)
        self.assertTrue(jnp.all(jnp.isin(mutated_genomes, jnp.array([0, 1]))))
    
    def test_mutation_statistics(self):
        """Test that mutation rate approximately matches expected flip rate."""
        mutation_rate = 0.1
        mutation = BitFlipMutation(mutation_rate=mutation_rate)
        mutation_fn = mutation.build(self.population)
        
        population_stack = self.population.to_stack()
        keys = jar.split(self.key, self.pop_size)
        
        mutated_genomes = mutation_fn(population_stack, keys)
        
        # Calculate actual flip rate
        total_bits = population_stack.size
        flipped_bits = jnp.sum(mutated_genomes != population_stack)
        actual_flip_rate = flipped_bits / total_bits
        
        # Should be approximately equal to mutation rate (within reasonable tolerance)
        # Using a relatively loose tolerance due to randomness
        self.assertAlmostEqual(actual_flip_rate, mutation_rate, delta=0.05)
