import unittest
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.selection.roulette import RouletteSelection
from malthusjax.core.genome import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.solution.base import FitnessTransforms
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
from malthusjax.core.base import SerializationContext


class Test_RouletteSelection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.key = jar.PRNGKey(42)
        self.genome_size = 10
        self.pop_size = 10
        self.genome_init_params = {'array_size': self.genome_size, 'p': 0.5}
                #import context
        
        self.context = SerializationContext(genome_class=BinaryGenome,
                                        solution_class=BinarySolution,
                                        genome_init_params=self.genome_init_params)
        # Create a test population
        self.population = Population(
            solution_class=BinarySolution,
            max_size=self.pop_size,
            random_init=True,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,  # Use raw fitness for clearer testing
        )
        
        # Evaluate fitness to ensure we have fitness values
        self.evaluator = BinarySumFitnessEvaluator()
        self.evaluator.evaluate_population(self.population)
    
    def test_default_initialization(self):
        """Test RouletteSelection with default parameters."""
        selection = RouletteSelection()
        self.assertEqual(selection.number_choices, 10)
    
    def test_custom_initialization(self):
        """Test RouletteSelection with custom number of choices."""
        custom_choices = 20
        selection = RouletteSelection(number_choices=custom_choices)
        self.assertEqual(selection.number_choices, custom_choices)
    
    def test_build_returns_callable(self):
        """Test that build method returns a callable function."""
        selection = RouletteSelection()
        selection_fn = selection.build(self.population)
        self.assertTrue(callable(selection_fn))
    
    def test_selection_output_shape(self):
        """Test that selection returns correct number of indices."""
        number_choices = 15
        selection = RouletteSelection(number_choices=number_choices)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Check that we get the expected number of selections
        self.assertEqual(selected_indices.shape, (number_choices,))
    
    def test_selection_indices_valid_range(self):
        """Test that selected indices are within valid population range."""
        selection = RouletteSelection(number_choices=20)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # All indices should be within [0, pop_size)
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < self.pop_size))
    
    def test_selection_indices_are_integers(self):
        """Test that selected indices are integers."""
        selection = RouletteSelection(number_choices=10)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Indices should be integers
        self.assertTrue(jnp.issubdtype(selected_indices.dtype, jnp.integer))
    
    def test_deterministic_with_same_key(self):
        """Test that same key produces same results."""
        selection = RouletteSelection(number_choices=8)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        
        selected_indices1 = selection_fn(fitness_values, self.key)
        selected_indices2 = selection_fn(fitness_values, self.key)
        
        # Should be identical with same key
        self.assertTrue(jnp.array_equal(selected_indices1, selected_indices2))
    
    def test_different_results_with_different_keys(self):
        """Test that different keys produce different results (with high probability)."""
        selection = RouletteSelection(number_choices=15)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        
        key1 = jar.PRNGKey(1)
        key2 = jar.PRNGKey(2)
        
        selected_indices1 = selection_fn(fitness_values, key1)
        selected_indices2 = selection_fn(fitness_values, key2)
        
        # With different keys, results should likely differ
        differences = jnp.sum(selected_indices1 != selected_indices2)
        self.assertGreater(differences, 0)
    
    def test_jit_compatibility(self):
        """Test that the selection function is JIT-compatible."""
        selection = RouletteSelection(number_choices=5)
        selection_fn = selection.build(self.population)
        jit_selection_fn = jax.jit(selection_fn)
        
        fitness_values = self.population.get_fitness_values()
        
        # Should not raise an error
        selected_indices = jit_selection_fn(fitness_values, self.key)
        
        # Check basic properties
        self.assertEqual(selected_indices.shape, (5,))
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < self.pop_size))
    
    def test_fitness_proportionate_selection_bias(self):
        """Test that higher fitness individuals are selected more often."""
        # Create a population with known fitness distribution
        high_fitness_population = Population(
            solution_class=BinarySolution,
            max_size=5,
            random_init=False,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        # Manually create solutions with different fitness values
        # Higher fitness = more 1s in binary genome for BinarySumFitnessEvaluator
        genomes = [
            jnp.ones(self.genome_size, dtype=jnp.int32),  # High fitness (all 1s)
            jnp.ones(self.genome_size, dtype=jnp.int32),  # High fitness (all 1s)
            jnp.ones(self.genome_size, dtype=jnp.int32),  # High fitness (all 1s)
            jnp.zeros(self.genome_size, dtype=jnp.int32), # Low fitness (all 0s)
            jnp.zeros(self.genome_size, dtype=jnp.int32), # Low fitness (all 0s)
        ]
        
        # Add solutions to population
        for genome in genomes:
            solution = BinarySolution.from_tensor(genome, genome_init_params=self.genome_init_params)
            high_fitness_population.add_solution(solution)

        # Evaluate fitness
        self.evaluator.evaluate_population(high_fitness_population)
        
        selection = RouletteSelection(number_choices=100)
        selection_fn = selection.build(high_fitness_population)
        
        # Run selection multiple times to get statistics
        total_selections = 0
        high_fitness_selections = 0
        
        for i in range(10):  # Multiple runs
            key = jar.PRNGKey(i)
            fitness_values = high_fitness_population.get_fitness_values()
            selected_indices = selection_fn(fitness_values, key)
            
            total_selections += len(selected_indices)
            # Count selections of high fitness individuals (indices 0, 1, 2)
            high_fitness_selections += jnp.sum(selected_indices < 3)
        
        # High fitness individuals should be selected more often than low fitness
        high_fitness_rate = high_fitness_selections / total_selections
        self.assertGreater(high_fitness_rate, 0.6)  # Should be > 60% for high fitness
    
    def test_uniform_fitness_uniform_selection(self):
        """Test that uniform fitness leads to approximately uniform selection."""
        # Create population with uniform fitness
        uniform_population = Population(
            solution_class=BinarySolution,
            max_size=4,
            random_init=False,
            random_key=self.key,
            genome_init_params={'array_size': 5, 'p': 0.5},
            fitness_transform=None,
        )
        
        # Create genomes with same fitness (same number of 1s)
        uniform_genome = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)
        for _ in range(4):
            solution = BinarySolution.from_tensor(uniform_genome, genome_init_params={'array_size': 5, 'p': 0.5})
            uniform_population.add_solution(solution)
        
        self.evaluator.evaluate_population(uniform_population)
        
        selection = RouletteSelection(number_choices=400)  # Large sample
        selection_fn = selection.build(uniform_population)
        
        fitness_values = uniform_population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Count selections for each individual
        selection_counts = jnp.bincount(selected_indices, length=4)
        
        # With uniform fitness, selections should be roughly equal
        expected_count = 100  # 400/4
        for count in selection_counts:
            self.assertAlmostEqual(count, expected_count, delta=30)  # Allow some variance
    
    def test_zero_fitness_handling(self):
        """Test behavior with zero fitness values."""
        # Create population with some zero fitness
        zero_fitness_population = Population(
            solution_class=BinarySolution,
            max_size=3,
            random_init=False,
            random_key=self.key,
            genome_init_params={'array_size': 5, 'p': 0.5},
            fitness_transform=None,
        )
        
        # Create genomes: one with fitness, two with zero fitness
        genomes = [
            jnp.array([1, 1, 1, 1, 1], dtype=jnp.int32),  # High fitness
            jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32),  # Zero fitness
            jnp.array([0, 0, 0, 0, 0], dtype=jnp.int32),  # Zero fitness
        ]
        
        for genome in genomes:
            solution = BinarySolution.from_tensor(genome, genome_init_params={'array_size': 5, 'p': 0.5})
            zero_fitness_population.add_solution(solution)
        
        self.evaluator.evaluate_population(zero_fitness_population)
        
        selection = RouletteSelection(number_choices=50)
        selection_fn = selection.build(zero_fitness_population)
        
        fitness_values = zero_fitness_population.get_fitness_values()
        
        # This should work without division by zero issues
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Should still return valid indices
        self.assertEqual(selected_indices.shape, (50,))
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < 3))
    
    def test_call_method_returns_population(self):
        """Test that call method returns a valid population."""
        selection = RouletteSelection(number_choices=8)

        result = selection.call(self.population, jar.PRNGKey(123))
        print(result)
        # Should return a population
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 8)  # Should match number_choices
        
        # All solutions in result should be valid
        for solution in result.get_solutions():
            self.assertIsNotNone(solution)
            self.assertIsNotNone(solution.genome)
    
    def test_call_method_with_kwargs(self):
        """Test that call method handles additional keyword arguments."""
        selection = RouletteSelection(number_choices=5)
        
        # Should work even with extra kwargs (though they might be ignored)
        result = selection.call(self.population, self.key, extra_param=42)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 5)
    
    def test_empty_population_raises_error(self):
        """Test that building with empty population raises error."""
        empty_population = Population(
            solution_class=BinarySolution,
            max_size=0,
            random_init=False,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        selection = RouletteSelection()
        
        with self.assertRaises(ValueError):
            selection.build(empty_population)
    
    def test_single_individual_population(self):
        """Test selection with population of size 1."""
        single_population = Population(
            solution_class=BinarySolution,
            max_size=1,
            random_init=True,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        self.evaluator.evaluate_population(single_population)
        
        selection = RouletteSelection(number_choices=5)
        selection_fn = selection.build(single_population)
        
        fitness_values = single_population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # All selections should be the same individual (index 0)
        expected_indices = jnp.zeros(5, dtype=jnp.int32)
        self.assertTrue(jnp.array_equal(selected_indices, expected_indices))
    
    def test_dunder_call_method(self):
        """Test the __call__ method (convenience method)."""
        selection = RouletteSelection(number_choices=6)
        
        # Should work without explicitly building
        result = selection(self.population, self.key)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 6)
    
    def test_dunder_call_with_no_key(self):
        """Test __call__ method with no random key provided."""
        selection = RouletteSelection(number_choices=4)
        
        # Should work with default key
        result = selection(self.population)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 4)