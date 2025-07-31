import unittest
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.core.genome import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.solution.base import FitnessTransforms
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator

class Test_TournamentSelection(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.key = jar.PRNGKey(42)
        self.genome_size = 10
        self.pop_size = 20
        self.genome_init_params = {'array_size': self.genome_size, 'p': 0.5}
        
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
        """Test TournamentSelection with default parameters."""
        selection = TournamentSelection()
        self.assertEqual(selection.number_of_tournaments, 10)
        self.assertEqual(selection.tournament_size, 4)
    
    def test_custom_initialization(self):
        """Test TournamentSelection with custom parameters."""
        num_tournaments = 15
        tournament_size = 6
        selection = TournamentSelection(
            number_of_tournaments=num_tournaments,
            tournament_size=tournament_size
        )
        self.assertEqual(selection.number_of_tournaments, num_tournaments)
        self.assertEqual(selection.tournament_size, tournament_size)
    
    def test_build_returns_callable(self):
        """Test that build method returns a callable function."""
        selection = TournamentSelection()
        selection_fn = selection.build(self.population)
        self.assertTrue(callable(selection_fn))
    
    def test_selection_output_shape(self):
        """Test that selection returns correct number of indices."""
        num_tournaments = 12
        selection = TournamentSelection(number_of_tournaments=num_tournaments)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Check that we get the expected number of selections
        self.assertEqual(selected_indices.shape, (num_tournaments,))
    
    def test_selection_indices_valid_range(self):
        """Test that selected indices are within valid population range."""
        selection = TournamentSelection(number_of_tournaments=25, tournament_size=3)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # All indices should be within [0, pop_size)
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < self.pop_size))
    
    def test_selection_indices_are_integers(self):
        """Test that selected indices are integers."""
        selection = TournamentSelection(number_of_tournaments=8)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Indices should be integers
        self.assertTrue(jnp.issubdtype(selected_indices.dtype, jnp.integer))
    
    def test_deterministic_with_same_key(self):
        """Test that same key produces same results."""
        selection = TournamentSelection(number_of_tournaments=10, tournament_size=5)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        
        selected_indices1 = selection_fn(fitness_values, self.key)
        selected_indices2 = selection_fn(fitness_values, self.key)
        
        # Should be identical with same key
        self.assertTrue(jnp.array_equal(selected_indices1, selected_indices2))
    
    def test_different_results_with_different_keys(self):
        """Test that different keys produce different results (with high probability)."""
        selection = TournamentSelection(number_of_tournaments=15, tournament_size=4)
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
        selection = TournamentSelection(number_of_tournaments=7, tournament_size=3)
        selection_fn = selection.build(self.population)
        jit_selection_fn = jax.jit(selection_fn)
        
        fitness_values = self.population.get_fitness_values()
        
        # Should not raise an error
        selected_indices = jit_selection_fn(fitness_values, self.key)
        
        # Check basic properties
        self.assertEqual(selected_indices.shape, (7,))
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < self.pop_size))
    
    def test_tournament_selection_bias_toward_high_fitness(self):
        """Test that tournament selection favors higher fitness individuals."""
        # Create a population with known fitness distribution
        biased_population = Population(
            solution_class=BinarySolution,
            max_size=10,
            random_init=False,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        # Create solutions with varying fitness levels
        # High fitness: more 1s, Low fitness: more 0s
        genomes = [
            # High fitness individuals (indices 0-2)
            jnp.ones(self.genome_size, dtype=jnp.int32),          # fitness = 10
            jnp.array([1]*9 + [0], dtype=jnp.int32),             # fitness = 9
            jnp.array([1]*8 + [0]*2, dtype=jnp.int32),           # fitness = 8
            # Medium fitness individuals (indices 3-5)
            jnp.array([1]*5 + [0]*5, dtype=jnp.int32),           # fitness = 5
            jnp.array([1]*4 + [0]*6, dtype=jnp.int32),           # fitness = 4
            jnp.array([1]*3 + [0]*7, dtype=jnp.int32),           # fitness = 3
            # Low fitness individuals (indices 6-9)
            jnp.array([1]*2 + [0]*8, dtype=jnp.int32),           # fitness = 2
            jnp.array([1]*1 + [0]*9, dtype=jnp.int32),           # fitness = 1
            jnp.zeros(self.genome_size, dtype=jnp.int32),        # fitness = 0
            jnp.zeros(self.genome_size, dtype=jnp.int32),        # fitness = 0
        ]
        
        # Add solutions to population
        for genome in genomes:
            solution = BinarySolution.from_tensor(genome, genome_init_params={'array_size': 10, 'p': 0.5})
            biased_population.add_solution(solution)
        
        # Evaluate fitness
        self.evaluator.evaluate_population(biased_population)
        
        selection = TournamentSelection(number_of_tournaments=100, tournament_size=4)
        selection_fn = selection.build(biased_population)
        
        # Run selection multiple times to get statistics
        total_selections = 0
        high_fitness_selections = 0  # Count selections of indices 0, 1, 2
        
        for i in range(100):  # Multiple runs
            key = jar.PRNGKey(i)
            fitness_values = biased_population.get_fitness_values()
            selected_indices = selection_fn(fitness_values, key)
            
            total_selections += len(selected_indices)
            # Count selections of high fitness individuals (indices 0, 1, 2)
            high_fitness_selections += jnp.sum(selected_indices < 3)
        
        # High fitness individuals should be selected more often
        high_fitness_rate = high_fitness_selections / total_selections
        self.assertGreater(high_fitness_rate, 0.4)  # Should be > 40% for high fitness
    
    def test_tournament_size_effect(self):
        """Test that larger tournament size increases selection pressure."""
        # Create population with extreme fitness values
        extreme_population = Population(
            solution_class=BinarySolution,
            max_size=6,
            random_init=False,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        genomes = [
            jnp.ones(self.genome_size, dtype=jnp.int32),    # Best individual
            jnp.zeros(self.genome_size, dtype=jnp.int32),   # Worst individuals
            jnp.zeros(self.genome_size, dtype=jnp.int32),
            jnp.zeros(self.genome_size, dtype=jnp.int32),
            jnp.zeros(self.genome_size, dtype=jnp.int32),
            jnp.zeros(self.genome_size, dtype=jnp.int32),
        ]
        
        for genome in genomes:
            solution = BinarySolution.from_tensor(genome, genome_init_params={'array_size': 10, 'p': 0.5})
            extreme_population.add_solution(solution)
        
        self.evaluator.evaluate_population(extreme_population)
        fitness_values = extreme_population.get_fitness_values()
        
        # Test with small tournament size
        small_tournament = TournamentSelection(number_of_tournaments=50, tournament_size=2)
        small_fn = small_tournament.build(extreme_population)
        small_selections = small_fn(fitness_values, self.key)
        small_best_rate = jnp.sum(small_selections == 0) / len(small_selections)
        
        # Test with large tournament size
        large_tournament = TournamentSelection(number_of_tournaments=50, tournament_size=5)
        large_fn = large_tournament.build(extreme_population)
        large_selections = large_fn(fitness_values, jar.PRNGKey(123))
        large_best_rate = jnp.sum(large_selections == 0) / len(large_selections)
        
        # Larger tournament size should select the best individual more often
        self.assertGreater(large_best_rate, small_best_rate)
    
    def test_uniform_fitness_approximately_uniform_selection(self):
        """Test that uniform fitness leads to approximately uniform selection."""
        # Create population with uniform fitness
        uniform_population = Population(
            solution_class=BinarySolution,
            max_size=5,
            random_init=False,
            random_key=self.key,
            genome_init_params={'array_size': 5, 'p': 0.5},
            fitness_transform=None,
        )
        
        # Create genomes with same fitness
        uniform_genome = jnp.array([1, 1, 0, 0, 0], dtype=jnp.int32)  # fitness = 2
        for _ in range(5):
            solution = BinarySolution.from_tensor(uniform_genome, genome_init_params={'array_size': 5, 'p': 0.5})
            uniform_population.add_solution(solution)
        
        self.evaluator.evaluate_population(uniform_population)
        
        selection = TournamentSelection(number_of_tournaments=500, tournament_size=3)
        selection_fn = selection.build(uniform_population)
        
        fitness_values = uniform_population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Count selections for each individual
        selection_counts = jnp.bincount(selected_indices, length=5)
        
        # With uniform fitness, selections should be roughly equal
        expected_count = 100  # 500/5
        for count in selection_counts:
            self.assertAlmostEqual(count, expected_count, delta=40)  # Allow some variance
    
    def test_single_tournament_behavior(self):
        """Test tournament selection with single tournament."""
        selection = TournamentSelection(number_of_tournaments=1, tournament_size=3)
        selection_fn = selection.build(self.population)
        
        fitness_values = self.population.get_fitness_values()
        selected_indices = selection_fn(fitness_values, self.key)
        
        # Should return exactly one index
        self.assertEqual(selected_indices.shape, (1,))
        self.assertTrue(0 <= selected_indices[0] < self.pop_size)

    
    def test_call_method_returns_population(self):
        """Test that call method returns a valid population."""
        selection = TournamentSelection(number_of_tournaments=8)
        
        result = selection.call(self.population, self.key)
        
        # Should return a population
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 8)  # Should match number_of_tournaments
        
        # All solutions in result should be valid
        for solution in result.get_solutions():
            self.assertIsNotNone(solution)
            self.assertIsNotNone(solution.genome)
    
    def test_call_method_with_kwargs(self):
        """Test that call method handles additional keyword arguments."""
        selection = TournamentSelection(number_of_tournaments=6)
        
        # Should work even with extra kwargs (though they might be ignored)
        result = selection.call(self.population, self.key, extra_param=42)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 6)
    
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
        
        selection = TournamentSelection()
        
        with self.assertRaises(ValueError):
            selection.build(empty_population)
    
    def test_tournament_size_larger_than_population(self):
        """Test behavior when tournament size is larger than population."""
        small_population = Population(
            solution_class=BinarySolution,
            max_size=3,
            random_init=True,
            random_key=self.key,
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        
        self.evaluator.evaluate_population(small_population)
        
        # Tournament size larger than population size
        selection = TournamentSelection(number_of_tournaments=5, tournament_size=10)
        selection_fn = selection.build(small_population)
        
        fitness_values = small_population.get_fitness_values()
        
        # Should still work (with replacement in tournaments)
        selected_indices = selection_fn(fitness_values, self.key)
        
        self.assertEqual(selected_indices.shape, (5,))
        self.assertTrue(jnp.all(selected_indices >= 0))
        self.assertTrue(jnp.all(selected_indices < 3))
    
    def test_dunder_call_method(self):
        """Test the __call__ method (convenience method)."""
        selection = TournamentSelection(number_of_tournaments=7)
        
        # Should work without explicitly building
        result = selection(self.population, self.key)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 7)
    
    def test_dunder_call_with_no_key(self):
        """Test __call__ method with no random key provided."""
        selection = TournamentSelection(number_of_tournaments=4)
        
        # Should work with default key
        result = selection(self.population)
        
        self.assertIsInstance(result, Population)
        self.assertEqual(len(result), 4)
    
    def test_multiple_builds_same_operator(self):
        """Test that the same operator can be built multiple times."""
        selection = TournamentSelection(number_of_tournaments=5)
        
        # Build with first population
        selection_fn1 = selection.build(self.population)
        
        # Create another population
        other_population = Population(
            solution_class=BinarySolution,
            max_size=15,
            random_init=True,
            random_key=jar.PRNGKey(999),
            genome_init_params=self.genome_init_params,
            fitness_transform=None,
        )
        self.evaluator.evaluate_population(other_population)
        
        # Build with second population should work
        selection_fn2 = selection.build(other_population)
        
        # Both functions should work
        fitness1 = self.population.get_fitness_values()
        fitness2 = other_population.get_fitness_values()
        
        result1 = selection_fn1(fitness1, self.key)
        result2 = selection_fn2(fitness2, self.key)
        
        self.assertEqual(result1.shape, (5,))
        self.assertEqual(result2.shape, (5,))