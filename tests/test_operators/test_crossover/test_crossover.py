import pytest
import jax.numpy as jnp
import jax.random as jar
import jax

from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.selection.roulette import RouletteSelection
from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.genome.real import RealGenome
from malthusjax.core.genome.permutation import PermutationGenome
from malthusjax.core.genome.categorical import CategoricalGenome
from malthusjax.core.population.population import Population
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
from malthusjax.core.fitness.real import SphereFitnessEvaluator


class TestTournamentSelection:
    """Test suite for TournamentSelection operator."""
    
    def test_tournament_selection_initialization(self):
        """Test TournamentSelection initialization."""
        selector = TournamentSelection(number_of_tournaments=5, tournament_size=3)
        assert selector.number_of_tournaments == 5
        assert selector.tournament_size == 3
        assert selector._compiled_fn is not None
    
    def test_tournament_selection_default_params(self):
        """Test TournamentSelection with default parameters."""
        selector = TournamentSelection()
        assert selector.number_of_tournaments == 10
        assert selector.tournament_size == 4
    
    def test_tournament_selection_function_creation(self):
        """Test that tournament selection function is created properly."""
        selector = TournamentSelection(number_of_tournaments=3, tournament_size=2)
        selection_fn = selector._create_selection_function()
        
        assert callable(selection_fn)
        
        # Test with sample fitness scores
        fitness_scores = jnp.array([0.1, 0.8, 0.3, 0.9, 0.5])
        key = jar.PRNGKey(42)
        
        selected_indices = selection_fn(fitness_scores, key)
        assert selected_indices.shape == (3,)  # number_of_tournaments
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_scores))
    
    def test_tournament_selection_jit_compilation(self):
        """Test that tournament selection function compiles with JIT."""
        selector = TournamentSelection(number_of_tournaments=5, tournament_size=3)
        selection_fn = selector.get_compiled_function()
        
        # Should be JIT compiled
        fitness_scores = jnp.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.4, 0.7])
        key = jar.PRNGKey(123)
        
        selected_indices = selection_fn(fitness_scores, key)
        assert selected_indices.shape == (5,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_scores))
    
    def test_tournament_selection_bias_toward_high_fitness(self):
        """Test that tournament selection favors higher fitness individuals."""
        selector = TournamentSelection(number_of_tournaments=100, tournament_size=4)
        selection_fn = selector.get_compiled_function()
        
        # Create fitness scores with clear best individual
        fitness_scores = jnp.array([0.1, 0.1, 0.1, 0.1, 0.9, 0.1, 0.1])  # Index 4 has highest fitness
        key = jar.PRNGKey(42)
        
        selected_indices = selection_fn(fitness_scores, key)
        
        # Count how often the best individual (index 4) is selected
        best_selections = jnp.sum(selected_indices == 4)
        
        # Should select the best individual more often than random chance
        # With many tournaments, high fitness individual should be selected frequently
        assert best_selections > 10  # Should be much higher than random selection
    
    def test_tournament_selection_reproducibility(self):
        """Test that tournament selection is reproducible with same random key."""
        selector = TournamentSelection(number_of_tournaments=5, tournament_size=3)
        selection_fn = selector.get_compiled_function()
        
        fitness_scores = jnp.array([0.2, 0.8, 0.1, 0.9, 0.3])
        key = jar.PRNGKey(42)
        
        selected1 = selection_fn(fitness_scores, key)
        selected2 = selection_fn(fitness_scores, key)
        
        assert jnp.array_equal(selected1, selected2)
    
    def test_tournament_selection_with_population(self):
        """Test tournament selection with actual population."""
        # Create a population
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=10,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 8, 'p': 0.5}
        )
        
        # Create fitness values
        fitness_evaluator = BinarySumFitnessEvaluator()
        fitness_values = fitness_evaluator.evaluate_batch(population, return_tensors=True)
        
        # Apply tournament selection
        selector = TournamentSelection(number_of_tournaments=5, tournament_size=3)
        selected_population = selector.call(
            population=population,
            fitness_values=fitness_values,
            random_key=jar.PRNGKey(123)
        )
        
        assert len(selected_population) == 5  # number_of_tournaments
        assert selected_population._pop_size == population._pop_size
    
    def test_tournament_selection_edge_cases(self):
        """Test tournament selection edge cases."""
        # Single individual tournaments
        selector = TournamentSelection(number_of_tournaments=3, tournament_size=1)
        selection_fn = selector.get_compiled_function()
        
        fitness_scores = jnp.array([0.1, 0.5, 0.3])
        selected = selection_fn(fitness_scores, jar.PRNGKey(42))
        assert selected.shape == (3,)
        
        # Large tournament size
        selector_large = TournamentSelection(number_of_tournaments=2, tournament_size=5)
        selection_fn_large = selector_large.get_compiled_function()
        
        fitness_scores = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        selected = selection_fn_large(fitness_scores, jar.PRNGKey(42))
        assert selected.shape == (2,)


class TestRouletteSelection:
    """Test suite for RouletteSelection operator."""
    
    def test_roulette_selection_initialization(self):
        """Test RouletteSelection initialization."""
        selector = RouletteSelection(number_choices=5)
        assert selector.number_choices == 5
        assert selector._compiled_fn is not None
    
    def test_roulette_selection_function_creation(self):
        """Test that roulette selection function is created properly."""
        selector = RouletteSelection(number_choices=3)
        selection_fn = selector._create_selection_function()
        
        assert callable(selection_fn)
        
        # Test with sample fitness scores
        fitness_scores = jnp.array([0.1, 0.3, 0.6, 0.2])
        key = jar.PRNGKey(42)
        
        selected_indices = selection_fn(fitness_scores, key)
        assert selected_indices.shape == (3,)  # number_choices
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_scores))
    
    def test_roulette_selection_jit_compilation(self):
        """Test that roulette selection function compiles with JIT."""
        selector = RouletteSelection(number_choices=4)
        selection_fn = selector.get_compiled_function()
        
        # Should be JIT compiled
        fitness_scores = jnp.array([0.1, 0.2, 0.4, 0.3, 0.5])
        key = jar.PRNGKey(123)
        
        selected_indices = selection_fn(fitness_scores, key)
        assert selected_indices.shape == (4,)
        assert jnp.all(selected_indices >= 0)
        assert jnp.all(selected_indices < len(fitness_scores))
    
    def test_roulette_selection_probability_bias(self):
        """Test that roulette selection respects fitness probabilities."""
        selector = RouletteSelection(number_choices=1000)
        selection_fn = selector.get_compiled_function()
        
        # Create fitness scores where one individual has much higher fitness
        fitness_scores = jnp.array([0.1, 0.1, 0.8, 0.1])  # Index 2 has 80% of total fitness
        key = jar.PRNGKey(42)
        
        selected_indices = selection_fn(fitness_scores, key)
        
        # Count selections for each individual
        unique_indices, counts = jnp.unique(selected_indices, return_counts=True)
        
        # Individual with highest fitness should be selected most often
        index_2_count = jnp.sum(jnp.where(unique_indices == 2, counts, 0))
        
        # Should be significantly more than random (25% would be 250)
        assert index_2_count > 600  # Should be around 80% of 1000
    
    def test_roulette_selection_reproducibility(self):
        """Test that roulette selection is reproducible with same random key."""
        selector = RouletteSelection(number_choices=5)
        selection_fn = selector.get_compiled_function()
        
        fitness_scores = jnp.array([0.2, 0.3, 0.1, 0.4])
        key = jar.PRNGKey(42)
        
        selected1 = selection_fn(fitness_scores, key)
        selected2 = selection_fn(fitness_scores, key)
        
        assert jnp.array_equal(selected1, selected2)
    
    def test_roulette_selection_with_population(self):
        """Test roulette selection with actual population."""
        # Create a population
        population = Population(
            genome_cls=RealGenome,
            pop_size=8,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'minval': 0.0, 'maxval': 1.0}
        )
        
        # Create fitness values (need positive values for roulette)
        fitness_evaluator = SphereFitnessEvaluator()
        raw_fitness = fitness_evaluator.evaluate_batch(population, return_tensors=True)
        # Convert to positive fitness (lower sphere values = higher fitness)
        fitness_values = 1.0 / (1.0 + raw_fitness)
        
        # Apply roulette selection
        selector = RouletteSelection(number_choices=6)
        selected_population = selector.call(
            population=population,
            fitness_values=fitness_values,
            random_key=jar.PRNGKey(123)
        )
        
        assert len(selected_population) == 6  # number_choices
        assert selected_population._pop_size == population._pop_size
    
    def test_roulette_selection_uniform_fitness(self):
        """Test roulette selection with uniform fitness scores."""
        selector = RouletteSelection(number_choices=100)
        selection_fn = selector.get_compiled_function()
        
        # All individuals have equal fitness
        fitness_scores = jnp.array([0.25, 0.25, 0.25, 0.25])
        key = jar.PRNGKey(42)
        
        selected_indices = selection_fn(fitness_scores, key)
        
        # Count selections for each individual
        unique_indices, counts = jnp.unique(selected_indices, return_counts=True)
        
        # With uniform fitness, selections should be roughly equal
        # Allow for some randomness - each should get roughly 25 selections
        for count in counts:
            assert 15 <= count <= 35  # Allow reasonable variance around 25
    
    def test_roulette_selection_edge_cases(self):
        """Test roulette selection edge cases."""
        # Single choice
        selector = RouletteSelection(number_choices=1)
        selection_fn = selector.get_compiled_function()
        
        fitness_scores = jnp.array([0.3, 0.7])
        selected = selection_fn(fitness_scores, jar.PRNGKey(42))
        assert selected.shape == (1,)
        
        # Many choices from small population
        selector_many = RouletteSelection(number_choices=10)
        selection_fn_many = selector_many.get_compiled_function()
        
        fitness_scores = jnp.array([0.4, 0.6])
        selected = selection_fn_many(fitness_scores, jar.PRNGKey(42))
        assert selected.shape == (10,)
        assert jnp.all(selected >= 0)
        assert jnp.all(selected < 2)


class TestSelectionOperatorBase:
    """Test suite for base selection operator functionality."""
    
    def test_abstract_selection_operator_interface(self):
        """Test that selection operators implement required interface."""
        tournament_selector = TournamentSelection(number_of_tournaments=3, tournament_size=2)
        roulette_selector = RouletteSelection(number_choices=5)
        
        # Both should have compiled functions
        assert tournament_selector.get_compiled_function() is not None
        assert roulette_selector.get_compiled_function() is not None
        
        # Both should be callable
        assert callable(tournament_selector.get_compiled_function())
        assert callable(roulette_selector.get_compiled_function())
    
    def test_selection_with_different_genome_types(self):
        """Test selection operators with different genome types."""
        genome_types = [
            (BinaryGenome, {'array_size': 6, 'p': 0.5}),
            (RealGenome, {'array_size': 4, 'minval': -1.0, 'maxval': 1.0}),
            (PermutationGenome, {'permutation_start': 0, 'permutation_end': 5}),
            (CategoricalGenome, {'array_size': 5, 'num_categories': 3})
        ]
        
        for genome_cls, init_params in genome_types:
            # Create population
            population = Population(
                genome_cls=genome_cls,
                pop_size=6,
                random_init=True,
                random_key=jar.PRNGKey(42),
                genome_init_params=init_params
            )
            
            # Create dummy fitness values
            fitness_values = jnp.array([0.1, 0.3, 0.8, 0.2, 0.6, 0.4])
            
            # Test tournament selection
            tournament_selector = TournamentSelection(number_of_tournaments=3, tournament_size=2)
            selected_tournament = tournament_selector.call(
                population=population,
                fitness_values=fitness_values,
                random_key=jar.PRNGKey(123)
            )
            assert len(selected_tournament) == 3
            
            # Test roulette selection
            roulette_selector = RouletteSelection(number_choices=4)
            selected_roulette = roulette_selector.call(
                population=population,
                fitness_values=fitness_values,
                random_key=jar.PRNGKey(456)
            )
            assert len(selected_roulette) == 4
    
    def test_selection_error_handling(self):
        """Test error handling in selection operators."""
        # Test with empty fitness array
        selector = TournamentSelection(number_of_tournaments=2, tournament_size=2)
        selection_fn = selector.get_compiled_function()
        
        empty_fitness = jnp.array([])
        
        # This should raise an error or handle gracefully
        with pytest.raises((IndexError, ValueError)):
            selection_fn(empty_fitness, jar.PRNGKey(42))
    
    def test_selection_performance_characteristics(self):
        """Test performance characteristics of selection operators."""
        # Create large population for performance testing
        large_fitness = jnp.array([jnp.float32(i * 0.1) for i in range(1000)])
        key = jar.PRNGKey(42)
        
        # Tournament selection should be fast
        tournament_selector = TournamentSelection(number_of_tournaments=100, tournament_size=5)
        tournament_fn = tournament_selector.get_compiled_function()
        
        selected_tournament = tournament_fn(large_fitness, key)
        assert selected_tournament.shape == (100,)
        
        # Roulette selection should also be fast
        roulette_selector = RouletteSelection(number_choices=100)
        roulette_fn = roulette_selector.get_compiled_function()
        
        selected_roulette = roulette_fn(large_fitness, key)
        assert selected_roulette.shape == (100,)


class TestSelectionComparison:
    """Test suite comparing different selection methods."""
    
    def test_selection_diversity(self):
        """Test that different selection methods produce different diversity."""
        fitness_scores = jnp.array([0.1, 0.2, 0.9, 0.3, 0.4])
        key = jar.PRNGKey(42)
        
        # Tournament selection
        tournament_selector = TournamentSelection(number_of_tournaments=20, tournament_size=3)
        tournament_fn = tournament_selector.get_compiled_function()
        tournament_selected = tournament_fn(fitness_scores, key)
        
        # Roulette selection
        roulette_selector = RouletteSelection(number_choices=20)
        roulette_fn = roulette_selector.get_compiled_function()
        roulette_selected = roulette_fn(fitness_scores, key)
        
        # Count unique selections
        tournament_unique = len(jnp.unique(tournament_selected))
        roulette_unique = len(jnp.unique(roulette_selected))
        
        # Both should select from the available individuals
        assert 1 <= tournament_unique <= len(fitness_scores)
        assert 1 <= roulette_unique <= len(fitness_scores)
    
    def test_selection_pressure_comparison(self):
        """Test selection pressure differences between methods."""
        # Create fitness with one dominant individual
        fitness_scores = jnp.array([0.05, 0.05, 0.85, 0.05])
        key = jar.PRNGKey(42)
        
        num_selections = 100
        
        # Tournament selection
        tournament_selector = TournamentSelection(number_of_tournaments=num_selections, tournament_size=4)
        tournament_fn = tournament_selector.get_compiled_function()
        tournament_selected = tournament_fn(fitness_scores, key)
        tournament_best_count = jnp.sum(tournament_selected == 2)
        
        # Roulette selection
        roulette_selector = RouletteSelection(number_choices=num_selections)
        roulette_fn = roulette_selector.get_compiled_function()
        roulette_selected = roulette_fn(fitness_scores, key)
        roulette_best_count = jnp.sum(roulette_selected == 2)
        
        # Both should favor the best individual, but potentially with different intensities
        assert tournament_best_count > 50  # Should select best individual frequently
        assert roulette_best_count > 60    # Should select roughly proportional to fitness
