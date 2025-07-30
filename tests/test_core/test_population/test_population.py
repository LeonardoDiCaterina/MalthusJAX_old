"""
Tests for the Population class in MalthusJAX.
"""

import pytest
import jax # type: ignore
import jax.numpy as jnp  # type: ignore
import jax.random as jar  # type: ignore

from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.population.population import Population
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator
from malthusjax.core.base import SerializationContext


class TestPopulation:
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.random_key = jar.PRNGKey(42)
        self.genome_init_params = {
            'array_size': 5,
            'p': 0.5
        }
        self.max_size = 10
        
    def test_initialization_with_random_init(self):
        """Test population initialization with random solutions."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_key=self.random_key,
            random_init=True,
            genome_init_params=self.genome_init_params
        )
        
        assert population.size == self.max_size
        assert len(population.get_solutions()) == self.max_size
        assert population.max_size == self.max_size
        assert population.solution_class == BinarySolution
        assert population.genome_init_params == self.genome_init_params
        
        # Check that all solutions are properly initialized
        for solution in population.get_solutions():
            assert isinstance(solution, BinarySolution)
            assert solution.genome.array_size == self.genome_init_params['array_size']
            assert solution.genome.p == self.genome_init_params['p']
            assert solution.genome.is_valid
    
    def test_initialization_without_random_init(self):
        """Test population initialization without random solutions."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_key=self.random_key,
            random_init=False,
            genome_init_params=self.genome_init_params
        )
        
        assert population.size == 0
        assert len(population.get_solutions()) == 0
        assert population.max_size == self.max_size
        assert population._best_solution is None
    
    def test_initialization_with_fitness_transform(self):
        """Test population initialization with fitness transformation."""
        fitness_transform = lambda x: x * 2
        
        population = Population(
            solution_class=BinarySolution,
            max_size=3,
            random_key=self.random_key,
            random_init=True,
            genome_init_params=self.genome_init_params,
            fitness_transform=fitness_transform
        )
        
        assert population.fitness_transform == fitness_transform
        # Check that solutions have the fitness transform
        for solution in population.get_solutions():
            assert solution.fitness_transform == fitness_transform
    
    def test_add_solution(self):
        """Test adding a solution to the population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        
        population.add_solution(solution)
        assert population.size == 1
        assert population.get_solutions()[0] is solution
        assert population._best_solution is None  # Should be reset when adding
        
    def test_add_solution_max_capacity(self):
        """Test adding a solution when population is at max capacity."""
        population = Population(
            solution_class=BinarySolution,
            max_size=1,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        solution1 = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        
        solution2 = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=jar.PRNGKey(43)
        )
        
        population.add_solution(solution1)
        
        with pytest.raises(ValueError, match="Population already at maximum capacity: 1"):
            population.add_solution(solution2)
    
    def test_get_best_solution(self):
        """Test getting the best solution from the population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add solutions with different fitness values
        solutions = []
        for i in range(5):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            solution.raw_fitness = float(i)  # Set different fitness values
            solutions.append(solution)
            population.add_solution(solution)
        
        best_solution = population.get_best_solution()
        assert best_solution.fitness == 4.0  # Highest fitness
        assert best_solution is solutions[4]
    
    def test_get_best_solution_empty(self):
        """Test getting the best solution from an empty population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        with pytest.raises(ValueError, match="Cannot get best solution from empty population"):
            population.get_best_solution()
    
    def test_get_fitness_values(self):
        """Test getting fitness values from the population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add solutions with different fitness values
        fitness_values = [1.0, 3.0, 2.0, 5.0]
        for i, fitness in enumerate(fitness_values):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            solution.raw_fitness = fitness
            population.add_solution(solution)
        
        retrieved_fitness = population.get_fitness_values()
        expected_fitness = jnp.array(fitness_values)
        
        assert jnp.allclose(retrieved_fitness, expected_fitness)
    
    def test_get_fitness_values_empty(self):
        """Test getting fitness values from an empty population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        fitness_values = population.get_fitness_values()
        assert fitness_values.size == 0
        assert jnp.array_equal(fitness_values, jnp.array([]))
    
    def test_get_statistics(self):
        """Test getting statistics from the population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add solutions with known fitness values [0, 1, 2, 3, 4]
        for i in range(5):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            solution.raw_fitness = float(i)
            population.add_solution(solution)
        
        stats = population.get_statistics()
        
        assert stats['pop_size'] == 5
        assert stats['max_fitness'] == 4.0
        assert stats['min_fitness'] == 0.0
        assert stats['avg_fitness'] == 2.0
        assert abs(stats['fitness_std'] - jnp.std(jnp.array([0, 1, 2, 3, 4]))) < 1e-6
        assert stats['25th_percentile'] == 1.0
        assert stats['50th_percentile'] == 2.0
        assert stats['75th_percentile'] == 3.0
    
    def test_get_statistics_empty(self):
        """Test getting statistics from an empty population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        stats = population.get_statistics()
        
        assert stats['pop_size'] == 0
        assert stats['max_fitness'] is None
        assert stats['min_fitness'] is None
        assert stats['avg_fitness'] is None
        assert stats['fitness_std'] is None
    
    def test_update_fitness(self):
        """Test updating fitness values in the population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Create solutions with known genomes
        genomes = [
            jnp.array([1, 1, 1, 1, 1]),  # 5 ones
            jnp.array([0, 0, 0, 0, 0]),  # 0 ones
            jnp.array([1, 0, 1, 0, 1])   # 3 ones
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=self.genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            population.add_solution(solution)
        
        evaluator = BinarySumFitnessEvaluator()
        population.update_fitness(evaluator)
        
        assert population.get_solutions()[0].raw_fitness == 5.0
        assert population.get_solutions()[1].raw_fitness == 0.0
        assert population.get_solutions()[2].raw_fitness == 3.0
        
        # Best solution should now be the first one (with 5 ones)
        assert population.get_best_solution().raw_fitness == 5.0
        assert population._best_solution is not None
    
    def test_update_fitness_empty_population(self):
        """Test updating fitness on empty population."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        evaluator = BinarySumFitnessEvaluator()
        # Should not raise an error
        population.update_fitness(evaluator)
        assert population.size == 0
    
    def test_to_stack(self):
        """Test converting population to stack."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add solutions with known genomes
        genomes = [
            jnp.array([1, 1, 1, 1, 1]),
            jnp.array([0, 0, 0, 0, 0]),
            jnp.array([1, 0, 1, 0, 1])
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=self.genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            population.add_solution(solution)
        
        stack = population.to_stack()
        
        # Check stack shape and values
        assert stack.shape == (3, 5)  # 3 solutions, 5 genes each
        assert jnp.array_equal(stack[0], genomes[0])
        assert jnp.array_equal(stack[1], genomes[1])
        assert jnp.array_equal(stack[2], genomes[2])
    
    def test_to_stack_empty(self):
        """Test converting empty population to stack."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        stack = population.to_stack()
        assert stack.size == 0
        assert jnp.array_equal(stack, jnp.array([]))
    
    def test_from_stack(self):
        """Test creating population from stack."""
        # Create original population
        original_population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add solutions with known genomes
        genomes = [
            jnp.array([1, 1, 1, 1, 1]),
            jnp.array([0, 0, 0, 0, 0]),
            jnp.array([1, 0, 1, 0, 1])
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=self.genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            original_population.add_solution(solution)
        
        # Convert to stack
        stack = original_population.to_stack()
        
        # Create context for reconstruction
        context = SerializationContext(
            genome_class=BinaryGenome,
            genome_init_params=self.genome_init_params,
            compatibility=None
        )
        
        # Create new population from stack
        new_population = original_population.from_stack(stack, context=context)
        
        # Check that the new population has the same genomes
        assert new_population.size == original_population.size
        
        for i in range(new_population.size):
            assert jnp.array_equal(
                new_population.get_solutions()[i].genome.to_tensor(),
                original_population.get_solutions()[i].genome.to_tensor()
            )
    
    def test_from_stack_with_fitness(self):
        """Test creating population from stack with fitness values."""
        stack = jnp.array([
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1]
        ])
        fitness_values = jnp.array([5.0, 0.0, 3.0])
        
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        context = SerializationContext(
            genome_class=BinaryGenome,
            genome_init_params=self.genome_init_params,
            compatibility=None
        )
        
        new_population = population.from_stack(stack, context=context, fitness_values=fitness_values)
        
        assert new_population.size == 3
        assert new_population.get_solutions()[0].raw_fitness == 5.0
        assert new_population.get_solutions()[1].raw_fitness == 0.0
        assert new_population.get_solutions()[2].raw_fitness == 3.0
    
    def test_from_solution_list(self):
        """Test creating population from solution list."""
        # Create original solutions
        solutions = []
        genomes = [
            jnp.array([1, 1, 1, 1, 1]),
            jnp.array([0, 0, 0, 0, 0]),
            jnp.array([1, 0, 1, 0, 1])
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=self.genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            solutions.append(solution)
        
        # Create population from solution list
        original_population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        new_population = original_population.from_solution_list(solutions)
        
        assert new_population.size == 3
        for i in range(3):
            assert jnp.array_equal(
                new_population.get_solutions()[i].genome.to_tensor(),
                solutions[i].genome.to_tensor()
            )
    
    def test_iterator_and_indexing(self):
        """Test population iterator and indexing operations."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        solutions = []
        for i in range(3):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            solutions.append(solution)
            population.add_solution(solution)
        
        # Test __len__
        assert len(population) == 3
        
        # Test __iter__
        count = 0
        for solution in population:
            assert solution in solutions
            count += 1
        assert count == 3
        
        # Test __getitem__
        assert population[0] is solutions[0]
        assert population[1] is solutions[1]
        assert population[2] is solutions[2]
        
        with pytest.raises(IndexError):
            _ = population[3]
        
        with pytest.raises(IndexError):
            _ = population[-1]  # Negative indices not supported
        
        # Test __contains__
        assert solutions[0] in population
        assert solutions[1] in population
    
    def test_validate(self):
        """Test population validation."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        # Add some valid solutions
        for i in range(3):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            population.add_solution(solution)
        
        # All solutions should be valid
        assert population.validate() is True
        
        # Empty population should return False
        empty_population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        assert empty_population.validate() is False
        
        # Test with invalid solution
        invalid_solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=False
        )
        invalid_solution.genome.genome = jnp.array([0, 2, 1, 0, 1])  # Invalid (contains 2)
        population.add_solution(invalid_solution)
        
        assert population.validate() is False
    
    def test_generate_random_keys(self):
        """Test random key generation."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        original_key = population.random_key
        
        # Generate some keys
        n_keys = 5
        keys = population.generate_random_keys(n_keys)
        
        # Should return the right number of keys
        assert len(keys) == n_keys
        
        # Each key should be a valid PRNGKey
        for key in keys:
            assert key.shape == (2,)
            
        # The population's random key should have been updated
        assert not jnp.array_equal(population.random_key, original_key)
        
        # Keys should be different from each other
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not jnp.array_equal(keys[i], keys[j])
    
    def test_size_property(self):
        """Test the size property."""
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        assert population.size == 0
        
        # Add solutions and check size updates
        for i in range(5):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            population.add_solution(solution)
            assert population.size == i + 1
    
    def test_inheritance_and_typing(self):
        """Test that Population properly inherits from AbstractPopulation."""
        from malthusjax.core.population.base import AbstractPopulation
        
        population = Population(
            solution_class=BinarySolution,
            max_size=self.max_size,
            random_init=False,
            random_key=self.random_key,
            genome_init_params=self.genome_init_params
        )
        
        assert isinstance(population, AbstractPopulation)
        assert population.solution_class == BinarySolution
        assert population.max_size == self.max_size