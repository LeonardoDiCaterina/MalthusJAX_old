
import jax.numpy as jnp #type: ignore
import jax.random as jar #type: ignore
import jax #type: ignore

import pytest
from malthusjax.core.population.population import Population

from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.genome.permutation import PermutationGenome
from malthusjax.core.genome.categorical import CategoricalGenome
from malthusjax.core.genome.real import RealGenome


class TestPopulation:
    
    def test_initialization_default(self):
        """Test basic population initialization."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=10,
            random_init=False
        )
        assert population._pop_size == 10
        assert population.size == 0
        assert population._genome_cls == BinaryGenome

    def test_initialization_with_random_init(self):
        """Test population initialization with random initialization."""
        population = Population(
            genome_cls=CategoricalGenome,
            pop_size=5,
            random_key=jar.PRNGKey(42),
            random_init=True,
            genome_init_params={'array_size': 10, 'num_categories': 3}
        )
        
        assert population._pop_size == 5
        assert hasattr(population, '_genome_init_params')
        assert population._genome_init_params['array_size'] == 10
        assert population._genome_init_params['num_categories'] == 3

    def test_add_genome(self):
        """Test adding genomes to the population."""
        population = Population(
            genome_cls=PermutationGenome,
            pop_size=5,
            random_init=False
        )
        
        # Create genomes
        genome1 = PermutationGenome(permutation_start=0, permutation_end=5, random_init=True, random_key=jar.PRNGKey(42))
        genome2 = PermutationGenome(permutation_start=0, permutation_end=5, random_init=True, random_key=jar.PRNGKey(123))

        # Add genomes
        population.add_genome(genome1)
        assert population.size == 1

        population.add_genome(genome2)
        assert population.size == 2
        
        # Check genomes are stored correctly
        genomes = population.get_genomes()
        assert len(genomes) == 2
        assert genomes[0] == genome1
        assert genomes[1] == genome2

    def test_add_genome_exceeds_capacity(self):
        """Test adding genomes beyond maximum capacity."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=2,
            random_init=False
        )
        
        # Fill population to capacity
        for i in range(2):
            genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(i))
            population.add_genome(genome)
        
        # Try to add one more genome
        genome_extra = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(999))
        with pytest.raises(ValueError, match="Population already at maximum capacity"):
            population.add_genome(genome_extra)

    def test_get_genomes_empty(self):
        """Test getting all genomes from the population."""
        population = Population(
            genome_cls=RealGenome,
            pop_size=5,
            random_init=False
        )
        
        # Initially empty
        assert population.get_genomes() == [] or population.get_genomes() is None

        # Add some genomes
        genomes_to_add = []
        for i in range(3):
            genome = RealGenome(array_size=5, minval=0.0, maxval=1.0, random_init=True, random_key=jar.PRNGKey(i))
            genomes_to_add.append(genome)
            population.add_genome(genome)

        retrieved_genomes = population.get_genomes()
        assert len(retrieved_genomes) == 3
        for g_added, g_retrieved in zip(genomes_to_add, retrieved_genomes):
            assert g_added == g_retrieved
            
    def test_size_property(self):
        """Test the size property."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=10,
            random_init=False
        )
        
        assert population.size == 0
        
        # Add genomes and check size updates
        for i in range(5):
            genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(i))
            population.add_genome(genome)
            assert population.size == i + 1

    def test_random_stack_classmethod(self):
        """Test the random_stack class method."""
        stack_size = 10
        genome_init_params = {'array_size': 8, 'p': 0.3}
        
        stack = Population.random_stack(
            genome_cls=BinaryGenome,
            stack_size=stack_size,
            random_key=jar.PRNGKey(42),
            genome_init_params=genome_init_params
        )
        
        assert stack.shape[0] == stack_size
        assert stack.shape[1] == genome_init_params['array_size']
        # Check reproducibility
        stack2 = Population.random_stack(
            genome_cls=BinaryGenome,
            stack_size=stack_size,
            random_key=jar.PRNGKey(42),
            genome_init_params=genome_init_params
        )
        assert jnp.array_equal(stack, stack2)

    def test_from_stack(self):
        """Test creating population from a stack of genomes."""
        population = Population(
            genome_cls=RealGenome,
            pop_size=5,
            random_init=False,
            genome_init_params={'array_size': 4, 'minval': -1.0, 'maxval': 1.0}
        )
        
        # Create a stack
        stack = Population.random_stack(
            genome_cls=RealGenome,
            stack_size=3,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 4, 'minval': -1.0, 'maxval': 1.0}
        )
        
        new_population = population.from_stack(stack)
        assert new_population._pop_size == 5
        assert len(new_population) == 3

    def test_from_list(self):
        """Test creating population from a list of genomes."""
        population = Population(
            genome_cls=CategoricalGenome,
            pop_size=5,
            random_init=False,
            genome_init_params={'array_size': 6, 'num_categories': 4}
        )
        
        # Create list of genomes
        genomes = [
            CategoricalGenome(array_size=6, num_categories=4, random_init=True, random_key=jar.PRNGKey(i))
            for i in range(3)
        ]
        
        new_population = population.from_list(genomes)
        assert new_population._pop_size == 5
        assert len(new_population) == 3

    def test_from_array_of_indexes(self):
        """Test creating population from array of indexes."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=10,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'p': 0.5}
        )
        
        # Select subset of population
        indexes = jnp.array([0, 2, 4, 1])
        new_population = population.from_array_of_indexes(indexes)
        
        assert len(new_population) == 4
        assert new_population._pop_size == 10

    def test_to_stack(self):
        """Test converting population to stack."""
        population = Population(
            genome_cls=RealGenome,
            pop_size=5,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 4, 'minval': 0.0, 'maxval': 1.0}
        )
        
        stack = population.to_stack()
        assert stack.shape[0] == 5
        assert stack.shape[1] == 4

    def test_to_list(self):
        """Test converting population to list."""
        population = Population(
            genome_cls=PermutationGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'permutation_start': 0, 'permutation_end': 5}
        )
        
        genome_list = population.to_list()
        assert len(genome_list) == 3
        assert all(isinstance(g, PermutationGenome) for g in genome_list)

    def test_fitness_values(self):
        """Test fitness value management."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=5,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 10, 'p': 0.5}
        )
        
        # Set fitness values
        fitness_values = jnp.array([0.8, 0.6, 0.9, 0.7, 0.5])
        population.set_fitness_values(fitness_values)
        
        retrieved_fitness = population.get_fitness_values()
        assert jnp.array_equal(fitness_values, retrieved_fitness)

    def test_best_genome(self):
        """Test getting the best genome."""
        population = Population(
            genome_cls=RealGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'minval': 0.0, 'maxval': 1.0}
        )
        
        # Set fitness values
        fitness_values = jnp.array([0.6, 0.9, 0.7])
        population.set_fitness_values(fitness_values)
        
        best_genome = population.get_best_genome()
        best_fitness = best_genome.fitness
        assert best_fitness == 0.9
        assert isinstance(best_genome, RealGenome)

    def test_jit_functions(self):
        """Test JIT-compiled functions."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=5,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 8, 'p': 0.3}
        )
        
        # Test distance matrix function
        distance_matrix_fn = population.get_distance_matrix_function_jit()
        stack = population.to_stack()
        distance_matrix = distance_matrix_fn(stack)
        assert distance_matrix.shape == (5, 5)
        assert jnp.all(jnp.diag(distance_matrix) == 0)  # Distance to self is 0
        
        # Test autocorrection function
        autocorrection_fn = population.get_autocorrection_function_jit()
        invalid_stack = jnp.array([[2, -1, 0, 1, 0, 1, 1, 0]])  # Invalid binary values
        corrected = autocorrection_fn(invalid_stack)
        assert jnp.all((corrected == 0) | (corrected == 1))
        
        # Test init function
        init_fn = population.get_init_function_jit()
        new_genome = init_fn(jar.PRNGKey(123))
        assert new_genome.shape == (8,)

    def test_population_indexing(self):
        """Test population indexing operations."""
        population = Population(
            genome_cls=CategoricalGenome,
            pop_size=5,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 6, 'num_categories': 3}
        )
        
        # Test __len__
        assert len(population) == 5
        
        # Test __getitem__
        genome = population[0]
        assert genome.shape == (6,)
        
        # Test out of bounds
        with pytest.raises(IndexError):
            _ = population[10]

    def test_population_iteration(self):
        """Test population iteration."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'p': 0.5}
        )
        
        # Test iteration
        count = 0
        for genome in population:
            assert genome.shape == (5,)
            count += 1
        assert count == 3

    def test_set_genomes(self):
        """Test setting genomes directly."""
        population = Population(
            genome_cls=RealGenome,
            pop_size=5,
            random_init=False,
            genome_init_params={'array_size': 4, 'minval': 0.0, 'maxval': 1.0}
        )
        
        # Create solutions
        solutions = Population.random_stack(
            genome_cls=RealGenome,
            stack_size=3,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 4, 'minval': 0.0, 'maxval': 1.0}
        )

        population.set_genomes(solutions)
        assert len(population) == 3
        
        # Test exceeding capacity
        large_solutions = Population.random_stack(
            genome_cls=RealGenome,
            stack_size=10,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 4, 'minval': 0.0, 'maxval': 1.0}
        )

        with pytest.raises(ValueError, match="Number of genomes exceeds population size"):
            population.set_genomes(large_solutions)

    def test_string_representations(self):
        """Test string representations."""
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=10,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'p': 0.5}
        )
        
        str_repr = str(population)
        assert "Population" in str_repr
        assert "BinaryGenome" in str_repr
        assert "10" in str_repr
        
        repr_str = repr(population)
        assert repr_str == str_repr

    def test_static_jit_methods(self):
        """Test static JIT-compiled methods."""
        # Test sort by fitness
        sort_fn = Population.get_sort_by_fitness_jit()
        genomes = jnp.array([[1, 0, 1], [0, 1, 0], [1, 1, 1]])
        fitness = jnp.array([0.5, 0.8, 0.3])
        
        sorted_genomes, sorted_fitness = sort_fn(genomes, fitness)
        assert jnp.array_equal(sorted_fitness, jnp.array([0.8, 0.5, 0.3]))
        # Test best index
        best_index_fn = Population.get_get_best_index_jit()
        fitness = jnp.array([0.5, 0.8, 0.3])
        best_idx = best_index_fn(fitness)
        assert best_idx == 1
        # Test best fitness
        best_fitness_fn = Population.get_get_best_fitness_jit()
        best_fit = best_fitness_fn(fitness)
        assert best_fit == 0.8
        # Test from array of indexes
        from_indexes_fn = Population.get_from_array_of_indexes_jit()
        stack = jnp.array([[1, 0], [0, 1], [1, 1]])
        indexes = jnp.array([0, 2])
        selected = from_indexes_fn(stack, indexes)
        expected = jnp.array([[1, 0], [1, 1]])
        assert jnp.array_equal(selected, expected)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        # Empty population operations
        empty_population = Population(
            genome_cls=BinaryGenome,
            pop_size=5,
            random_init=False
        )
        
        # Test operations on empty population
        with pytest.raises((ValueError, IndexError)):
            empty_population.get_best_genome()
        
        # Test invalid fitness values size
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'p': 0.5}
        )
        
        wrong_size_fitness = jnp.array([0.5, 0.8])  # Wrong size
        with pytest.raises(ValueError, match="Fitness values array size does not match"):
            population.set_fitness_values(wrong_size_fitness)

    def test_genome_init_params_getter(self):
        """Test genome initialization parameters getter."""
        init_params = {'array_size': 10, 'num_categories': 5}
        population = Population(
            genome_cls=CategoricalGenome,
            pop_size=5,
            random_init=False,
            genome_init_params=init_params
        )
        
        retrieved_params = population.get_genome_init_params()
        assert retrieved_params == init_params

    def test_random_key_getter(self):
        """Test random key getter."""
        key = jar.PRNGKey(42)
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=5,
            random_init=False,
            random_key=key
        )
        
        retrieved_key = population.get_random_key()
        assert jnp.array_equal(retrieved_key, key)