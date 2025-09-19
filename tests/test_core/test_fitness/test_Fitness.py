import pytest
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import jax # type: ignore

from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator
from malthusjax.core.fitness.real import SphereFitnessEvaluator, RastriginFitnessEvaluator, RosenbrockFitnessEvaluator, AckleyFitnessEvaluator, GriewankFitnessEvaluator
from malthusjax.core.fitness.real_ode import TaylorSeriesFitnessEvaluator
from malthusjax.core.fitness.permutation import SortingFitnessEvaluator, TSPFitnessEvaluator, FixedGroupingFitnessEvaluator
from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.genome.real import RealGenome
from malthusjax.core.genome.permutation import PermutationGenome 
from malthusjax.core.population.population import Population



class TestBinaryFitnessEvaluators:
    """Test suite for binary fitness evaluators."""
    
    def test_binary_sum_fitness_evaluator_initialization(self):
        """Test BinarySumFitnessEvaluator initialization."""
        evaluator = BinarySumFitnessEvaluator()
        assert evaluator.name == "BinarySumFitnessEvaluator"
        assert evaluator._tensor_fitness_fn is not None
        assert evaluator._batch_fitness_fn is not None
    
    def test_binary_sum_tensor_fitness_function(self):
        """Test tensor fitness function for binary sum."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Test with known values
        genome_tensor = jnp.array([1, 0, 1, 1, 0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 3.0  # Sum of ones
        
        # Test with all zeros
        genome_tensor = jnp.array([0, 0, 0, 0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 0.0
        
        # Test with all ones
        genome_tensor = jnp.array([1, 1, 1, 1])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 4.0
    
    def test_binary_sum_evaluate_single(self):
        """Test single genome evaluation."""
        evaluator = BinarySumFitnessEvaluator()
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        
        fitness = evaluator.evaluate_single(genome)
        assert isinstance(fitness, float)
        assert 0 <= fitness <= 5  # Should be between 0 and array size
        
        # Test with tensor directly
        tensor = jnp.array([1, 0, 1, 0, 1])
        fitness = evaluator.evaluate_single(tensor)
        assert fitness == 3.0
    
    def test_binary_sum_evaluate_batch(self):
        """Test batch evaluation."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Create a population
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=5,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 10, 'p': 0.5}
        )
        
        fitness_values = evaluator.evaluate_batch(population)
        assert len(fitness_values) == 5
        assert all(isinstance(f, float) for f in fitness_values)
        assert all(0 <= f <= 10 for f in fitness_values)
        
        # Test with return_tensors=True
        fitness_tensors = evaluator.evaluate_batch(population, return_tensors=True)
        assert fitness_tensors.shape == (5,)
        #assert jnp.issubdtype(fitness_tensors.dtype, (jnp.floating, jnp.integer))
    
    def test_knapsack_fitness_evaluator_initialization(self):
        """Test KnapsackFitnessEvaluator initialization."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0])
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit=10.0)
        assert evaluator.name == "KnapsackFitnessEvaluator"
        assert jnp.array_equal(evaluator.weights, weights)
        assert jnp.array_equal(evaluator.values, values)
        assert evaluator.weight_limit == 10.0
    
    def test_knapsack_tensor_fitness_function(self):
        """Test knapsack tensor fitness function."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0])
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit=10.0)
        
        # Valid solution within weight limit
        genome_tensor = jnp.array([1, 1, 0, 0])  # weight = 5, value = 7
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 7.0
        
        # Solution exceeding weight limit
        genome_tensor = jnp.array([1, 1, 1, 1])  # weight = 14, exceeds limit
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == -1.0  # Default penalty
    
    def test_knapsack_edge_cases(self):
        """Test knapsack edge cases."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([1.0, 2.0, 3.0])
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit=5.0, 
                                           default_exceding_weight_penalization=-100.0)
        
        # Empty knapsack
        genome_tensor = jnp.array([0, 0, 0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 0.0
        
        # Heavily penalized solution
        genome_tensor = jnp.array([1, 1, 1])  # weight = 6, exceeds limit
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == -100.0


class TestRealFitnessEvaluators:
    """Test suite for real-valued fitness evaluators."""
    
    def test_sphere_fitness_evaluator(self):
        """Test SphereFitnessEvaluator."""
        evaluator = SphereFitnessEvaluator()
        
        # Test with zeros (global minimum)
        genome_tensor = jnp.array([0.0, 0.0, 0.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert jnp.isclose(fitness, 0.0)
        
        # Test with known values
        genome_tensor = jnp.array([1.0, 2.0, 3.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        expected = 1.0**2 + 2.0**2 + 3.0**2  # 1 + 4 + 9 = 14
        assert jnp.isclose(fitness, expected)
    
    def test_rastrigin_fitness_evaluator(self):
        """Test RastriginFitnessEvaluator."""
        evaluator = RastriginFitnessEvaluator(A=10.0)
        
        # Test with zeros (global minimum)
        genome_tensor = jnp.array([0.0, 0.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert jnp.isclose(fitness, 0.0, atol=1e-6)
        
        # Test that it's not just sphere function
        #genome_tensor = jnp.array([3.0, -2.0, 5.0])
        #fitness = evaluator.tensor_fitness_function(genome_tensor)
        #sphere_fitness = jnp.sum(genome_tensor**2)
        #print(f"Rastrigin fitness: {fitness}, Sphere fitness: {sphere_fitness}")
        #assert not jnp.isclose(fitness, sphere_fitness)
    
    def test_rosenbrock_fitness_evaluator(self):
        """Test RosenbrockFitnessEvaluator."""
        evaluator = RosenbrockFitnessEvaluator()
        
        # Test with ones (global minimum)
        genome_tensor = jnp.array([1.0, 1.0, 1.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert jnp.isclose(fitness, 0.0, atol=1e-6)
        
        # Test with other values
        genome_tensor = jnp.array([0.0, 0.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness > 0  # Should be positive for non-optimal values
    
    def test_ackley_fitness_evaluator(self):
        """Test AckleyFitnessEvaluator."""
        evaluator = AckleyFitnessEvaluator()
        
        # Test with zeros (global minimum)
        genome_tensor = jnp.array([0.0, 0.0, 0.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert jnp.isclose(fitness, 0.0, atol=1e-6)
        
        # Test with custom parameters
        evaluator_custom = AckleyFitnessEvaluator(a=5.0, b=0.1, c=jnp.pi)
        assert evaluator_custom.a == 5.0
        assert evaluator_custom.b == 0.1
        assert evaluator_custom.c == jnp.pi
    
    def test_griewank_fitness_evaluator(self):
        """Test GriewankFitnessEvaluator."""
        evaluator = GriewankFitnessEvaluator()
        
        # Test with zeros (global minimum)
        genome_tensor = jnp.array([0.0, 0.0, 0.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert jnp.isclose(fitness, 0.0, atol=1e-6)
        
        # Test with other values
        genome_tensor = jnp.array([1.0, 2.0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness >= 0  # Should be non-negative
    
    def test_real_evaluators_batch_processing(self):
        """Test batch processing for real evaluators."""
        evaluators = [
            SphereFitnessEvaluator(),
            RastriginFitnessEvaluator(),
            RosenbrockFitnessEvaluator(),
            AckleyFitnessEvaluator(),
            GriewankFitnessEvaluator()
        ]
        
        population = Population(
            genome_cls=RealGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'minval': -5.0, 'maxval': 5.0}
        )
        
        for evaluator in evaluators:
            fitness_values = evaluator.evaluate_batch(population)
            assert len(fitness_values) == 3
            assert all(isinstance(f, float) for f in fitness_values)


class TestTaylorSeriesFitnessEvaluator:
    """Test suite for TaylorSeriesFitnessEvaluator."""
    
    def test_taylor_series_initialization(self):
        """Test TaylorSeriesFitnessEvaluator initialization."""
        def target_func(x):
            return x**2 + 2*x + 1
        
        x_values = jnp.linspace(-2, 2, 10)
        evaluator = TaylorSeriesFitnessEvaluator(target_func, x_values)
        
        assert evaluator.target_function is not None
        assert jnp.array_equal(evaluator.x_values, x_values)
    
    def test_taylor_series_tensor_fitness_function(self):
        """Test Taylor series tensor fitness function."""
        # Target function: f(x) = x^2 (coefficients: [0, 0, 1])
        def target_func(x):
            return x**2
        
        x_values = jnp.array([0.0, 1.0, 2.0])
        evaluator = TaylorSeriesFitnessEvaluator(target_func, x_values)
        
        # Perfect coefficients for x^2
        genome_tensor = jnp.array([0.0, 0.0, 1.0])
        
        # Note: This might print debug info due to the print statement in the function
        # We expect a very good fitness (close to 0 for MSE, so negative MSE close to 0)
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness <= 0  # Should be negative MSE
    
    def test_taylor_series_from_coefficients(self):
        """Test taylor_series_from_coefficients function."""
        from malthusjax.core.fitness.real_ode import taylor_series_from_coefficients
        
        # Test polynomial: 1 + 2x + 3x^2
        coefficients = jnp.array([1.0, 2.0, 3.0])
        taylor_func = taylor_series_from_coefficients(coefficients)
        
        # Evaluate at x = 2: 1 + 2*2 + 3*4 = 1 + 4 + 12 = 17
        result = taylor_func(2.0)
        assert jnp.isclose(result, 17.0)
        
        # Evaluate at x = 0: should be 1
        result = taylor_func(0.0)
        assert jnp.isclose(result, 1.0)


class TestPermutationFitnessEvaluators:
    """Test suite for permutation fitness evaluators."""
    
    def test_sorting_fitness_evaluator(self):
        """Test SortingFitnessEvaluator."""
        evaluator = SortingFitnessEvaluator()
        assert evaluator.name == "SortingFitnessEvaluator"
        
        # Test with already sorted array
        genome_tensor = jnp.array([0, 1, 2, 3, 4])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness == 0.0  # Perfect sort
        
        # Test with reverse sorted array
        genome_tensor = jnp.array([4, 3, 2, 1, 0])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert fitness < 0  # Should be negative (penalized)
    
    def test_tsp_fitness_evaluator(self):
        """Test TSPFitnessEvaluator."""
        # Create a simple distance matrix
        distance_matrix = jnp.array([
            [0, 1, 2, 3],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [3, 2, 1, 0]
        ])
        
        evaluator = TSPFitnessEvaluator(distance_matrix)
        assert evaluator.name == "TSPFitnessEvaluator"
        assert jnp.array_equal(evaluator.distance_matrix, distance_matrix)
        
        # Test with a simple tour
        genome_tensor = jnp.array([0, 1, 2, 3])
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        # Distance: 0->1 (1) + 1->2 (1) + 2->3 (1) = 3
        assert fitness == 3.0
    
    def test_fixed_grouping_fitness_evaluator(self):
        """Test FixedGroupingFitnessEvaluator."""
        # Create a simple penalty matrix
        penalty_matrix = jnp.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        evaluator = FixedGroupingFitnessEvaluator(group_size=2, penalty_matrix=penalty_matrix)
        assert evaluator.name == "FixedGroupingFitnessEvaluator"
        assert evaluator.group_size == 2
        
        # Test with a simple permutation
        genome_tensor = jnp.array([0, 1, 2])  # Will form one complete group [0,1] and partial group
        fitness = evaluator.tensor_fitness_function(genome_tensor)
        assert isinstance(fitness, (float, jnp.ndarray))


class TestFitnessEvaluatorBase:
    """Test suite for base fitness evaluator functionality."""
    
    def test_get_tensor_fitness_function(self):
        """Test get_tensor_fitness_function method."""
        evaluator = BinarySumFitnessEvaluator()
        tensor_fn = evaluator.get_tensor_fitness_function()
        
        assert callable(tensor_fn)
        
        # Test that it's JIT compiled
        genome_tensor = jnp.array([1, 0, 1])
        result = tensor_fn(genome_tensor)
        assert result == 2.0
    
    def test_get_batch_fitness_function(self):
        """Test get_batch_fitness_function method."""
        evaluator = SphereFitnessEvaluator()
        batch_fn = evaluator.get_batch_fitness_function()
        
        assert callable(batch_fn)
        
        # Test batch evaluation
        genome_stack = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]])
        results = batch_fn(genome_stack)
        
        assert results.shape == (3,)
        assert jnp.isclose(results[0], 5.0)  # 1^2 + 2^2
        assert jnp.isclose(results[1], 25.0)  # 3^2 + 4^2
        assert jnp.isclose(results[2], 0.0)   # 0^2 + 0^2
    
    def test_debug_tensor_fitness_function(self):
        """Test debug_tensor_fitness_function method."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Test cases with known expected results
        test_genomes = [
            jnp.array([1, 1, 1, 1]),  # All ones
            jnp.array([0, 0, 0, 0]),  # All zeros
            jnp.array([1, 0, 1, 0])   # Alternating
        ]
        expected_fitness = [4.0, 0.0, 2.0]
        
        # This will print debug information
        evaluator.debug_tensor_fitness_function(test_genomes, expected_fitness)
    
    def test_get_compatibility_info(self):
        """Test get_compatibility_info method."""
        evaluators = [
            BinarySumFitnessEvaluator(),
            SphereFitnessEvaluator(),
            SortingFitnessEvaluator()
        ]
        
        for evaluator in evaluators:
            info = evaluator.get_compatibility_info()
            assert isinstance(info, dict)
            assert info['requires_tensor_function'] == True
            assert info['supports_batch_evaluation'] == True
            assert info['supports_population_stack'] == True
            assert info['jax_compatible'] == True
    
    def test_callable_interface(self):
        """Test the __call__ interface."""
        evaluator = BinarySumFitnessEvaluator()
        
        population = Population(
            genome_cls=BinaryGenome,
            pop_size=3,
            random_init=True,
            random_key=jar.PRNGKey(42),
            genome_init_params={'array_size': 5, 'p': 0.5}
        )
        
        # Test callable interface
        fitness_values = evaluator(population)
        assert len(fitness_values) == 3
        assert all(isinstance(f, (float, jnp.floating)) for f in fitness_values)
        
        # Test with return_tensors=True
        fitness_tensors = evaluator(population, return_tensors=True)
        assert fitness_tensors.shape == (3,)
    
    def test_error_handling(self):
        """Test error handling in fitness evaluators."""
        evaluator = BinarySumFitnessEvaluator()
        
        # Test with invalid input types
        with pytest.raises((ValueError, TypeError)):
            evaluator.evaluate_batch("not_a_list")
        
        # Test with empty batch
        empty_fitness = evaluator.evaluate_batch([])
        assert empty_fitness == []


class TestFitnessJITCompatibility:
    """Test JAX JIT compatibility of fitness functions."""
    
    def test_jit_compilation(self):
        """Test that all fitness functions can be JIT compiled."""
        evaluators = [
            BinarySumFitnessEvaluator(),
            SphereFitnessEvaluator(),
            RastriginFitnessEvaluator(),
            SortingFitnessEvaluator()
        ]
        
        for evaluator in evaluators:
            # Test JIT compilation
            jit_fn = jax.jit(evaluator.tensor_fitness_function)
            
            # Create appropriate test tensor
            if isinstance(evaluator, BinarySumFitnessEvaluator):
                test_tensor = jnp.array([1, 0, 1, 0])
            elif isinstance(evaluator, SortingFitnessEvaluator):
                test_tensor = jnp.array([3, 1, 2, 0])
            else:  # Real-valued evaluators
                test_tensor = jnp.array([1.0, 2.0, 3.0])
            
            # Should not raise errors
            result = jit_fn(test_tensor)
            assert isinstance(result, (float, jnp.floating, jnp.ndarray))
    
    def test_vmap_compatibility(self):
        """Test that fitness functions work with vmap."""
        evaluator = SphereFitnessEvaluator()
        vmap_fn = jax.vmap(evaluator.tensor_fitness_function)
        
        # Test batch evaluation with vmap
        batch_tensors = jnp.array([[1.0, 2.0], [3.0, 4.0], [0.0, 1.0]])
        results = vmap_fn(batch_tensors)
        
        assert results.shape == (3,)
        assert jnp.isclose(results[0], 5.0)   # 1^2 + 2^2
        assert jnp.isclose(results[1], 25.0)  # 3^2 + 4^2
        assert jnp.isclose(results[2], 1.0)   # 0^2 + 1^2

