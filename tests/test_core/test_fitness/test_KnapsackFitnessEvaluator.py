from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator
import jax # type: ignore
import jax.numpy as jnp # type: ignore


class TestKnapsackFitnessEvaluator:
    
    def test_initialization(self):
        """Test if the evaluator can be initialized without exploding."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0, 1.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0, 2.0])
        weight_limit = 10.0
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        assert evaluator is not None
        assert jnp.array_equal(evaluator.weights, weights)
        assert jnp.array_equal(evaluator.values, values)
        assert evaluator.weight_limit == weight_limit
        assert evaluator.default_exceding_weight_penalization == -1.0
    
    def test_initialization_with_custom_penalty(self):
        """Test initialization with custom penalty value."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([1.0, 2.0, 3.0])
        weight_limit = 5.0
        penalty = -100.0
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit, penalty)
        assert evaluator.default_exceding_weight_penalization == penalty
    
    def test_individual_fitness_within_limit(self):
        """Test evaluation of a single solution within weight limit."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0, 1.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0, 2.0])
        weight_limit = 10.0
        
        genome_init_params = {'array_size': 5, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1, 0, 0, 1])  # Weight: 2+3+1=6, Value: 3+4+2=9
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 9.0  # Total value
    
    def test_individual_fitness_exceeds_limit(self):
        """Test evaluation of a single solution exceeding weight limit."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0, 1.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0, 2.0])
        weight_limit = 10.0
        
        genome_init_params = {'array_size': 5, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1, 1, 1, 0])  # Weight: 2+3+4+5=14 > 10
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == -1.0  # Default penalty
    
    def test_tensor_fitness_within_limit(self):
        """Test the tensor-based fitness function within weight limit."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([2.0, 3.0, 5.0])
        weight_limit = 5.0
        
        tensor = jnp.array([1, 1, 0])  # Weight: 1+2=3, Value: 2+3=5
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.tensor_fitness_function(tensor)
        
        assert fitness == 5.0
    
    def test_tensor_fitness_exceeds_limit(self):
        """Test the tensor-based fitness function exceeding weight limit."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([2.0, 3.0, 5.0])
        weight_limit = 5.0
        
        tensor = jnp.array([1, 1, 1])  # Weight: 1+2+3=6 > 5
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.tensor_fitness_function(tensor)
        
        assert fitness == -1.0  # Default penalty
    
    def test_batch_evaluation(self):
        """Test evaluation of multiple solutions in batch."""
        weights = jnp.array([2.0, 3.0, 4.0, 5.0])
        values = jnp.array([3.0, 4.0, 5.0, 6.0])
        weight_limit = 10.0
        
        genome_init_params = {'array_size': 4, 'p': 0.3}
        solutions = []
        genomes = [
            jnp.array([1, 1, 0, 0]),  # Weight: 2+3=5, Value: 3+4=7 (valid)
            jnp.array([0, 0, 0, 0]),  # Weight: 0, Value: 0 (valid)
            jnp.array([1, 1, 1, 1]),  # Weight: 2+3+4+5=14 > 10 (invalid)
            jnp.array([1, 0, 1, 0])   # Weight: 2+4=6, Value: 3+5=8 (valid)
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            solutions.append(solution)
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        evaluator.evaluate_solutions(solutions)
        
        assert solutions[0].raw_fitness == 7.0
        assert solutions[1].raw_fitness == 0.0
        assert solutions[2].raw_fitness == -1.0  # Penalty for exceeding weight
        assert solutions[3].raw_fitness == 8.0
    
    def test_population_stack_evaluation(self):
        """Test evaluation using population stack for efficiency."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        values = jnp.array([2.0, 3.0, 4.0, 5.0])
        weight_limit = 6.0
        
        population_stack = jnp.array([
            [1, 1, 0, 0],  # Weight: 1+2=3, Value: 2+3=5 (valid)
            [0, 0, 1, 1],  # Weight: 3+4=7 > 6 (invalid)
            [1, 0, 1, 0],  # Weight: 1+3=4, Value: 2+4=6 (valid)
            [0, 1, 0, 1]   # Weight: 2+4=6, Value: 3+5=8 (valid, at limit)
        ])
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness_values = evaluator.evaluate_population_stack(population_stack)
        
        expected_fitness = jnp.array([5.0, -1.0, 6.0, 8.0])
        assert jnp.allclose(fitness_values, expected_fitness)
    
    def test_single_solution_evaluation(self):
        """Test single solution evaluation method."""
        weights = jnp.array([1.5, 2.5, 3.5])
        values = jnp.array([2.5, 3.5, 4.5])
        weight_limit = 5.0
        
        genome_init_params = {'array_size': 3, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1, 0])  # Weight: 1.5+2.5=4, Value: 2.5+3.5=6
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        evaluator.evaluate_single_solution(solution)
        
        assert solution.raw_fitness == 6.0
    
    def test_empty_knapsack(self):
        """Test evaluation with no items selected."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([2.0, 3.0, 4.0])
        weight_limit = 5.0
        
        genome_init_params = {'array_size': 3, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([0, 0, 0])  # No items selected
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 0.0
    
    def test_exactly_at_weight_limit(self):
        """Test evaluation when weight exactly equals the limit."""
        weights = jnp.array([2.0, 3.0, 5.0])
        values = jnp.array([4.0, 6.0, 10.0])
        weight_limit = 10.0
        
        genome_init_params = {'array_size': 3, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1, 1])  # Weight: 2+3+5=10 (exactly at limit)
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 20.0  # Should be valid: 4+6+10=20
    
    def test_consistency_between_methods(self):
        """Test that tensor_fitness_function and fitness_function give same results."""
        weights = jnp.array([1.0, 2.0, 3.0, 4.0])
        values = jnp.array([1.5, 2.5, 3.5, 4.5])
        weight_limit = 7.0
        
        genome_init_params = {'array_size': 4, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 0, 1, 0])  # Weight: 1+3=4, Value: 1.5+3.5=5
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        
        fitness1 = evaluator.fitness_function(solution)
        fitness2 = evaluator.tensor_fitness_function(solution.genome.to_tensor())
        
        assert fitness1 == fitness2 == 5.0
    
    def test_batch_vs_individual_consistency(self):
        """Test that batch evaluation gives same results as individual evaluation."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([2.0, 4.0, 6.0])
        weight_limit = 5.0
        
        genome_init_params = {'array_size': 3, 'p': 0.3}
        solutions = []
        genomes = [
            jnp.array([1, 1, 0]),  # Weight: 1+2=3, Value: 2+4=6 (valid)
            jnp.array([0, 1, 1]),  # Weight: 2+3=5, Value: 4+6=10 (valid, at limit)
            jnp.array([1, 1, 1])   # Weight: 1+2+3=6 > 5 (invalid)
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            solutions.append(solution)
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        
        # Get individual fitness values
        individual_fitness = [evaluator.fitness_function(sol) for sol in solutions]
        
        # Get batch fitness values
        evaluator.evaluate_solutions(solutions)
        batch_fitness = [sol.raw_fitness for sol in solutions]
        
        assert individual_fitness == batch_fitness
        assert individual_fitness == [6.0, 10.0, -1.0]
    
    def test_custom_penalty_value(self):
        """Test using custom penalty value for weight violations."""
        weights = jnp.array([5.0, 6.0])
        values = jnp.array([10.0, 12.0])
        weight_limit = 8.0
        custom_penalty = -50.0
        
        genome_init_params = {'array_size': 2, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1])  # Weight: 5+6=11 > 8
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit, custom_penalty)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == custom_penalty
    
    def test_fractional_weights_and_values(self):
        """Test with fractional weights and values."""
        weights = jnp.array([1.5, 2.3, 3.7, 0.8])
        values = jnp.array([2.1, 3.4, 5.2, 1.3])
        weight_limit = 6.0
        
        genome_init_params = {'array_size': 4, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 1, 0, 1])  # Weight: 1.5+2.3+0.8=4.6, Value: 2.1+3.4+1.3=6.8
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        expected_value = 2.1 + 3.4 + 1.3
        assert abs(fitness - expected_value) < 1e-6
    
    def test_zero_weight_limit(self):
        """Test behavior with zero weight limit."""
        weights = jnp.array([1.0, 2.0, 3.0])
        values = jnp.array([2.0, 3.0, 4.0])
        weight_limit = 0.0
        
        genome_init_params = {'array_size': 3, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([0, 0, 0])  # No items selected
        
        evaluator = KnapsackFitnessEvaluator(weights, values, weight_limit)
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 0.0  # Should be valid with no items
        
        # Test with items selected (should be invalid)
        solution.genome.genome = jnp.array([1, 0, 0])  # Weight: 1 > 0
        fitness = evaluator.fitness_function(solution)
        assert fitness == -1.0  # Should be penalized

