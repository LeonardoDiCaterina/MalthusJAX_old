from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator
import jax
import jax.numpy as jnp



class TestBinarySumFitnessEvaluator:
    
    def test_initialization(self):
        """Test if the evaluator can be initialized without exploding."""
        evaluator = BinarySumFitnessEvaluator()
        assert evaluator is not None
    
    def test_individual_fitness(self):
        """Test evaluation of a single solution."""
        # Create a binary solution with known genome
        genome_init_params = {'array_size': 5, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 0, 1, 1, 0])  # 3 ones
        
        evaluator = BinarySumFitnessEvaluator()
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 3.0
    
    def test_tensor_fitness(self):
        """Test the tensor-based fitness function."""
        tensor = jnp.array([1, 1, 0, 0, 1])  # 3 ones
        
        evaluator = BinarySumFitnessEvaluator()
        fitness = evaluator.tensor_fitness_function(tensor)
        
        assert fitness == 3.0
    
    def test_batch_evaluation(self):
        """Test evaluation of multiple solutions in batch."""
        # Create solutions with known genomes
        genome_init_params = {'array_size': 5, 'p': 0.3}
        solutions = []
        genomes = [
            jnp.array([1, 1, 1, 1, 1]),  # 5 ones
            jnp.array([0, 0, 0, 0, 0]),  # 0 ones
            jnp.array([1, 0, 1, 0, 1])   # 3 ones
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            solutions.append(solution)
        
        evaluator = BinarySumFitnessEvaluator()
        evaluator.evaluate_solutions(solutions)
        
        assert solutions[0].raw_fitness == 5.0
        assert solutions[1].raw_fitness == 0.0
        assert solutions[2].raw_fitness == 3.0
    
    def test_population_stack_evaluation(self):
        """Test evaluation using population stack for efficiency."""
        # Create a stack of genome tensors
        population_stack = jnp.array([
            [1, 1, 1, 1, 1],  # 5 ones
            [0, 0, 0, 0, 0],  # 0 ones
            [1, 0, 1, 0, 1],  # 3 ones
            [1, 1, 0, 0, 0]   # 2 ones
        ])
        
        evaluator = BinarySumFitnessEvaluator()
        fitness_values = evaluator.evaluate_population_stack(population_stack)
        
        expected_fitness = jnp.array([5.0, 0.0, 3.0, 2.0])
        assert jnp.allclose(fitness_values, expected_fitness)
    
    def test_single_solution_evaluation(self):
        """Test single solution evaluation method."""
        genome_init_params = {'array_size': 4, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 0, 1, 1])  # 3 ones
        
        evaluator = BinarySumFitnessEvaluator()
        evaluator.evaluate_single_solution(solution)
        
        assert solution.raw_fitness == 3.0
    
    def test_empty_genome(self):
        """Test evaluation of empty genome."""
        genome_init_params = {'array_size': 0, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([])
        
        evaluator = BinarySumFitnessEvaluator()
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 0.0
    
    def test_all_ones_genome(self):
        """Test evaluation of genome with all ones."""
        genome_init_params = {'array_size': 10, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.ones(10)
        
        evaluator = BinarySumFitnessEvaluator()
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 10.0
    
    def test_all_zeros_genome(self):
        """Test evaluation of genome with all zeros."""
        genome_init_params = {'array_size': 7, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.zeros(7)
        
        evaluator = BinarySumFitnessEvaluator()
        fitness = evaluator.fitness_function(solution)
        
        assert fitness == 0.0
    
    def test_consistency_between_methods(self):
        """Test that tensor_fitness_function and fitness_function give same results."""
        genome_init_params = {'array_size': 6, 'p': 0.3}
        solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
        solution.genome.genome = jnp.array([1, 0, 1, 1, 0, 1])  # 4 ones
        
        evaluator = BinarySumFitnessEvaluator()
        
        fitness1 = evaluator.fitness_function(solution)
        fitness2 = evaluator.tensor_fitness_function(solution.genome.to_tensor())
        
        assert fitness1 == fitness2 == 4.0
    
    def test_batch_vs_individual_consistency(self):
        """Test that batch evaluation gives same results as individual evaluation."""
        genome_init_params = {'array_size': 5, 'p': 0.3}
        solutions = []
        genomes = [
            jnp.array([1, 1, 0, 1, 0]),  # 3 ones
            jnp.array([0, 1, 1, 1, 1]),  # 4 ones
            jnp.array([1, 0, 0, 0, 1])   # 2 ones
        ]
        
        for genome_data in genomes:
            solution = BinarySolution(genome_init_params=genome_init_params, random_init=False)
            solution.genome.genome = genome_data
            solutions.append(solution)
        
        evaluator = BinarySumFitnessEvaluator()
        
        # Get individual fitness values
        individual_fitness = [evaluator.fitness_function(sol) for sol in solutions]
        
        # Get batch fitness values
        evaluator.evaluate_solutions(solutions)
        batch_fitness = [sol.raw_fitness for sol in solutions]
        
        assert individual_fitness == batch_fitness
        assert individual_fitness == [3.0, 4.0, 2.0]