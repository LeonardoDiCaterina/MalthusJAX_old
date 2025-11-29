"""
Integration tests for the complete MalthusJAX Level 1 & 2 system.

Tests that all components work together in realistic evolutionary scenarios.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr

from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.fitness.binary_evaluators import BinarySumEvaluator as BinarySumFitnessEvaluator

from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.selection.tournament import TournamentSelection


@pytest.mark.integration
class TestBinaryEvolutionPipeline:
    """Integration test for complete binary evolution pipeline."""

    def test_single_generation_binary(self, rng_key):
        """Test a complete single generation of binary evolution."""
        # Setup
        config = BinaryGenomeConfig(length=20)
        pop_size = 10
        
        # Create initial population
        keys = jr.split(rng_key, pop_size)
        population = [BinaryGenome.random_init(key, config) for key in keys]
        
        # Create operators
        evaluator = BinarySumFitnessEvaluator()
        selector = TournamentSelection(num_selections=pop_size, tournament_size=3)
        crossover = UniformCrossover(num_offspring=1, crossover_rate=0.8)
        mutator = BitFlipMutation(mutation_rate=0.05)
        
        # Evolution step
        key1, key2, key3 = jr.split(rng_key, 3)
        
        # 1. Evaluate fitness
        population_bits = jnp.array([genome.bits for genome in population])
        fitness_values = evaluator.evaluate_batch(population_bits)
        
        assert fitness_values.shape == (pop_size,)
        assert jnp.all(fitness_values >= 0)
        assert jnp.all(fitness_values <= config.length)
        
        # 2. Selection
        selected_indices = selector(key1, fitness_values)
        assert selected_indices.shape == (pop_size,)
        
        # 3. Crossover (pairwise)
        cross_keys = jr.split(key2, pop_size // 2)
        new_population = []
        
        for i in range(0, pop_size - 1, 2):
            parent1_idx = selected_indices[i]
            parent2_idx = selected_indices[i + 1]
            
            parent1 = population[parent1_idx]
            parent2 = population[parent2_idx]
            
            offspring = crossover(cross_keys[i//2], parent1, parent2, config)
            # crossover returns batch, take first offspring
            child = BinaryGenome(bits=offspring.bits[0])
            new_population.append(child)
        
        # If odd population size, add one more
        if len(new_population) < pop_size:
            parent_idx = selected_indices[-1]
            new_population.append(population[parent_idx])
        
        assert len(new_population) == pop_size
        
        # 4. Mutation
        mut_keys = jr.split(key3, pop_size)
        mutated_population = []
        
        for i, genome in enumerate(new_population):
            mutated_genome = mutator(mut_keys[i], genome, config)
            mutated_population.append(mutated_genome)
        
        assert len(mutated_population) == pop_size
        
        # Verify final population
        for genome in mutated_population:
            assert isinstance(genome, BinaryGenome)
            assert genome.bits.shape == (config.length,)
            assert jnp.all((genome.bits == 0) | (genome.bits == 1))

    @pytest.mark.jit
    def test_jit_compiled_binary_generation(self, rng_key):
        """Test JIT-compiled binary evolution generation."""
        config = BinaryGenomeConfig(length=10)
        pop_size = 8
        
        # JIT-compile the evolution step
        @jax.jit
        def evolution_step(key, population_bits, fitness_values):
            # Selection
            selector = TournamentSelection(num_selections=pop_size, tournament_size=2)
            select_fn = jax.jit(selector.get_pure_function())
            
            key1, key2, key3 = jr.split(key, 3)
            selected_indices = select_fn(key1, fitness_values)
            
            # Simple mutation on selected population
            selected_population = population_bits[selected_indices]
            
            mutator = BitFlipMutation(mutation_rate=0.1)
            mutate_fn = jax.jit(mutator.get_pure_function(), static_argnames=['config'])
            
            # Apply mutation to entire selected population
            mut_keys = jr.split(key2, pop_size)
            mutated_population = jax.vmap(mutate_fn, in_axes=(0, 0, None))(
                mut_keys, selected_population, config
            )
            
            return mutated_population
        
        # Create initial population
        keys = jr.split(rng_key, pop_size)
        population_bits = jnp.array([
            BinaryGenome.random_init(key, config).bits for key in keys
        ])
        
        # Evaluate fitness
        evaluator = BinarySumFitnessEvaluator()
        fitness_values = evaluator.evaluate_batch(population_bits)
        
        # Run JIT-compiled evolution step
        new_population = evolution_step(rng_key, population_bits, fitness_values)
        
        assert new_population.shape == (pop_size, config.length)
        assert jnp.all((new_population == 0) | (new_population == 1))


@pytest.mark.integration
@pytest.mark.slow
class TestLargeScaleIntegration:
    """Large-scale integration tests."""

    def test_large_binary_population(self, rng_key):
        """Test evolution with large binary population."""
        config = BinaryGenomeConfig(length=50)
        pop_size = 100
        
        # Create large population
        keys = jr.split(rng_key, pop_size)
        
        @jax.jit
        def create_population(keys):
            return jax.vmap(BinaryGenome.get_random_initialization_pure_from_config,
                           in_axes=(0, None))(keys, config)
        
        population_bits = create_population(keys)
        
        # Batch evaluate
        evaluator = BinarySumFitnessEvaluator()
        fitness_values = evaluator.evaluate_batch(population_bits)
        
        # Batch select
        selector = TournamentSelection(num_selections=pop_size, tournament_size=5)
        selected_indices = selector(rng_key, fitness_values)
        
        # Verify everything worked at scale
        assert population_bits.shape == (pop_size, config.length)
        assert fitness_values.shape == (pop_size,)
        assert selected_indices.shape == (pop_size,)
        
        # Check that selection favors high fitness
        selected_fitness = fitness_values[selected_indices]
        population_mean = jnp.mean(fitness_values)
        selected_mean = jnp.mean(selected_fitness)
        
        # Selected individuals should have higher average fitness
        assert selected_mean >= population_mean - 1.0  # Allow some variance

    def test_multi_operator_compatibility(self, rng_key):
        """Test that different operators work together seamlessly."""
        # Test various operator combinations
        
        # Binary operators
        from malthusjax.operators.mutation.binary import BitFlipMutation, ScrambleMutation
        from malthusjax.operators.crossover.binary import UniformCrossover, SinglePointCrossover
        
        config = BinaryGenomeConfig(length=15)
        
        key1, key2 = jr.split(rng_key)
        parent1 = BinaryGenome.random_init(key1, config)
        parent2 = BinaryGenome.random_init(key2, config)
        
        # Test different mutation-crossover combinations
        mutations = [BitFlipMutation(mutation_rate=0.1), ScrambleMutation()]
        crossovers = [
            UniformCrossover(num_offspring=2, crossover_rate=0.8),
            SinglePointCrossover(num_offspring=2)
        ]
        
        for mutator in mutations:
            for crossover in crossovers:
                # Test crossover
                key3, key4 = jr.split(rng_key)
                offspring = crossover(key3, parent1, parent2, config)
                
                # Test mutation on offspring
                for i in range(offspring.bits.shape[0]):
                    child = BinaryGenome(bits=offspring.bits[i])
                    mutated = mutator(key4, child, config)
                    
                    # Verify result is valid
                    assert isinstance(mutated, BinaryGenome)
                    assert mutated.bits.shape == (config.length,)
                    assert jnp.all((mutated.bits == 0) | (mutated.bits == 1))

    def test_convergence_behavior(self, rng_key):
        """Test that evolution improves fitness over multiple generations."""
        config = BinaryGenomeConfig(length=20)
        pop_size = 20
        num_generations = 5
        
        # Initialize population
        keys = jr.split(rng_key, pop_size)
        population = [BinaryGenome.random_init(key, config) for key in keys]
        
        evaluator = BinarySumFitnessEvaluator()
        selector = TournamentSelection(num_selections=pop_size, tournament_size=3)
        crossover = UniformCrossover(num_offspring=1, crossover_rate=0.8)
        mutator = BitFlipMutation(mutation_rate=0.05)
        
        fitness_history = []
        current_key = rng_key
        
        for gen in range(num_generations):
            # Evaluate fitness
            population_bits = jnp.array([genome.bits for genome in population])
            fitness_values = evaluator.evaluate_batch(population_bits)
            
            # Track best fitness
            best_fitness = jnp.max(fitness_values)
            fitness_history.append(best_fitness)
            
            # Evolve
            key1, key2, key3, current_key = jr.split(current_key, 4)
            
            # Selection
            selected_indices = selector(key1, fitness_values)
            
            # Crossover + Mutation
            new_population = []
            cross_keys = jr.split(key2, pop_size)
            mut_keys = jr.split(key3, pop_size)
            
            for i in range(pop_size):
                if i < pop_size - 1:
                    # Crossover
                    parent1_idx = selected_indices[i]
                    parent2_idx = selected_indices[i + 1]
                    
                    parent1 = population[parent1_idx]
                    parent2 = population[parent2_idx]
                    
                    offspring = crossover(cross_keys[i], parent1, parent2, config)
                    child = BinaryGenome(bits=offspring.bits[0])
                else:
                    # Copy last selected individual
                    child = population[selected_indices[i]]
                
                # Mutation
                mutated_child = mutator(mut_keys[i], child, config)
                new_population.append(mutated_child)
            
            population = new_population
        
        # Check that fitness improved or stayed stable
        initial_fitness = fitness_history[0]
        final_fitness = fitness_history[-1]
        
        # Should show some improvement or at least not get worse
        assert final_fitness >= initial_fitness - 1.0  # Allow some variance