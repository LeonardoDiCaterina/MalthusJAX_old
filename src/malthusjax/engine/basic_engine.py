"""
Level 3 Engine Architecture - Abstract Base Classes

This module defines the core abstractions that all Level 3 engines must follow.
Provides type safety, JIT compatibility, and universal visualization support.
"""

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import flax.struct # type: ignore
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Any, List
import functools
from malthusjax.core.genome import AbstractGenome
from malthusjax.core.fitness import AbstractFitnessEvaluator
from malthusjax.operators.mutation import AbstractMutation
from malthusjax.operators.crossover import AbstractCrossover
from malthusjax.operators.selection import AbstractSelectionOperator
from malthusjax.engine.base import AbstractEngineParams, AbstractEvolutionState, AbstractGenerationOutput, AbstractEngine


@flax.struct.dataclass  
class GeneticEngineParams(AbstractEngineParams):
    """Genetic algorithm specific parameters."""
    mutation_rate: float = flax.struct.field(pytree_node=False, default=0.01)
    crossover_rate: float = flax.struct.field(pytree_node=False, default=0.8)
    tournament_size: int = flax.struct.field(pytree_node=False, default=3)

@flax.struct.dataclass  
class GeneticEvolutionState(AbstractEvolutionState):
    """Genetic algorithm evolution state."""
    current_population: jnp.ndarray
    current_fitness: jnp.ndarray
    best_genome: jnp.ndarray
    ema_delta_fitness: jnp.ndarray


@flax.struct.dataclass  
class GeneticGenerationOutput(AbstractGenerationOutput):
    """Genetic algorithm generation KPIs."""
    std_fitness: jnp.ndarray
    min_fitness: jnp.ndarray
    best_genome: jnp.ndarray
    ema_delta_fitness: jnp.ndarray


class GeneticEngine(AbstractEngine):
    """Genetic Algorithm engine with modular operator composition."""
    
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation):
        
        # Store components
        self.genome = genome_representation
        self.fitness = fitness_evaluator
        self.selection = selection_operator
        self.crossover = crossover_operator
        self.mutation = mutation_operator
        
        # Extract pure functions following Level 2 pattern
        self.init_genome_fn = self.genome.get_random_initialization_pure()
        self.evaluate_fn = self.fitness.get_pure_fitness_function()
        self.select_fn = self.selection.get_pure_function()
        self.crossover_fn = self.crossover.get_pure_function()
        self.mutation_fn = self.mutation.get_pure_function()
        
        AbstractEngine.__init__(self)
    
    def init_state(self, rng_key: jnp.ndarray, params: GeneticEngineParams) -> GeneticEvolutionState:
        """Initialize population and evaluation state."""
        init_key, state_key = jar.split(rng_key)
        
        # Generate initial population
        pop_keys = jar.split(init_key, params.pop_size)
        initial_population = jax.vmap(self.init_genome_fn)(pop_keys)
        
        # Evaluate initial fitness
        initial_fitness = jax.vmap(self.evaluate_fn)(initial_population)
        
        # Find best solution
        best_idx = jnp.argmax(initial_fitness)
        best_fitness = initial_fitness[best_idx]
        best_genome = initial_population[best_idx]
        
        return GeneticEvolutionState(
            generation=jnp.array(0, dtype=jnp.int32),
            stagnation_counter=jnp.array(0, dtype=jnp.int32),
            best_fitness=best_fitness,
            rng_key=state_key,
            current_population=initial_population,
            current_fitness=initial_fitness,
            best_genome=best_genome,
            ema_delta_fitness=jnp.array(0.0, dtype=jnp.float32)
        )
    
    def step(self, key: jnp.ndarray, state: GeneticEvolutionState,
             params: GeneticEngineParams) -> Tuple[jnp.ndarray, GeneticEvolutionState, GeneticGenerationOutput]:
        """Execute one generation: selection → crossover → mutation → evaluation."""
        
        # Split random keys
        sel_key, cross_key, mut_key = jar.split(key, 3)
        
        # 1. Elitism - preserve best individuals
        sorted_indices = jnp.argsort(-state.current_fitness)
        elite_indices = sorted_indices[:params.elitism]
        elite_individuals = state.current_population[elite_indices]
        
        # 2. Selection for breeding
        num_offspring = params.pop_size - params.elitism
        selected_indices_1 = self.select_fn(sel_key, state.current_fitness)[:num_offspring]
        selected_indices_2 = self.select_fn(sel_key, state.current_fitness)[:num_offspring]
        parent_1 = state.current_population[selected_indices_1]
        parent_2 = state.current_population[selected_indices_2]
        
        # 3. Crossover
        crossover_keys = jar.split(cross_key, num_offspring)
        offspring = jax.vmap(self.crossover_fn, in_axes=(0,0,0))(crossover_keys, parent_1, parent_2)
        
        # Handle crossover output shape (take first offspring if multiple)
        if len(offspring.shape) > 2:
            offspring = offspring[:, 0, :]
        
        # 4. Mutation
        mutation_keys = jar.split(mut_key, num_offspring)
        mutated_offspring = jax.vmap(self.mutation_fn, in_axes=(0,0))(mutation_keys, offspring)
        
        # 5. Combine elite + offspring
        new_population = jnp.vstack([elite_individuals, mutated_offspring])
        
        # 6. Evaluate new population
        new_fitness = jax.vmap(self.evaluate_fn)(new_population)
        
        # 7. Update metrics
        current_best_idx = jnp.argmax(new_fitness)
        current_best_fitness = new_fitness[current_best_idx]
        current_best_genome = new_population[current_best_idx]
        
        # Fitness improvement tracking
        delta_fitness = current_best_fitness - state.best_fitness
        is_stagnant = (delta_fitness <= 1e-6)
        new_stagnation_counter = jnp.where(is_stagnant, state.stagnation_counter + 1, 0)
        
        # Exponential moving average of fitness change
        ema_alpha = 0.1
        new_ema_delta = (ema_alpha * delta_fitness + 
                        (1 - ema_alpha) * state.ema_delta_fitness)
        
        # 8. Construct new state
        new_state = GeneticEvolutionState(
            generation=state.generation + 1,
            stagnation_counter=new_stagnation_counter,
            best_fitness=jnp.maximum(current_best_fitness, state.best_fitness),
            rng_key=key,
            current_population=new_population,
            current_fitness=new_fitness,
            best_genome=jnp.where(current_best_fitness > state.best_fitness,
                                  current_best_genome, state.best_genome),
            ema_delta_fitness=new_ema_delta
        )
        
        # 9. Prepare KPI output
        stats = GeneticGenerationOutput(
            best_fitness=new_state.best_fitness,
            mean_fitness=jnp.mean(new_fitness),
            std_fitness=jnp.std(new_fitness),
            min_fitness=jnp.min(new_fitness),
            generation=new_state.generation,
            best_genome=new_state.best_genome,
            ema_delta_fitness=new_ema_delta
        )
        
        return key, new_state, stats
