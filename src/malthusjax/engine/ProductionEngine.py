"""
Production Engine for MalthusJAX.

A lean, high-performance engine that discards intermediate generational data
to save memory while providing maximum performance.
"""

from typing import Callable, Dict, Tuple, Optional
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from jax.random import PRNGKey # type: ignore
from jax import Array # type: ignore
import flax.struct # type: ignore
import functools

from malthusjax.core.genome import AbstractGenome
from malthusjax.core.fitness import AbstractFitnessEvaluator
from malthusjax.operators.selection import AbstractSelectionOperator
from malthusjax.operators.crossover import AbstractCrossover
from malthusjax.operators.mutation import AbstractMutation
from malthusjax.engine.base import AbstractEngine, AbstractState


@flax.struct.dataclass
class ProductionState(AbstractState):
    """
    Lean state for production use.
    
    Functionally identical to the original MalthusState but inherits
    from AbstractState for type compatibility.
    """
    pass


class ProductionEngine(AbstractEngine):
    """
    Lean, high-performance engine for production use.
    
    This engine focuses on speed and memory efficiency:
    - Uses simple JIT-compiled step function
    - Collects minimal history (only best fitness per generation)
    - Discards intermediate generational data
    - Optimized for deployment scenarios
    """
    
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation,
                 elitism: int):
        """Initialize ProductionEngine with GA components."""
        super().__init__(
            genome_representation=genome_representation,
            fitness_evaluator=fitness_evaluator,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            elitism=elitism
        )
    
    def _create_initial_state(self, 
                             key: PRNGKey, 
                             pop_size: int,
                             initial_population: Optional[Array] = None) -> ProductionState:
        """Create the initial GA state."""
        
        # --- Population Initialization ---
        if initial_population is None:
            pop_keys = jar.split(key, pop_size)
            population_array = jax.vmap(self._init_fn)(pop_keys)
            key, _ = jar.split(key)
        else:
            population_array = initial_population
            
        # --- Fitness Evaluation ---
        fitness_array = jax.vmap(self._fitness_fn)(population_array)
        
        # --- Best Genome & Fitness ---
        best_index = jnp.argmax(fitness_array)
        best_genome = population_array[best_index]
        best_fitness = fitness_array[best_index]
        
        return ProductionState(
            population=population_array,
            fitness=fitness_array,
            best_genome=best_genome,
            best_fitness=best_fitness,
            key=key,
            generation=0
        )
    
    def _production_step_fn(self,
                          state: ProductionState,
                          _: None,
                          fitness_fn: Callable,
                          selection_fn: Callable,
                          crossover_fn: Callable,
                          mutation_fn: Callable,
                          pop_size: int,
                          elitism: int) -> Tuple[ProductionState, float]:
        """
        Lean GA step function optimized for performance.
        
        Returns minimal metrics (only best fitness) to save memory.
        """
        
        # Split key for this generation
        key, selection_key_1, selection_key_2, crossover_key, mutation_key = jar.split(state.key, 5)
        
        # --- Elitism ---
        sorted_indices = jnp.argsort(-state.fitness)  # Sort descending
        elite_indices = sorted_indices[:elitism]
        elite_individuals = state.population[elite_indices]
        best_genome = elite_individuals[0]
        best_fitness = state.fitness[elite_indices[0]]
        
        # --- Selection ---
        num_offspring = pop_size - elitism
        selected_indices_1 = selection_fn(selection_key_1, state.fitness)[:num_offspring]
        selected_indices_2 = selection_fn(selection_key_2, state.fitness)[:num_offspring]
        
        # --- Crossover ---
        parent_1 = state.population[selected_indices_1]
        parent_2 = state.population[selected_indices_2]
        crossover_keys = jar.split(crossover_key, num_offspring)
        crossover_fn_batched = jax.vmap(crossover_fn, in_axes=(0, 0, 0))
        offspring = crossover_fn_batched(crossover_keys, parent_1, parent_2)
        
        # Handle crossover output shape
        if len(offspring.shape) > 2:
            offspring = offspring[:, 0, :]  # Take first offspring from each pair
        
        # --- Mutation ---
        mutation_keys = jar.split(mutation_key, num_offspring)
        mutation_fn_batched = jax.vmap(mutation_fn, in_axes=(0, 0))
        mutated_offspring = mutation_fn_batched(mutation_keys, offspring)
        
        # --- Create New Population ---
        new_population = jnp.zeros_like(state.population)
        new_population = new_population.at[:elitism].set(elite_individuals)
        new_population = new_population.at[elitism:elitism+num_offspring].set(mutated_offspring)
        
        new_fitness = jax.vmap(fitness_fn)(new_population)
        
        # --- Create New State ---
        new_state = ProductionState(
            population=new_population,
            fitness=new_fitness,
            best_genome=best_genome,
            best_fitness=best_fitness,
            key=key,
            generation=state.generation + 1
        )
        
        return new_state, best_fitness
    
    def run(self, 
           key: PRNGKey, 
           num_generations: int, 
           pop_size: int,
           initial_population: Optional[Array] = None) -> Tuple[ProductionState, jnp.ndarray]:
        """
        Run the evolutionary algorithm with lean production settings.
        
        Args:
            key: JAX random key
            num_generations: Number of generations to run
            pop_size: Population size
            initial_population: Optional initial population
            
        Returns:
            Tuple of (final_state, fitness_history)
            - final_state: The final ProductionState
            - fitness_history: Array of best fitness values per generation
        """
        
        # Create initial state
        init_key, scan_key = jar.split(key)
        initial_state = self._create_initial_state(init_key, pop_size, initial_population)
        initial_state = initial_state.replace(key=scan_key)
        
        # Create JIT-compiled step function
        step_fn = functools.partial(
            self._production_step_fn,
            fitness_fn=self._fitness_fn,
            selection_fn=self._selection_fn,
            crossover_fn=self._crossover_fn,
            mutation_fn=self._mutation_fn,
            pop_size=pop_size,
            elitism=self._elitism
        )
        
        jit_step_fn = jax.jit(step_fn)
        
        # Run evolution using scan
        final_state, fitness_history = jax.lax.scan(
            jit_step_fn,
            initial_state,
            None,
            length=num_generations
        )
        
        return final_state, fitness_history