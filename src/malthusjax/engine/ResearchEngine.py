"""
Research Engine for MalthusJAX.

A full-featured, introspectable engine that captures detailed intermediate results
from every step of the GA (selection, crossover, mutation) for analysis and callbacks.
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
class CallbackMetrics:
    """
    Holds accumulated metrics across generations.
    This is the minimal metrics state that gets carried forward.
    """
    
    selection_pressure: float
    """
    A metric measuring selection pressure.
    Could be variance in selection frequencies, tournament intensity, etc.
    """
    
    def update_selection_pressure(self, new_pressure: float) -> 'CallbackMetrics':
        """
        Update the selection pressure metric.
        
        Args:
            new_pressure: New selection pressure value
            
        Returns:
            Updated CallbackMetrics instance
        """
        return self.replace(selection_pressure=new_pressure)
    
    @staticmethod
    def empty() -> 'CallbackMetrics':
        """Create empty metrics state."""
        return CallbackMetrics(selection_pressure=0.0)


@flax.struct.dataclass
class FullIntermediateState:
    """
    Complete intermediate state capturing ALL GA phase results.
    Each phase stores its results and uses previous phase results.
    """
    
    # --- Phase 1: Selection ---
    selected_indices_1: jnp.ndarray
    selected_indices_2: jnp.ndarray
    selection_pressure: jnp.ndarray  # JAX array, not float
    
    # --- Phase 2: Crossover ---
    offspring_raw: jnp.ndarray  # Before mutation
    crossover_success_rate: jnp.ndarray  # JAX array, not float
    
    # --- Phase 3: Mutation ---
    offspring_final: jnp.ndarray  # After mutation  
    mutation_impact: jnp.ndarray  # JAX array, not float
    
    # --- Phase 4: Elitism ---
    elite_indices: jnp.ndarray
    new_population: jnp.ndarray
    
    # best fitness 
    best_fitness: jnp.ndarray
            
    def get_selected_parents(self, population: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get parent genomes from selection phase results."""
        return population[self.selected_indices_1], population[self.selected_indices_2]
    
    def get_elites(self, population: jnp.ndarray) -> jnp.ndarray:
        """Get elite genomes from elitism phase results."""
        return population[self.elite_indices]


@flax.struct.dataclass
class ResearchState(AbstractState):
    """
    Enhanced state for research use with callback metrics.
    
    This is the state that gets carried forward in the scan loop,
    containing both core GA data and research metrics.
    """
    
    # --- Callback Metrics (NEW) ---
    metrics: CallbackMetrics
    """The accumulated callback metrics."""
    
    @classmethod
    def from_abstract_state(cls, 
                          abstract_state: AbstractState, 
                          metrics: CallbackMetrics) -> 'ResearchState':
        """Create ResearchState from AbstractState."""
        return cls(
            population=abstract_state.population,
            fitness=abstract_state.fitness,
            best_genome=abstract_state.best_genome,
            best_fitness=abstract_state.best_fitness,
            key=abstract_state.key,
            generation=abstract_state.generation,
            metrics=metrics
        )


class ResearchEngine(AbstractEngine):
    """
    Full-featured research engine with complete introspection capabilities.
    
    This engine captures detailed intermediate results from every GA operation:
    - Selection indices and pressure metrics
    - Crossover success rates and offspring before/after mutation
    - Mutation impact measurements
    - Elite individual tracking
    - Complete pipeline traceability for research and debugging
    """
    
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation,
                 elitism: int):
        """Initialize ResearchEngine with GA components."""
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
                             initial_population: Optional[Array] = None) -> ResearchState:
        """Create the initial GA state with research capabilities."""
        
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
        
        # --- Initialize Metrics ---
        initial_metrics = CallbackMetrics.empty()
        
        return ResearchState(
            population=population_array,
            fitness=fitness_array,
            best_genome=best_genome,
            best_fitness=best_fitness,
            key=key,
            generation=0,
            metrics=initial_metrics
        )
    
    def _research_step_fn(self,
                         state: ResearchState,
                         _: None, # unused scan input
                         fitness_fn: Callable,
                         selection_fn: Callable,
                         crossover_fn: Callable,
                         mutation_fn: Callable,
                         pop_size: int,
                         elitism: int) -> Tuple[ResearchState, FullIntermediateState]:
        """
        JIT-compatible research GA step with complete pipeline capture.
        
        This is the productionized version of ga_step_fn_full_pipeline
        from the L3_callbacks_Scratchpad.ipynb notebook.
        """
        
        key, sel_key1, sel_key2, cross_key, mut_key = jar.split(state.key, 5)
        
        # --- Phase 0: Elitism (Static Shape) ---
        sorted_indices = jnp.argsort(-state.fitness)  
        elite_indices = sorted_indices[:elitism]
        elite_individuals = state.population[elite_indices]
        
        # --- Phase 1: Selection (Fixed Number of Offspring) ---
        num_offspring = pop_size - elitism  # Static calculation
        selected_indices_1 = selection_fn(sel_key1, state.fitness)[:num_offspring]
        selected_indices_2 = selection_fn(sel_key2, state.fitness)[:num_offspring]
        
        # Calculate selection pressure from stored indices
        selection_counts_1 = jnp.bincount(selected_indices_1, length=pop_size)
        selection_counts_2 = jnp.bincount(selected_indices_2, length=pop_size)
        selection_counts = selection_counts_1 + selection_counts_2
        selection_pressure = jnp.var(selection_counts)
        
        # --- Phase 2: Crossover (Static Shapes) ---
        parent_1 = state.population[selected_indices_1]  
        parent_2 = state.population[selected_indices_2]  
        
        # Fixed number of crossover operations
        crossover_keys = jar.split(cross_key, num_offspring)
        # get dimension of parents:
        #print(f"parents 1 shape{parent_1.shape}")
        #print(f"parents 2 shape{parent_2.shape}")
        #print(f"keys shape{crossover_keys.shape}")

        
        crossover_fn_batched = jax.vmap(crossover_fn, in_axes=(0, 0, 0))
        offspring_raw = crossover_fn_batched(crossover_keys, parent_1, parent_2)
        
        # Handle crossover output shape (ensure it's (num_offspring, genome_shape))
        if len(offspring_raw.shape) > 2:  # If crossover returns (n, 1, genome_shape)
            offspring_raw = offspring_raw[:, 0, :]  # Take first offspring from each pair
        
        # Calculate crossover success rate
        parent_avg = (parent_1.astype(jnp.float32) + parent_2.astype(jnp.float32)) / 2
        offspring_float = offspring_raw.astype(jnp.float32)
        crossover_success_rate = jnp.mean(jnp.abs(offspring_float - parent_avg))
        
        # --- Phase 3: Mutation (Fixed Shapes) ---
        mutation_keys = jar.split(mut_key, num_offspring)  # Static number
        mutation_fn_batched = jax.vmap(mutation_fn, in_axes=(0, 0))
        offspring_final = mutation_fn_batched(mutation_keys, offspring_raw)
        
        # Calculate mutation impact
        mutation_impact = jnp.mean(jnp.not_equal(offspring_final, offspring_raw).astype(jnp.float32))
        
        # --- Phase 4: Population Assembly (Static Shape) ---
        # Create new population with known shape: (pop_size, genome_shape)
        new_population = jnp.zeros_like(state.population)  # Start with zeros
        new_population = new_population.at[:elitism].set(elite_individuals)  # Set elites
        new_population = new_population.at[elitism:elitism+num_offspring].set(offspring_final)  # Set offspring
        
        new_fitness = jax.vmap(fitness_fn)(new_population)
        
        # --- Build Complete Intermediate State ---
        full_intermediate = FullIntermediateState(
            selected_indices_1=selected_indices_1,
            selected_indices_2=selected_indices_2, 
            selection_pressure=selection_pressure,
            offspring_raw=offspring_raw,
            crossover_success_rate=crossover_success_rate,
            offspring_final=offspring_final,
            mutation_impact=mutation_impact,
            elite_indices=elite_indices,
            new_population=new_population,
            best_fitness=state.fitness[elite_indices[0]]
        )
        
        # --- Update State ---
        updated_metrics = state.metrics.update_selection_pressure(selection_pressure)
        new_state = ResearchState(
            population=new_population,
            fitness=new_fitness,
            best_genome=elite_individuals[0],
            best_fitness=state.fitness[elite_indices[0]],
            key=key,
            generation=state.generation + 1,
            metrics=updated_metrics
        )
        
        return new_state, full_intermediate
    
    def run(self, 
           key: jnp.ndarray,
           num_generations: int, 
           pop_size: int,
           initial_population: Optional[Array] = None) -> Tuple[ResearchState, FullIntermediateState]:
        """
        Run the evolutionary algorithm with full research capabilities.
        
        Args:
            key: JAX random key
            num_generations: Number of generations to run
            pop_size: Population size
            initial_population: Optional initial population
            
        Returns:
            Tuple of (final_state, all_intermediates)
            - final_state: The final ResearchState
            - all_intermediates: FullIntermediateState Pytree containing complete
              intermediate results for every generation
        """
        
        # Create initial state
        init_key, scan_key = jar.split(key)
        initial_state = self._create_initial_state(init_key, pop_size, initial_population)
        initial_state = initial_state.replace(key=scan_key)
        
        # Create JIT-compiled step function
        step_fn = functools.partial(
            self._research_step_fn,
            fitness_fn=self._fitness_fn,
            selection_fn=self._selection_fn,
            crossover_fn=self._crossover_fn,
            mutation_fn=self._mutation_fn,
            pop_size=pop_size,
            elitism=self._elitism
        )
        
        jit_step_fn = jax.jit(step_fn)
        #jit_step_fn = step_fn  # Disable JIT for debugging
        # Run evolution using scan - collects ALL intermediate states
        final_state, all_intermediates = jax.lax.scan(
            jit_step_fn,
            initial_state,
            None,
            length=num_generations
        )
        
        return final_state, all_intermediates