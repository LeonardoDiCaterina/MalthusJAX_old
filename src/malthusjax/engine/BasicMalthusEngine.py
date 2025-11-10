
"""

A concrete implementation of the AbstractMalthusEngine.

"""
from typing import Callable, Dict, Optional, Tuple
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from functools import partial


from malthusjax.core.genome import AbstractGenome
from malthusjax.core.fitness.base import AbstractFitnessEvaluator
from malthusjax.operators.selection.base import AbstractSelectionOperator
from malthusjax.operators.crossover.base import AbstractCrossover
from malthusjax.operators.mutation.base import AbstractMutation
from malthusjax.engine.base import AbstractMalthusEngine
from malthusjax.engine.state import MalthusState



class BasicMalthusEngine(AbstractMalthusEngine):
    """
    A concrete implementation of the AbstractMalthusEngine.

    This "Basic" engine implements the standard GA loop
    (selection, crossover, mutation, elitism, re-evaluation)
    using the pure functions provided.
    """

    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation,
                 elitism: int):
        """
        Initializes the BasicMalthusEngine with the required components.

        Args:
            genome_representation: The genome representation object.
            fitness_evaluator: The fitness evaluator object.
            selection_operator: The selection operator object.
            crossover_operator: The crossover operator object.
            mutation_operator: The mutation operator object.
            elitism: The number of elite individuals to retain.
        """
        self._genome_representation = genome_representation
        self._fitness_evaluator = fitness_evaluator
        self._selection_operator = selection_operator
        self._crossover_operator = crossover_operator
        self._mutation_operator = mutation_operator
        self._elitism = elitism

        # Call the parent class initializer
        super().__init__(
            genome_representation=genome_representation,
            fitness_evaluator=fitness_evaluator,
            selection_op=selection_operator,
            crossover_op=crossover_operator,
            mutation_op=mutation_operator,
            elitism=elitism,
            pop_size=10
        )

    def _get_step_fn(self) -> Callable:
        """
        Hook to return the pure, JIT-able GA step function.
        
        This implementation simply returns the `ga_step_fn`
        function we defined separately.
        """
        return ga_step_fn

    
    def _get_static_args(self, pop_size: int) -> Dict:
        """
        Hook to gather all static arguments (pure functions and 
        static config) to be "baked" into the step function.
        """
        # Calculate the number of offspring to generate
        num_offspring = pop_size - self.elitism
        
        # Calculate crossover pairs needed.
        # We use ceil to handle odd numbers of offspring.
        # e.g., 9 offspring -> ceil(4.5) = 5 pairs -> 10 offspring
        # We will simply truncate the extra offspring later.
        # This keeps the JAX shapes static and predictable.
        #num_crossover_pairs = jnp.ceil(num_offspring / 2).astype(jnp.int32)
        return {
            'fitness_fn': self._fitness_evaluator.get_tensor_fitness_function(),
            'selection_fn': self._selection_operator.get_pure_function(),
            'crossover_fn': self._crossover_operator.get_pure_function(),
            'mutation_fn': self._mutation_operator.get_pure_function(),
            'pop_size': pop_size,
            'elitism': self._elitism        }


    
    def _get_initial_state(self,
                           key: jax.Array,
                           pop_size: int,
                           population_array: Optional[jax.Array],
                           fitness_array: Optional[jax.Array]
                           ) -> MalthusState:
        """
        Hook to create the Generation 0 state.
        
        If population_array is None, it must be randomly initialized.
        If fitness_array is None, it must be calculated.
        """
        # --- Population Initialization ---
        if population_array is None:
            init_fn = self._genome_representation.get_random_initialization_pure()
            pop_keys = jar.split(key, pop_size)
            population_array = jax.vmap(
                init_fn,
                in_axes=0
            )(pop_keys)
            # Update the key after initialization
            key, _ = jar.split(key)
        # --- Fitness Evaluation ---
        if fitness_array is None:
            fitness_fn = self._fitness_evaluator.get_tensor_fitness_function()
            fitness_array = jax.vmap(
                fitness_fn,
                in_axes=0
            )(population_array)
        # --- Best Genome & Fitness ---
        best_index = jnp.argmax(fitness_array)
        best_genome = population_array[best_index]
        best_fitness = fitness_array[best_index]    
        
        return MalthusState(
            population=population_array,
            fitness=fitness_array,
            best_genome=best_genome,
            best_fitness=best_fitness,
            key=key,
            generation=0
        )
        
        
        
# ---- The GA Step Function ----
"""
Level 3: The GA Step Function (Loop Body)

This file defines the core `ga_step_fn`, which is the pure JAX function
that will be passed to `jax.lax.scan` to run the evolutionary loop.
"""

from malthusjax.engine.state import MalthusState
#from .state import MalthusState

# Define a standard type for the metrics we collect each generation
Metrics = Dict[str, jax.Array]

def ga_step_fn(
    state: MalthusState,
    _: None, # Placeholder for `xs` in jax.lax.scan
    
    # --- JIT-able functions passed in via functools.partial ---
    # These are our "factory-built" functions from Levels 1 & 2
    fitness_fn: Callable,
    selection_fn: Callable,
    crossover_fn: Callable,
    mutation_fn: Callable,
    
    # --- Static configuration values passed in via functools.partial ---
    pop_size: int,
    elitism: int    
) -> Tuple[MalthusState, Metrics]:
    """
    Executes one single generation of the Genetic Algorithm.
    
    This function is designed to be pure, JIT-compiled, and
    used inside `jax.lax.scan`.
    
    Args:
        state: The MalthusState from the previous generation (the "carry").
        _: Unused, for jax.lax.scan.
        fitness_fn: A pure function (genome) -> fitness.
        selection_fn: A pure function (key, fitnesses) -> indices.
        crossover_fn: A pure function (key, p1, p2) -> offspring_batch.
        mutation_fn: A pure function (key, genome) -> mutated_genome.
        pop_size: Static int for the total population size.
        elitism: Static int for the number of elite individuals to carry over.

    Returns:
        (new_state, metrics): A tuple containing the updated MalthusState
                              for the next generation and a dictionary
                              of metrics from this generation.
    """
    
    # --- Split the key for this generation ---
    key, selection_key_1, selection_key_2, crossover_key, mutation_key = jar.split(state.key, 5)
    
    
    # --- Elitism ---
    sorted_indices = jnp.argsort(state.fitness)  # Ensure axis is specified for sorting
    elite_indices = sorted_indices[:elitism]
    elite_individuals = state.population[elite_indices]
    elite_fitnesses = state.fitness[elite_indices]
    # best so-far
    best_genome = elite_individuals[0]
    best_fitness = elite_fitnesses[0]
    
    # --- Selection ---
    selected_indices_1 = selection_fn(selection_key_1, state.fitness )
    selected_indices_2 = selection_fn(selection_key_2, state.fitness )
    # -- Crossover ---
    parent_1 = state.population[selected_indices_1]
    parent_2 = state.population[selected_indices_2]
    crossover_fn_batched = jax.vmap(crossover_fn, in_axes=(0, 0, 0))
    crossover_keys = jar.split(crossover_key, parent_1.shape[0])
    offspring = crossover_fn_batched(crossover_keys,parent_1, parent_2)
    
    # --- Mutation ---
    mutation_keys = jar.split(mutation_key, offspring.shape[0])
    mutation_fn_batched = jax.vmap(mutation_fn, in_axes=(0, 0))
    mutated_offspring = mutation_fn_batched(mutation_keys, offspring )
    mutated_offspring = jnp.squeeze(mutated_offspring)

    # --- Create New Population ---
    new_population = jnp.vstack([elite_individuals, mutated_offspring])
    new_population = new_population[:pop_size]  # Ensure population size for now 
    new_fitness = jax.vmap(fitness_fn)(new_population)   # <-- was fitness_fn(new_population)
     
    new_state = state.replace(
        population=new_population,
        fitness=new_fitness,
        best_genome=best_genome,
        best_fitness=best_fitness,
        key=key,
        generation=state.generation + 1
    )
       
    # --- Collect Metrics ---
    # not yet implemented
    
    return new_state, best_fitness