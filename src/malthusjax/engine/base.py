from typing import Callable, Dict, Optional, Tuple
from abc import ABC, abstractmethod
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import functools
from malthusjax.core.genome import AbstractGenome
from malthusjax.core.fitness import AbstractFitnessEvaluator
from malthusjax.operators.selection import AbstractSelectionOperator
from malthusjax.operators.crossover import AbstractCrossover
from malthusjax.operators.mutation import AbstractMutation

from malthusjax.engine.state import MalthusState

class AbstractMalthusEngine(ABC):
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_op: AbstractSelectionOperator,
                 crossover_op: AbstractCrossover,
                 mutation_op: AbstractMutation,
                 pop_size: int,
                 elitism: int = 1,
                 ): 
        
        self.pop_size = pop_size
        self.elitism = elitism
        
        # 1. Get all the pure JAX functions from our factories
        self.init_fn = genome_representation.get_random_initialization_pure()
        self.fitness_fn = fitness_evaluator.get_tensor_fitness_function()
        self.selection_fn = selection_op.get_pure_function()
        self.crossover_fn = crossover_op.get_pure_function()
        self.mutation_fn = mutation_op.get_pure_function()
        self._compiled_run_loop = None

        # This cache will store compiled `scan` functions,
        # mapping pop_size -> compiled_fn
        self._compiled_cache: Dict[int, Callable] = {}


    def _determine_pop_size(self,
                           initial_population: Optional[jax.Array],
                           pop_size: Optional[int]) -> int:
        """
        Helper function to determine the static pop_size for a run.
        
        It checks for a provided population first, then falls back
        to the user-provided pop_size.
        """
        if initial_population is not None:
            return initial_population.shape[0]
        elif pop_size is not None:
            return pop_size
        else:
            raise ValueError(
                "Engine must be given a population size. "
                "Pass either `pop_size` (int) or an `initial_population` (Array) to the run() method."
            )
            
        

    # --- Abstract Methods  ---
    
    @abstractmethod
    def _get_step_fn(self) -> Callable:
        """
        Hook for the concrete engine to return its pure `ga_step_fn`.
        """
        pass    
    
    @abstractmethod
    def _get_static_args(self, pop_size: int) -> Dict:
        """
        Hook for the concrete engine to gather all static arguments
        (pure functions and static config) to be "baked" into the
        step function.
        """
        pass
    
    
    @abstractmethod
    def _get_initial_state(self,
                           key: jax.Array,
                           pop_size: int,
                           population_array: Optional[jax.Array],
                           fitness_array: Optional[jax.Array]
                           ) -> MalthusState:
        """
        Hook for the concrete engine to create the Generation 0 state.
        
        If population_array is None, it must be randomly initialized.
        If fitness_array is None, it must be calculated.
        """
        pass    
    
    # --- Concrete Methods (We will implement these next) ---
    
    def _compile_for_pop_size(self, pop_size: int):
        """
        Triggers the one-time JIT compilation for a specific pop_size
        and caches the result.
        """
        step_fn = self._get_step_fn()
        static_args = self._get_static_args(pop_size)
        
        partial_step_fn = functools.partial(step_fn, **static_args)

        def scan_fn(init, xs, length):
            return jax.lax.scan(partial_step_fn, init, xs, length=length)

        compiled_scan_fn = scan_fn
        
        self._compiled_cache[pop_size] = compiled_scan_fn
    
    def run(self,
            key: jax.Array,
            num_generations: int,
            initial_population: Optional[jax.Array] = None,
            initial_fitness: Optional[jax.Array] = None,
            pop_size: Optional[int] = None
            ) -> Tuple[MalthusState, Dict[str, jax.Array]]:
        """
        Runs the full evolutionary loop for a set number of generations.

        This function implements the "just-in-time" compilation cache.
        It will compile a new `scan` loop for each unique population
        size and cache it.

        Args:
            key: The top-level JAX PRNGKey for the entire run.
            num_generations: The number of generations to run.
            initial_population: (Optional) A JAX array of genomes.
                If provided, `pop_size` is inferred from this.
            initial_fitness: (Optional) A JAX array of fitnesses.
                If not provided, it will be calculated.
            pop_size: (Optional) The population size.
                Only required if `initial_population` is not provided.

        Returns:
            A tuple of (final_state, history_metrics):
            - `final_state`: The MalthusState after the final generation.
            - `history_metrics`: A Pytree (dictionary) where each leaf
              is an array of shape (num_generations, ...) containing
              the metrics from each step.
        """
        # 1. Determine static population size
        actual_pop_size = self._determine_pop_size(initial_population, pop_size)
        
        # 2. Check cache and compile if needed
        if actual_pop_size not in self._compiled_cache:
            self._compile_for_pop_size(actual_pop_size)
            
        compiled_scan_fn = self._compiled_cache[actual_pop_size]
        
        key, init_key = jar.split(key)
        
        initial_state = self._get_initial_state(
            key=init_key,
            pop_size=actual_pop_size,
            population_array=initial_population,
            fitness_array=initial_fitness
        )

        # Pass the *other* key into the state for the scan loop
        initial_state = initial_state.replace(key=key)
        single_step_fn = self._get_step_fn()
        static_args = self._get_static_args(actual_pop_size)
        
        single_step_fn = functools.partial(single_step_fn, **static_args)        
        first_generation_state, _ = single_step_fn(initial_state, None)
        # start timing
        #start_time = jax.default_timer() 
        # 3. Run the compiled scan loop
        final_state, history_metrics = compiled_scan_fn(
            first_generation_state,
            xs= None, # No xs needed for GA loop
            length=num_generations
        )
        # end timing
        #end_time = jax.default_timer()
        #print(f"Total time for {num_generations} generations: {end_time - start_time} seconds")
        # Return results of the run
        return final_state, history_metrics
             