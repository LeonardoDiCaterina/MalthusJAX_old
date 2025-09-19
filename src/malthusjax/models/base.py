 # Path: src/malthusjax/models/base.py

"""Base classes for genetic algorithm models.

This module provides the foundation for building evolutionary computation
models with a Keras-like API, including compilation, fitting, and callback support.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Callable, Tuple, Union
import time
import jax # type: ignore
import jax.numpy as jnp # type: ignore
from jax.random import PRNGKey, split # type: ignore

from malthusjax.core.base import Compatibility
from malthusjax.core.population.base import AbstractPopulation
from malthusjax.core.solution.base import AbstractSolution
from malthusjax.core.fitness.base import AbstractFitnessEvaluator


class AbstractCallback:
    """Abstract base class for callbacks during evolutionary computation.
    
    Callbacks can be used to monitor, visualize, or modify the evolution
    process during training.
    """
    
    def on_evolution_begin(self, model: 'AbstractGeneticModel') -> None:
        """Called at the beginning of evolution."""
        pass
        
    def on_evolution_end(self, model: 'AbstractGeneticModel') -> None:
        """Called at the end of evolution."""
        pass
        
    def on_generation_begin(self, generation: int, model: 'AbstractGeneticModel') -> None:
        """Called at the beginning of each generation."""
        pass
        
    def on_generation_end(self, generation: int, model: 'AbstractGeneticModel') -> None:
        """Called at the end of each generation."""
        pass


class EvolutionHistory:
    """Tracks and stores statistics during the evolution process."""
    
    def __init__(self) -> None:
        """Initialize an empty history object."""
        self.history: Dict[str, List[Any]] = {
            'max_fitness': [],
            'min_fitness': [],
            'avg_fitness': [],
            'fitness_std': [],
            'best_solution': [],
            'generation_time': [],
        }
        self.metrics: Dict[str, List[Any]] = {}
        self.total_generations: int = 0
        self.total_time: float = 0.0
        
    def record_generation(self, 
                          stats: Dict[str, Any], 
                          best_solution: Optional[AbstractSolution] = None,
                          generation_time: float = 0.0) -> None:
        """Record statistics for a single generation.
        
        Args:
            stats: Dictionary of population statistics.
            best_solution: Current best solution.
            generation_time: Time taken for this generation in seconds.
        """
        self.history['max_fitness'].append(stats.get('max_fitness'))
        self.history['min_fitness'].append(stats.get('min_fitness'))
        self.history['avg_fitness'].append(stats.get('avg_fitness'))
        self.history['fitness_std'].append(stats.get('fitness_std'))
        self.history['best_solution'].append(best_solution)
        self.history['generation_time'].append(generation_time)
        self.total_generations += 1
        self.total_time += generation_time
        
    def record_metric(self, name: str, value: Any) -> None:
        """Record a custom metric value.
        
        Args:
            name: Name of the metric.
            value: Value of the metric.
        """
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)
        
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evolution history.
        
        Returns:
            Dictionary with summary statistics.
        """
        return {
            'total_generations': self.total_generations,
            'total_time': self.total_time,
            'best_fitness': max(self.history['max_fitness']) if self.history['max_fitness'] else None,
            'final_avg_fitness': self.history['avg_fitness'][-1] if self.history['avg_fitness'] else None,
            'best_solution': self.get_best_solution(),
            'avg_generation_time': (self.total_time / self.total_generations 
                                   if self.total_generations > 0 else 0),
        }
    
    def record_stop(self, reason: str) -> None:
        """Record the reason for stopping evolution.
        
        Args:
            reason: Reason for stopping (e.g., 'max_generations', 'convergence').
        """
        self.record_metric('stop_reason', reason)
    
    def get_best_solution(self) -> Optional[AbstractSolution]:
        """Get the best solution found during evolution.
        
        Returns:
            The solution with highest fitness, or None if no solutions recorded.
        """
        if not self.history['best_solution']:
            return None
            
        return max(self.history['best_solution'], 
                   key=lambda x: x.fitness if x is not None else float('-inf'))
    
    def reset(self) -> None:
        """Reset the history to an empty state."""
        self.__init__()  # Re-initialize to clear all data


class AbstractGeneticModel(ABC):
    """Abstract base class for genetic algorithm models.
    
    This class defines the core interface for all genetic models,
    providing a Keras-like API for compilation and fitting.
    """
    
    def __init__(self, name: Optional[str] = None, genome_init_params: Dict = {}) -> None:
        """Initialize the genetic model.
        
        Args:
            name: Optional name for the model.
        """
        self.name = name or self.__class__.__name__
        self.genome_init_params = genome_init_params
        self.compiled = False
        self.history = EvolutionHistory()
        self.fitness_evaluator: Optional[AbstractFitnessEvaluator] = None
        self.population: Optional[AbstractPopulation] = None
        self.random_key: Optional[jax.Array] = None
        self.callbacks: List[AbstractCallback] = []
        self._metadata: Dict[str, Any] = {
            'stop_evolution': False,
            'stop_reason': None,
            'log_interval': 1,  # Default log every generation
        }
        
    @abstractmethod
    def build(self, 
              solution_class: Type[AbstractSolution], 
              initial_population: Optional[AbstractPopulation] = None,
              **kwargs) -> None:
        """Build the model architecture.
        
        Args:
            solution_class: Class to use for solutions in the population.
            initial_population: Optional initial population to start with.
            **kwargs: Additional keyword arguments for specific models.
            
        Raises:
            ValueError: If model architecture cannot be built with given parameters.
        """
        pass
    
    def compile(self, 
                fitness_evaluator: AbstractFitnessEvaluator,
                random_key: Optional[Union[int, jax.Array]] = None) -> None:
        """Configure the model for evolution.
        
        Args:
            fitness_evaluator: Function to evaluate solution fitness.
            random_key: Optional JAX PRNG key for reproducibility or an integer seed.
            
        Raises:
            ValueError: If fitness_evaluator is incompatible with model.
        """
        # Convert integer seed to PRNG key if needed
        if random_key is None:
            random_key = jax.random.PRNGKey(int(time.time()))
        elif isinstance(random_key, int):
            random_key = jax.random.PRNGKey(random_key)
            
        self.random_key = random_key
        self.fitness_evaluator = fitness_evaluator
        
        # Check if the model is built
        if self.population is None:
            raise ValueError("Model must be built before compilation. Call build() first.")
        
        # Check compatibility between fitness_evaluator and population
        if hasattr(self.fitness_evaluator, "compatibility") and hasattr(self.population, "compatibility"):
            if not self.fitness_evaluator.compatibility.is_compatible(self.population.compatibility):
                raise ValueError(f"Fitness evaluator {fitness_evaluator.__class__.__name__} "
                               f"is not compatible with population in {self.name}.")
        
        # Reset history
        self.history.reset()
        self.compiled = True
    
    def add_callback(self, callback: AbstractCallback) -> None:
        """Add a callback to the model.
        
        Args:
            callback: Callback object to add.
        """
        self.callbacks.append(callback)
    
    def _call_callbacks(self, method_name: str, *args, **kwargs) -> None:
        """Call the specified method on all callbacks.
        
        Args:
            method_name: Name of the callback method to call.
            *args, **kwargs: Arguments to pass to the callback method.
        """
        for callback in self.callbacks:
            method = getattr(callback, method_name, None)
            if method:
                method(*args, **kwargs)
    
    @abstractmethod
    def fit(self,
            generations: int = 100, 
            callbacks: Optional[List[AbstractCallback]] = None,
            verbose: int = 1) -> EvolutionHistory:
        """Run the evolutionary process.
        
        Args:
            generations: Number of generations to evolve.
            callbacks: Optional list of callbacks to use during evolution.
            verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
            
        Returns:
            History object containing evolution statistics.
            
        Raises:
            ValueError: If model is not compiled before fitting.
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Use the best solution to make predictions.
        
        This method should be implemented by specific models to use
        the best solution for inference on new inputs.
        
        Args:
            inputs: Input data for prediction.
            
        Returns:
            Predictions from the best solution.
            
        Raises:
            ValueError: If model hasn't been fit yet.
        """
        pass
    
    @abstractmethod
    def get_best_solution(self) -> Optional[AbstractSolution]:
        """Get the best solution found by the model.
        
        Returns:
            Best solution found, or None if not available.
        """
        pass
    
    def summary(self, print_fn: Callable[[str], None] = print) -> None:
        """Print a summary of the model architecture.
        
        Args:
            print_fn: Function to use for printing.
        """
        print_fn(f"Model: {self.name}")
        print_fn("=" * (len(self.name) + 7))
        
        if not hasattr(self, "population") or self.population is None:
            print_fn("(Model not built)")
            return
            
        print_fn(f"Population size: {self.population.size}")
        print_fn(f"Solution type: {self.population.solution_class.__name__}")
        
        if hasattr(self, "operators") and getattr(self, "operators", None):
            print_fn("\nOperators:")
            for i, op in enumerate(self.operators):
                print_fn(f" {i+1}: {op.__class__.__name__}")
        
        if self.compiled:
            print_fn("\nCompiled: Yes")
            print_fn(f"Fitness evaluator: {self.fitness_evaluator.__class__.__name__}")
        else:
            print_fn("\nCompiled: No")
            
        if self.history and self.history.total_generations > 0:
            summary = self.history.get_summary()
            print_fn("\nTraining:")
            print_fn(f"  Generations: {summary['total_generations']}")
            print_fn(f"  Best fitness: {summary['best_fitness']}")
            print_fn(f"  Total time: {summary['total_time']:.2f}s")