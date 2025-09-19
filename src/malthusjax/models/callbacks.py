# Path: src/malthusjax/models/callbacks.py

"""Callbacks for monitoring and controlling genetic algorithm training.

This module provides callback implementations for common tasks like
early stopping, checkpointing, and logging during evolution.
"""

import os
import pickle
from typing import Optional, Dict, Any, List, Callable, Union
import time

from malthusjax.models.base import AbstractCallback, AbstractGeneticModel


class EarlyStopping(AbstractCallback):
    """Stop training when a monitored metric stops improving.
    
    Similar to Keras' EarlyStopping, this callback will halt evolution
    if the specified metric doesn't improve for a given number of generations.
    """
    
    def __init__(self, 
                patience: int = 10, 
                monitor: str = 'max_fitness',
                min_delta: float = 0.0,
                mode: str = 'max',
                baseline: Optional[float] = None,
                verbose: int = 0) -> None:
        """Initialize early stopping callback.
        
        Args:
            patience: Number of generations with no improvement to wait before stopping.
            monitor: Metric to monitor ('max_fitness', 'avg_fitness', etc).
            min_delta: Minimum change to qualify as an improvement.
            mode: One of 'max' or 'min'. For max, training stops when metric stops increasing.
            baseline: Baseline value for the monitored metric.
            verbose: Verbosity level (0=silent, 1=verbose).
        """
        super().__init__()
        self.patience = patience
        self.monitor = monitor
        self.min_delta = min_delta
        self.mode = mode.lower()
        self.baseline = baseline
        self.verbose = verbose
        
        self.stopped_generation = 0
        self.wait = 0
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.stopped = False
        
    def on_evolution_begin(self, model: AbstractGeneticModel) -> None:
        """Called at the beginning of evolution."""
        self.wait = 0
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.stopped = False
        self.stopped_generation = 0
        
    def on_generation_end(self, generation: int, model: AbstractGeneticModel) -> None:
        """Called at the end of each generation."""
        if self.stopped:
            return
            
        # Get current value from history or population stats
        current = None
        if model.history.history.get(self.monitor) and model.history.history[self.monitor]:
            current = model.history.history[self.monitor][-1]
        elif self.monitor in model.population.get_statistics():
            current = model.population.get_statistics()[self.monitor]
        
        if current is None:
            return
        
        # Check if improvement
        if self.mode == 'max':
            improved = current - self.best_value > self.min_delta
        else:  # mode == 'min'
            improved = self.best_value - current > self.min_delta
            
        if improved:
            self.best_value = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_generation = generation
                self.stopped = True
                model._metadata['stop_evolution'] = True
                if self.verbose > 0:
                    print(f"\nEarly stopping at generation {generation}. "
                          f"No improvement in {self.monitor} for {self.patience} generations.")


class ModelCheckpoint(AbstractCallback):
    """Save model after each generation.
    
    This callback saves the best solution or entire model to a file
    after specified generations or when improvement occurs.
    """
    
    def __init__(self, 
                filepath: str,
                monitor: str = 'max_fitness',
                verbose: int = 0,
                save_best_only: bool = True,
                save_weights_only: bool = False,
                mode: str = 'max',
                period: int = 1) -> None:
        """Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save the model file. Can contain format specifiers
                like {generation} or {max_fitness}.
            monitor: Quantity to monitor for determining "best".
            verbose: Verbosity level.
            save_best_only: If True, only save when monitor metric improves.
            save_weights_only: If True, save only best solution not entire model.
            mode: One of 'max' or 'min'. For 'max', checkpoint when monitored value increases.
            period: Interval between checkpoints in generations.
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode.lower()
        self.period = period
        
        self.best_value = float('-inf') if self.mode == 'max' else float('inf')
        self.generations_since_save = 0
        
    def on_generation_end(self, generation: int, model: AbstractGeneticModel) -> None:
        """Called at the end of each generation."""
        self.generations_since_save += 1
        if self.generations_since_save < self.period:
            return
            
        self.generations_since_save = 0
        
        # Get current value from history or population stats
        current = None
        if model.history.history.get(self.monitor) and model.history.history[self.monitor]:
            current = model.history.history[self.monitor][-1]
        elif hasattr(model.population, "get_statistics") and self.monitor in model.population.get_statistics():
            current = model.population.get_statistics()[self.monitor]
        
        if current is None:
            if self.verbose > 0:
                print(f"Warning: {self.monitor} not found in model history or population stats.")
            return
            
        # Determine if we should save
        if self.save_best_only:
            if self.mode == 'max':
                improved = current > self.best_value
            else:  # mode == 'min'
                improved = current < self.best_value
                
            if not improved:
                return
                
            self.best_value = current
        
        # Format filepath with current values
        formatted_path = self.filepath.format(
            generation=generation,
            max_fitness=model.history.history['max_fitness'][-1] if model.history.history['max_fitness'] else 0,
            avg_fitness=model.history.history['avg_fitness'][-1] if model.history.history['avg_fitness'] else 0,
            **model._metadata
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(formatted_path)), exist_ok=True)
        
        # Save the model or best solution
        if self.save_weights_only:
            best_solution = model.get_best_solution()
            if best_solution:
                with open(formatted_path, 'wb') as f:
                    pickle.dump(best_solution, f)
        else:
            with open(formatted_path, 'wb') as f:
                pickle.dump(model, f)
                
        if self.verbose > 0:
            print(f"\nCheckpoint saved to {formatted_path}")


class ProgressLogger(AbstractCallback):
    """Log progress during evolution.
    
    This callback prints detailed information during training, including
    statistics and timing information.
    """
    
    def __init__(self, 
                log_interval: int = 1,
                detailed: bool = False,
                log_fn: Callable[[str], None] = print) -> None:
        """Initialize progress logger callback.
        
        Args:
            log_interval: Interval (in generations) between logs.
            detailed: Whether to print detailed statistics.
            log_fn: Function to use for logging.
        """
        super().__init__()
        self.log_interval = log_interval
        self.detailed = detailed
        self.log_fn = log_fn
        self.start_time = None
        
    def on_evolution_begin(self, model: AbstractGeneticModel) -> None:
        """Called at the beginning of evolution."""
        self.start_time = time.time()
        self.log_fn("Evolution started")
        
    def on_generation_end(self, generation: int, model: AbstractGeneticModel) -> None:
        """Called at the end of each generation."""
        if generation % self.log_interval != 0:
            return
            
        elapsed = time.time() - self.start_time
        stats = model.population.get_statistics()
        
        if self.detailed:
            self.log_fn(f"Generation {generation} - "
                      f"Max: {stats['max_fitness']:.4f}, "
                      f"Avg: {stats['avg_fitness']:.4f}, "
                      f"Min: {stats['min_fitness']:.4f}, "
                      f"Std: {stats['fitness_std']:.4f}, "
                      f"Time: {elapsed:.2f}s")
        else:
            self.log_fn(f"Generation {generation} - "
                      f"Max: {stats['max_fitness']:.4f}, "
                      f"Avg: {stats['avg_fitness']:.4f}, "
                      f"Time: {elapsed:.2f}s")
        
    def on_evolution_end(self, model: AbstractGeneticModel) -> None:
        """Called at the end of evolution."""
        total_time = time.time() - self.start_time
        summary = model.history.get_summary()
        
        self.log_fn("\nEvolution completed")
        self.log_fn(f"Total time: {total_time:.2f}s")
        self.log_fn(f"Generations: {summary['total_generations']}")
        self.log_fn(f"Best fitness: {summary['best_fitness']}")
        
        if self.detailed:
            self.log_fn(f"Average generation time: {summary['avg_generation_time']:.4f}s")
            self.log_fn(f"Final average fitness: {summary['final_avg_fitness']}")