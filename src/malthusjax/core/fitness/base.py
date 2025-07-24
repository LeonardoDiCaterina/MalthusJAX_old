"""
Abstract base classes for fitness functions in MalthusJAX.

This module defines the fundamental fitness evaluation abstractions that leverage
JAX's vmap and jit for efficient batch evaluation of genome populations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
from jax import Array  # type: ignore

from ..base import JAXTensorizable, Compatibility
from ..genome.base import AbstractGenome
from ..solution.base import AbstractSolution


class AbstractFitnessEvaluator(ABC):
    """
    Abstract base class for fitness evaluation using JAX vectorization.
    
    This class provides both individual solution evaluation and efficient
    batch evaluation using JAX's vmap and jit compilation.
    """
    
    def __init__(self):
        """Initialize the fitness evaluator."""
        self._batch_fitness_fn: Optional[Callable] = None
        self._compiled_for_shape: Optional[tuple] = None
    
    @abstractmethod  
    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        """
        Abstract tensor-only version of fitness function for vectorized operations.
        
        This method must be implemented by subclasses and should be JAX-compatible
        (no Python control flow, pure functions only).
        
        Args:
            genome_tensor: JAX array representing a single genome
            
        Returns:
            Fitness value as a scalar float
        """
        pass
    
    def fitness_function(self, solution: AbstractSolution) -> float:
        """
        Evaluate a single solution's fitness.
        
        Args:
            solution: The solution to evaluate
            
        Returns:
            Fitness value as float
        """
        genome_tensor = solution.genome.to_tensor()
        return float(self.tensor_fitness_function(genome_tensor))
    
    def _get_batch_fitness_function(self, tensor_shape: tuple) -> Callable:
        """
        Get or create the JIT-compiled batch fitness function.
        
        Args:
            tensor_shape: Shape of the genome tensors for compilation
            
        Returns:
            JIT-compiled vectorized fitness function
        """
        if self._batch_fitness_fn is None or self._compiled_for_shape != tensor_shape:
            # Create and cache the batch function
            self._batch_fitness_fn = jax.jit(jax.vmap(self.tensor_fitness_function))
            self._compiled_for_shape = tensor_shape
            
        return self._batch_fitness_fn
    
    def evaluate_solutions(self, solutions: List[AbstractSolution]) -> None:
        """
        Evaluate a list of solutions simultaneously using JAX vectorization.
        
        This method is optimized for batch evaluation and will JIT-compile
        the fitness function for efficient repeated use.
        
        Args:
            solutions: List of Solution objects to evaluate
            
        Raises:
            ValueError: If solutions list is empty
            RuntimeError: If fitness evaluation fails
        """
        if not solutions:
            raise ValueError("Cannot evaluate empty solutions list")
        
        try:
            # Convert to tensors (Python overhead - unavoidable)
            genome_tensors = jnp.stack([solution.genome.to_tensor() for solution in solutions])
            
            # Get cached or create new batch function
            batch_fitness_fn = self._get_batch_fitness_function(genome_tensors.shape[1:])
            
            # Evaluate batch
            fitness_values = batch_fitness_fn(genome_tensors)
            
            # Assign results back (Python overhead - unavoidable)  
            for solution, fitness in zip(solutions, fitness_values):
                solution.raw_fitness = float(fitness)
                
        except Exception as e:
            raise RuntimeError(f"Fitness evaluation failed: {e}") from e
    
    def evaluate_population_stack(self, population_stack: Array) -> Array:
        """
        Evaluate a stacked population tensor directly.
        
        This method works with the Population.to_stack() output for maximum efficiency.
        Use this when you don't need to update individual solution fitness values.
        
        Args:
            population_stack: Stacked genome tensors from Population.to_stack()
            
        Returns:
            Array of fitness values corresponding to each genome
            
        Raises:
            ValueError: If population stack is empty
            RuntimeError: If fitness evaluation fails
        """
        if population_stack.size == 0:
            raise ValueError("Cannot evaluate empty population stack")
        
        try:
            # Get cached or create new batch function
            batch_fitness_fn = self._get_batch_fitness_function(population_stack.shape[1:])
            
            # Evaluate entire stack
            fitness_values = batch_fitness_fn(population_stack)
            
            return fitness_values
            
        except Exception as e:
            raise RuntimeError(f"Population stack fitness evaluation failed: {e}") from e
    
    def evaluate_population(self, population) -> None:
        """
        Evaluate an entire population efficiently using stack operations.
        
        This method leverages Population.to_stack() for optimal performance,
        then updates individual solution fitness values.
        
        Args:
            population: Population object to evaluate
            
        Raises:
            ValueError: If population is empty
            RuntimeError: If fitness evaluation fails
        """
        if population.size == 0:
            raise ValueError("Cannot evaluate empty population")
        
        try:
            # Use stack-based evaluation for efficiency
            population_stack = population.to_stack()
            fitness_values = self.evaluate_population_stack(population_stack)
            
            # Update individual solution fitness values
            solutions = population.get_solutions()
            for solution, fitness in zip(solutions, fitness_values):
                solution.raw_fitness = float(fitness)
                
        except Exception as e:
            raise RuntimeError(f"Population fitness evaluation failed: {e}") from e
    
    def evaluate_single_solution(self, solution: AbstractSolution) -> None:
        """
        Evaluate a single solution and set its fitness.
        
        Args:
            solution: Solution to evaluate
        """
        solution.raw_fitness = self.fitness_function(solution)
    
    def get_compatibility_info(self) -> Dict[str, Any]:
        """
        Get compatibility information for this fitness evaluator.
        
        Returns:
            Dictionary containing compatibility requirements
        """
        return {
            'requires_tensor_function': True,
            'supports_batch_evaluation': True,
            'supports_population_stack': True,
            'jax_compatible': True
        }


class FitnessEvaluatorMixin:
    """
    Mixin class providing common fitness evaluation utilities.
    """
    
    @staticmethod
    def validate_fitness_value(fitness: float) -> bool:
        """
        Validate that a fitness value is valid (not NaN, not infinite).
        
        Args:
            fitness: Fitness value to validate
            
        Returns:
            True if valid, False otherwise
        """
        return jnp.isfinite(fitness) and not jnp.isnan(fitness)
    
    @staticmethod
    def clip_fitness(fitness: float, min_val: float = -1e6, max_val: float = 1e6) -> float:
        """
        Clip fitness values to prevent numerical issues.
        
        Args:
            fitness: Original fitness value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Clipped fitness value
        """
        return float(jnp.clip(fitness, min_val, max_val))
    
    @staticmethod
    def normalize_fitness_array(fitness_values: Array, method: str = 'minmax') -> Array:
        """
        Normalize an array of fitness values.
        
        Args:
            fitness_values: Array of raw fitness values
            method: Normalization method ('minmax', 'zscore', 'rank')
            
        Returns:
            Normalized fitness values
        """
        if method == 'minmax':
            min_val = jnp.min(fitness_values)
            max_val = jnp.max(fitness_values)
            return (fitness_values - min_val) / (max_val - min_val + 1e-10)
        elif method == 'zscore':
            mean_val = jnp.mean(fitness_values)
            std_val = jnp.std(fitness_values)
            return (fitness_values - mean_val) / (std_val + 1e-10)
        elif method == 'rank':
            ranks = jnp.argsort(jnp.argsort(fitness_values))
            return ranks / (len(fitness_values) - 1)
        else:
            raise ValueError(f"Unknown normalization method: {method}")