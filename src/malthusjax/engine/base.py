"""
Abstract base classes for MalthusJAX engines.

This module defines the core abstractions that all engines must implement:
- AbstractState: Base state for all engine states
- AbstractEngine: Base engine interface
"""

from typing import Callable, Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
from jax.random import PRNGKey # type: ignore
from jax import Array # type: ignore
import flax.struct # type: ignore

from malthusjax.core.genome import AbstractGenome
from malthusjax.core.fitness import AbstractFitnessEvaluator
from malthusjax.operators.selection import AbstractSelectionOperator
from malthusjax.operators.crossover import AbstractCrossover
from malthusjax.operators.mutation import AbstractMutation


@flax.struct.dataclass
class AbstractState:
    """
    Abstract base class for all MalthusJAX states.
    
    This defines the minimal fields required by all engines.
    Must be a flax.struct.dataclass to be a JAX Pytree.
    """
    
    # --- Core GA Data ---
    population: Array
    """The JAX array of all genomes. Shape: (population_size, *genome_shape)"""
    
    fitness: Array
    """The JAX array of fitness values for the population. Shape: (population_size,)"""
    
    # --- Elitism & Tracking ---
    best_genome: Array
    """The single best genome found so far in the run. Shape: (*genome_shape)"""
    
    best_fitness: float
    """The fitness value of the single best genome."""
    
    # --- Loop State ---
    key: PRNGKey
    """The JAX PRNGKey. Must be updated every step for reproducible randomness."""
    
    generation: int
    """A simple integer counter for the current generation."""


class AbstractEngine(ABC):
    """
    Abstract base class for all MalthusJAX engines.
    
    Defines the interface that all engines must implement.
    """
    
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation,
                 elitism: int):
        """
        Initialize the engine with core Level 1 and Level 2 components.
        
        Args:
            genome_representation: The genome representation object
            fitness_evaluator: The fitness evaluator object
            selection_operator: The selection operator object
            crossover_operator: The crossover operator object
            mutation_operator: The mutation operator object
            elitism: Number of elite individuals to retain each generation
        """
        self._genome_representation = genome_representation
        self._fitness_evaluator = fitness_evaluator
        self._selection_operator = selection_operator
        self._crossover_operator = crossover_operator
        self._mutation_operator = mutation_operator
        self._elitism = elitism
        
        # Get pure functions for JIT compilation
        self._init_fn = genome_representation.get_random_initialization_pure()
        self._fitness_fn = fitness_evaluator.get_pure_fitness_function()
        self._selection_fn = selection_operator.get_pure_function()
        self._crossover_fn = crossover_operator.get_pure_function()
        self._mutation_fn = mutation_operator.get_pure_function()
    
    @property
    def elitism(self) -> int:
        """Number of elite individuals."""
        return self._elitism

    @elitism.setter
    def elitism(self, value: int):
        """Set the number of elite individuals."""
        self._elitism = value
    
    @abstractmethod
    def run(self, 
           key: PRNGKey, 
           num_generations: int, 
           pop_size: int,
           initial_population: Optional[Array] = None) -> Tuple[AbstractState, Any]:
        """
        Run the evolutionary algorithm.
        
        Args:
            key: JAX random key
            num_generations: Number of generations to run
            pop_size: Population size
            initial_population: Optional initial population (if None, will be randomly initialized)
            
        Returns:
            Tuple of (final_state, history_or_intermediates)
            - final_state: The final evolution state
            - history_or_intermediates: Engine-specific data (history for Production, intermediates for Research)
        """
        pass
             