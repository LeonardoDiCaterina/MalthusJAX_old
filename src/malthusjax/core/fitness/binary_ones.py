from typing import Any
from malthusjax.core.base import Compatibility
from jax import Array # type: ignore
import jax.numpy as jnp # type: ignore
import jax # type: ignore
from .base import AbstractFitnessEvaluator
from .base_old import AbstractFitnessEvaluator_old
from ..solution.base import AbstractSolution


class BinarySumFitnessEvaluator_old(AbstractFitnessEvaluator_old):
    """
    Concrete fitness evaluator that sums binary values in the genome.
    """
    
    @staticmethod
    @jax.jit
    def fitness_function(solution: AbstractSolution) -> float:
        """
        Sum the binary values in the genome.
        """
        return jnp.sum(solution.to_tensor())

    @staticmethod
    @jax.jit
    def tensor_fitness_function(genome_tensor: Array) -> float:
        """
        Sum the binary values in the genome tensor.
        """
        return jnp.sum(genome_tensor)
    
class BinarySumFitnessEvaluator(AbstractFitnessEvaluator):
    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        return jnp.sum(genome_tensor)

class KnapsackFitnessEvaluator_old(AbstractFitnessEvaluator_old):
    """
    Concrete fitness evaluator for knapsack problem.
    """
    def __init__(self, weights: Array, values: Array, weight_limit: float = 10.0, default_exceding_weight_penalization = -1.0) -> None:
        """
        Initialize knapsack fitness evaluator with weights and values.
        
        Args:
            weights: JAX array of item weights
            values: JAX array of item values
        """
        self.weights = weights
        self.values = values
        self.weight_limit = weight_limit
        self.default_exceding_weight_penalization = default_exceding_weight_penalization
    
    @jax.jit
    def fitness_function(self, solution: AbstractSolution) -> float:
        """
        Evaluate the fitness of a solution for the knapsack problem.
        
        Args:
            solution: The solution to evaluate
            
        Returns:
            Fitness value as float
        """
        genome_tensor = solution.genome.to_tensor()
        total_weight = jnp.sum(genome_tensor * self.weights)
        total_value = jnp.sum(genome_tensor * self.values)
        
        # Penalize solutions that exceed weight limit 
        if total_weight > self.weight_limit:
            print(f"Total weight {total_weight} exceeds limit {self.weight_limit}. Applying penalty.")
            return self.default_exceding_weight_penalization
        
        return total_value
    @jax.jit
    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        """
        Tensor-only version of knapsack fitness function.
        
        Args:
            genome_tensor: JAX array representing genome
            
        Returns:
            Fitness value as float
        """ 
        # Note: Using closure to access weights and values from instance
        total_weight = jnp.sum(genome_tensor * self.weights)
        total_value = jnp.sum(genome_tensor * self.values)
        
        return jnp.where(total_weight <= self.weight_limit, total_value, 
                         self.default_exceding_weight_penalization)
        


class KnapsackFitnessEvaluator(AbstractFitnessEvaluator):
    """
    Concrete fitness evaluator for knapsack problem that extends AbstractFitnessEvaluator2.
    """
    def __init__(self, weights: Array, values: Array, weight_limit: float = 10.0, default_exceding_weight_penalization = -1.0) -> None:
        """
        Initialize knapsack fitness evaluator with weights and values.
        
        Args:
            weights: JAX array of item weights
            values: JAX array of item values
            weight_limit: Maximum allowed weight for the knapsack
            default_exceding_weight_penalization: Penalty value for exceeding weight limit
        """
        self.weights = weights
        self.values = values
        self.weight_limit = weight_limit
        self.default_exceding_weight_penalization = default_exceding_weight_penalization
    
    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        """
        Tensor-only version of knapsack fitness function.
        
        Args:
            genome_tensor: JAX array representing genome
            
        Returns:
            Fitness value as float
        """
        total_weight = jnp.sum(genome_tensor * self.weights)
        total_value = jnp.sum(genome_tensor * self.values)
        
        # Use jnp.where to avoid control flow issues in JIT compilation
        return jnp.where(total_weight <= self.weight_limit, 
                         total_value, 
                         self.default_exceding_weight_penalization)