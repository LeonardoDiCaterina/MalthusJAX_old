from typing import Callable
from jax import Array # type: ignore
import jax.numpy as jnp # type: ignore
import jax # type: ignore
from .base import AbstractFitnessEvaluator

import functools
    
class BinarySumFitnessEvaluator(AbstractFitnessEvaluator):
    """
    Concrete fitness evaluator that extends AbstractFitnessEvaluator.
    Computes fitness as the sum of ones in a binary genome.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.name = "BinarySumFitnessEvaluator"


    def get_pure_fitness_function(self) -> Callable[[Array], float]:
        
        def sum_function(genome_tensor: Array) -> float:
            """
            JIT-compatible tensor-only fitness function that computes the sum of ones in the genome tensor.

            Args:
                genome_tensor (Array): JAX array representing the genome

            Returns:
                float: Fitness value as float
            """
            return jnp.sum(genome_tensor)
        return sum_function
    
    
    

        


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
        super().__init__()
        self.name = "KnapsackFitnessEvaluator"


    def get_pure_fitness_function(self) -> Callable[[Array], float]:

        def knapsack_function(genome_tensor: Array, weights: Array, values: Array, weight_limit:float,default_exceding_weight_penalization:float ) -> float:
            """
            JIT-compatible tensor-only fitness function for knapsack problem.
            
            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            total_weight = jnp.sum(genome_tensor * weights)
            total_value = jnp.sum(genome_tensor * values)
            
            # Use jnp.where to avoid control flow issues in JIT compilation
            return jnp.where(total_weight <= weight_limit, 
                            total_value, 
                            default_exceding_weight_penalization)
            
        # Use functools.partial to fix weights and values parameters
        return functools.partial(knapsack_function,
                                 weights=self.weights,
                                 values=self.values,
                                 weight_limit=self.weight_limit,
                                 default_exceding_weight_penalization=self.default_exceding_weight_penalization)