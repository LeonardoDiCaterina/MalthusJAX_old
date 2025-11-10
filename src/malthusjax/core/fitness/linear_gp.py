import functools
import pandas as pd # type: ignore
from typing import Any
from jax import Array # type: ignore
import jax.numpy as jnp # type: ignore
import jax # type: ignore
from .base import AbstractFitnessEvaluator
    
class TreeGPEvaluator(AbstractFitnessEvaluator):
    """
    Concrete fitness evaluator that extends AbstractFitnessEvaluator.
    Computes fitness as the sum of ones in a binary genome.
    """
    
    def __init__(self,dataframe: pd.DataFrame, n_instructions_in_genome: int) -> None:
        
        self.name = "BinarySumFitnessEvaluator"
        self.data_array = dataframe
        self.n_features = len(dataframe)
        self.n_instructions_in_genome = n_instructions_in_genome
        super().__init__()

    def get_tensor_fitness_function(self) -> float:
        """
        JIT-compatible tensor-only fitness function that computes the sum of ones in the genome tensor.

        Args:
            genome_tensor (Array): JAX array representing the genome

        Returns:
            float: Fitness value as float
        """
        def operation_function(x1_value, x2_value, operation_index):
            """Define mathematical operations for tree nodes"""
            return jax.lax.switch(operation_index, [
                lambda: x1_value + x2_value,        # 0: Addition
                lambda: x1_value - x2_value,        # 1: Subtraction
                lambda: x1_value * x2_value,        # 2: Multiplication
                lambda: x1_value / (x2_value + 1e-8),  # 3: Division (with small epsilon)
                lambda: jnp.maximum(x1_value, x2_value)  # 4: Maximum
            ])
    
    
        def evaluate_single_row(operations_genome, indexes_genome , X_sample, depth_instructions, n_features):
            """Evaluate the tree for a single sample"""
                        
            def body_fn(carry, input_element):
                results_history = carry
                
                index_0 = indexes_genome[input_element, 0]
                index_1 = indexes_genome[input_element, 1]
                
                
                def get_value( index, X_sample, results_history):
                    """Get value from either original features or computed results"""
                    return jax.lax.cond(
                        index < len(X_sample),
                        lambda: X_sample[index],  # Feature value
                        lambda: results_history[index - len(X_sample)]  # Computed node value
                    )
                value0 = get_value(index_0, X_sample, results_history)
                value1 = get_value(index_1, X_sample, results_history)
                
                atomic_result = operation_function(value0, value1, operations_genome[input_element])
                results_history = results_history.at[input_element].set(atomic_result)
                
                return results_history, atomic_result
            
            # Initialize and execute scan
            initial_results_history = jnp.zeros((depth_instructions,))
            input_data = jnp.arange(depth_instructions-5)
            
            final_results_history, all_results = jax.lax.scan(
                f=body_fn,
                init=initial_results_history,
                xs=input_data
            )
            
            return jnp.max(final_results_history)        
        return functools.partial(evaluate_single_row,
                                 X_sample=self.data_array,
                                 depth_instructions=self.n_instructions_in_genome,
                                 n_features=self.n_features)
