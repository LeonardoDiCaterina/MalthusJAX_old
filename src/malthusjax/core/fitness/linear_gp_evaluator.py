"""
Linear Genetic Programming fitness evaluator with symbiotic evaluation.

Implements sophisticated evaluation where each instruction in a linear genome
is treated as an atomic tree, enabling symbiotic evolution where the best
sub-components compete and are selected independently.
"""

from typing import Tuple
from functools import partial
from flax import struct  # type: ignore
import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
import chex  # type: ignore

from malthusjax.core.fitness.evaluators import BaseEvaluator, RegressionData
from malthusjax.core.genome.linear import LinearGenome, LinearGenomeConfig


# --- Robust Primitive Functions ---
def op_add(x: float, y: float) -> float:
    """Addition operation."""
    return x + y

def op_sub(x: float, y: float) -> float:
    """Subtraction operation."""
    return x - y

def op_mul(x: float, y: float) -> float:
    """Multiplication operation."""
    return x * y

def op_div(x: float, y: float) -> float:
    """Protected division operation."""
    return jnp.where(jnp.abs(y) < 1e-6, 1.0, x / y)

def op_sin(x: float, y: float) -> float:
    """Sine operation (uses only first argument)."""
    return jnp.sin(x)

def op_cos(x: float, y: float) -> float:
    """Cosine operation (uses only first argument)."""
    return jnp.cos(x)

def op_exp(x: float, y: float) -> float:
    """Protected exponential operation."""
    return jnp.where(x > 10.0, jnp.exp(10.0), jnp.exp(x))

def op_log(x: float, y: float) -> float:
    """Protected logarithm operation."""
    return jnp.where(x <= 0.0, 0.0, jnp.log(jnp.abs(x)))


# Function registry for jax.lax.switch
OP_FUNCTIONS = (op_add, op_sub, op_mul, op_div, op_sin, op_cos, op_exp, op_log)
OP_NAMES = ["ADD", "SUB", "MUL", "DIV", "SIN", "COS", "EXP", "LOG"]


@struct.dataclass
class LinearGPEvaluator(BaseEvaluator[LinearGenome, LinearGenomeConfig, RegressionData]):
    """
    Linear Genetic Programming evaluator with symbiotic fitness.
    
    Evaluates each instruction as an atomic tree and returns fitness
    for all instructions, enabling sophisticated selection strategies
    that can pick the best sub-components of programs.
    """
    
    def predict_one(self, genome: LinearGenome, x_input: chex.Array) -> chex.Array:
        """
        Execute one genome on one input vector.
        
        Args:
            genome: Linear genome to execute
            x_input: Input vector
            
        Returns:
            Array of shape (length,) containing output of each instruction
        """
        # 1. Initialize memory: inputs + instruction outputs
        total_mem = self.config.num_inputs + self.config.length
        memory = jnp.zeros(total_mem)
        memory = memory.at[:self.config.num_inputs].set(x_input)
        
        # 2. Execute instructions sequentially
        def step(current_mem, inputs):
            mem, write_idx = current_mem
            op_code, arg_indices = inputs
            
            # Fetch arguments and execute operation
            args_val = jnp.take(mem, arg_indices)
            result = jax.lax.switch(op_code, OP_FUNCTIONS, args_val[0], args_val[1])
            result = jnp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Store result in memory
            new_mem = mem.at[write_idx].set(result)
            
            # Return new memory state and the instruction output
            return (new_mem, write_idx + 1), result

        init_state = (memory, self.config.num_inputs)
        
        # Execute all instructions and collect outputs
        (_, _), instruction_outputs = jax.lax.scan(
            step, init_state, (genome.ops, genome.args)
        )
        
        return instruction_outputs

    def evaluate(self, genome: LinearGenome, data: RegressionData) -> chex.Array:
        """
        Evaluate genome with symbiotic fitness.
        
        Args:
            genome: Genome to evaluate
            data: Tuple of (X, y) for regression
            
        Returns:
            Array of shape (length,) with fitness of each instruction
        """
        X, y = data
        
        # 1. Get predictions for all instructions on all data points
        # Shape: (n_samples, n_instructions)
        all_predictions = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        
        # 2. Calculate MSE for each instruction
        # Broadcast targets to shape (n_samples, n_instructions)
        y_broadcast = y[:, None]
        
        # Compute squared errors for all instruction-sample pairs
        squared_errors = (all_predictions - y_broadcast) ** 2
        
        # Mean over samples (axis 0) -> shape (n_instructions,)
        mse_per_instruction = jnp.mean(squared_errors, axis=0)
        
        # 3. Return negative MSE (higher is better)
        return -mse_per_instruction

    def get_best_instruction_fitness(self, fitness: chex.Array) -> float:
        """
        Get the fitness of the best instruction in a genome.
        
        Args:
            fitness: Array of instruction fitnesses
            
        Returns:
            Scalar fitness of best instruction
        """
        return jnp.max(fitness)

    def get_program_prediction(self, genome: LinearGenome, X: chex.Array, instruction_idx: int = -1) -> chex.Array:
        """
        Get predictions from a specific instruction or the last instruction.
        
        Args:
            genome: Genome to execute
            X: Input data
            instruction_idx: Which instruction to use (-1 for last)
            
        Returns:
            Predictions from the specified instruction
        """
        all_outputs = jax.vmap(self.predict_one, in_axes=(None, 0))(genome, X)
        return all_outputs[:, instruction_idx]