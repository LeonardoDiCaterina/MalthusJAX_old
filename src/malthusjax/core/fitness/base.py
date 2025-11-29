"""
Abstract base classes for fitness functions in MalthusJAX.

This module defines the fundamental fitness evaluation abstractions that leverage
JAX's vmap and jit for efficient batch evaluation of genome populations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Union
import jax # type: ignore
import jax.numpy as jnp # type: ignore
from jax import Array # type: ignore

# Import from compatibility layer  
from malthusjax.compat import JAXTensorizable

class AbstractFitnessEvaluator(ABC):
    """
    Abstract base class for fitness evaluation using JAX vectorization.
    
    This class provides both individual solution evaluation and efficient
    batch evaluation using JAX's vmap and jit compilation.
    """
    
    def __init__(self):
        """Initialize the fitness evaluator."""
        self._tensor_fitness_fn: Optional[Callable] = self.get_pure_fitness_function()
        self._batch_fitness_fn: Optional[Callable] = self.get_batch_fitness_function()
        
    
    def debug_tensor_fitness_function(self, list_of_genome_edge_cases: List[Array], list_of_expected_fitness: List[float]) -> None:
        """
        This method is meant to be used for checking the tensor fitness function outside of
        the evolutionary loop.
        It will check if the tensor_fitness_function can be jit-compiled and vectorized,
        and will evaluate it on a list of edge case genomes, comparing the results to expected fitness
        values.
        Args:
            list_of_genome_edge_cases: List of genome tensors to test
            list_of_expected_fitness: Corresponding expected fitness values for the test genomes
        
        Raises:
            AssertionError: If the fitness function does not return expected values
            ValueError: If tensor_fitness_function is not implemented
        
        Note:
            This method is not JIT-compiled or vectorized, as it is intended for debugging purposes only.
        """
        # check if the tensor_fitness_function can be jit-compiled and vectorized
        
        try:
            jax.jit(self._tensor_fitness_fn)
            jax.jit(self._batch_fitness_fn)
            print("the tensor_fitness_function can be jit-compiled and vectorized correctly.")

        except Exception as e:
            print(f"JIT or vmap compilation failed: {e}")
            return
        
        print("*"*50)
        print("\n\n\nEvaluating edge cases:\n\n")

        for genome_tensor, expected_fitness in zip(list_of_genome_edge_cases, list_of_expected_fitness):
            try:
                result = self._tensor_fitness_fn(genome_tensor)
                print("-"*20)
                assert result == expected_fitness, f"Expected {expected_fitness}, but got {result} on genome {genome_tensor}"
                print(f"Genome {genome_tensor} evaluated correctly with fitness {result}.")
            except Exception as e:
                print(f"Error evaluating genome {genome_tensor}: {e}")
        if self._tensor_fitness_fn is None:
            raise ValueError("tensor_fitness_function must be implemented")
    
    @abstractmethod
    def get_pure_fitness_function(self) -> Callable:
        """
        Get the tensor-only fitness function.
        
        Returns:
            Callable that takes a genome tensor and returns a float fitness value
        """
        pass

    def get_batch_fitness_function(self) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Get the batch fitness function that can evaluate multiple genomes at once.
        
        Returns:    
            Callable that takes a batch of genome tensors and returns a JAX array of fitness values
            
        Raises:
            ValueError: If tensor_fitness_function is not implemented
        """
        if self._tensor_fitness_fn is None:
            raise ValueError("tensor_fitness_function must be implemented")
        
        # Vectorize the tensor fitness function
        vmap_fn = jax.vmap(self._tensor_fitness_fn)

        return vmap_fn

    def evaluate_single(self, genome: Any) -> float:
        """
        Evaluate a single genome's fitness.

        Args:
            genome: Either an AbstractGenome instance or a JAX tensor
        Returns:
            Fitness value as float
        Raises:
            RuntimeError: If fitness evaluation fails
        """
        if callable(getattr(genome, 'to_tensor', None)):
            genome_tensor = genome.to_tensor()
        else:
            genome_tensor = genome
        try:
            return float(self._tensor_fitness_fn(genome_tensor))
        except Exception as e:
            raise RuntimeError(f"Single genome fitness evaluation failed: {e}") from e

    def evaluate_batch(self, genomes: Union[List[jnp.ndarray], List[JAXTensorizable], jnp.ndarray], return_tensors: bool = False) -> List[float]:
        """
        Evaluate a batch of genomes efficiently.

        Args:
            genomes: List of AbstractGenome instances or JAX tensors

        Returns:
            List of fitness values as floats
        """
        if len(genomes) == 0:
            return []
        
        population_stack = None
        #if genomes is a jnp.ndarray, assume it's already a stack of tensors
        if isinstance(genomes, jnp.ndarray):
            population_stack = genomes
            
        elif isinstance(genomes):
            population_stack = genomes.to_stack()
        elif not isinstance(genomes, list):
            raise ValueError("Genomes must be provided as a list of AbstractGenome instances or a JAX tensor stack")
        
        # Convert genomes to tensors if they are AbstractGenome instances
        if callable(getattr(genomes[0], 'to_tensor', None)):
            population_stack = jnp.stack([g.to_tensor() for g in genomes])
        elif callable(getattr(genomes, 'to_stack', None)):
            population_stack = genomes.to_stack()
        elif  population_stack is None:
            # Assume they are already tensors, try to stack them
            try:
                population_stack = jnp.stack(genomes)
            except Exception as e:
                raise ValueError(f"Genomes must be AbstractGenome instances, a Population Instance, or JAX tensors {e}") from e

        fitness_values = self._batch_fitness_fn(population_stack)
        if return_tensors:
            return fitness_values
        return [float(f) for f in fitness_values]

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

    def __call__(self, population: List[Any], return_tensors: bool = False) -> jnp.ndarray:
        """
        Allow the fitness evaluator to be called directly on a population.
        
        Args:
            population: List of AbstractGenome instances or JAX tensors
            return_tensors: If True, return JAX array of fitness values instead of list of floats
            
        Returns:
            List of fitness values as floats or JAX array if return_tensors is True
        """
        return self.evaluate_batch(population.to_stack(), return_tensors=return_tensors)
