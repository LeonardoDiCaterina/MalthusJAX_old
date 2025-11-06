from functools import partial
from typing import Any, Callable

from jax import Array # type: ignore
import jax.numpy as jnp # type: ignore
import jax # type: ignore
from jax import jit, vmap # type: ignore
from .base import AbstractFitnessEvaluator



class SphereFitnessEvaluator(AbstractFitnessEvaluator):


    def __init__(self) -> None:
        super().__init__()
    
    def get_tensor_fitness_function(self) -> Callable[[Array], float]:
        
        def tensor_fitness_function(genome_tensor: Array) -> float:
            """
            Tensor-only version of Sphere fitness function.

            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            return jnp.sum(genome_tensor ** 2)
        
        return tensor_fitness_function
    
    
class RastriginFitnessEvaluator(AbstractFitnessEvaluator):


    def __init__(self, A: float = 10.0) -> None:
        super().__init__()
        self.A = A

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:
        
        def tensor_fitness_function(genome_tensor: Array) -> float:
            """
            Tensor-only version of Rastrigin fitness function.

            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            n = genome_tensor.shape[0]
            return self.A * n + jnp.sum(genome_tensor ** 2 - self.A * jnp.cos(2 * jnp.pi * genome_tensor))
        
        return tensor_fitness_function
    

class RosenbrockFitnessEvaluator(AbstractFitnessEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:

        def tensor_fitness_function(genome_tensor: Array) -> float:
            """
            Tensor-only version of Rosenbrock fitness function.

            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            return jnp.sum(100.0 * (genome_tensor[1:] - genome_tensor[:-1] ** 2) ** 2 + (genome_tensor[:-1] - 1) ** 2)
        
        return tensor_fitness_function


class AckleyFitnessEvaluator(AbstractFitnessEvaluator):

    def __init__(self, a: float = 20.0, b: float = 0.2, c: float = 2 * jnp.pi) -> None:
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:

        def tensor_fitness_function(genome_tensor: Array) -> float:
            """
            Tensor-only version of Ackley fitness function.

            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            n = genome_tensor.shape[0]
            sum1 = jnp.sum(genome_tensor ** 2)
            sum2 = jnp.sum(jnp.cos(self.c * genome_tensor))
            term1 = -self.a * jnp.exp(-self.b * jnp.sqrt(sum1 / n))
            term2 = -jnp.exp(sum2 / n)
            return term1 + term2 + self.a + jnp.exp(1)
        
        return tensor_fitness_function
    
class GriewankFitnessEvaluator(AbstractFitnessEvaluator):

    def __init__(self) -> None:
        super().__init__()

    def get_tensor_fitness_function(self) -> Callable[[Array], float]:
        
        def tensor_fitness_function(genome_tensor: Array) -> float:
            """
            Tensor-only version of Griewank fitness function.

            Args:
                genome_tensor: JAX array representing genome
                
            Returns:
                Fitness value as float
            """
            sum_term = jnp.sum(genome_tensor ** 2) / 4000.0
            prod_term = jnp.prod(jnp.cos(genome_tensor / jnp.sqrt(jnp.arange(1, genome_tensor.shape[0] + 1))))
            return sum_term - prod_term + 1.0
        
        return tensor_fitness_function