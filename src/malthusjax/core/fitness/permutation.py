from jax import Array  # type: ignore
import jax.numpy as jnp  # type: ignore
from .base import AbstractFitnessEvaluator
import jax # type: ignore


class SortingFitnessEvaluator(AbstractFitnessEvaluator):

    def __init__(self):
         super().__init__()
         self.name = "SortingFitnessEvaluator"

    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        # Implement the fitness function for sorting
        sorted_tensor = jnp.sort(genome_tensor)
        fitness = -jnp.sum(jnp.abs(genome_tensor - sorted_tensor))
        return fitness.astype(float)

class TSPFitnessEvaluator(AbstractFitnessEvaluator):
    def __init__(self, distance_matrix: jnp.ndarray):
        super().__init__()
        self.name = "TSPFitnessEvaluator"
        self.distance_matrix = distance_matrix

    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        # Implement the fitness function for TSP using jax.lax
        indices = genome_tensor.astype(int)
        def body_fun(i, acc):
            return acc + self.distance_matrix[indices[i - 1], indices[i]]
        total_distance = jax.lax.fori_loop(1, len(indices), body_fun, 0.0)
        return float(total_distance)

class FixedGroupingFitnessEvaluator(AbstractFitnessEvaluator):
    def __init__(self, group_size: int, penalty_matrix: jnp.ndarray = None):
        super().__init__()
        self.name = "FixedGroupingFitnessEvaluator"
        self.group_size = group_size
        self.penalty_matrix = penalty_matrix # jnp.ndarray of shape (num_elements, num_elements)

    def tensor_fitness_function(self, genome_tensor: Array) -> float:
        """
        JIT-compatible tensor-only fitness function for fixed grouping.
        Args:
            genome_tensor (Array): JAX array representing the genome

        Returns:
            float: Fitness value as float
        """
        n = genome_tensor.shape[0]
        num_groups = jnp.floor_divide(n, self.group_size)
        def body_fun(i, fitness):
            start = i * self.group_size
            group = jax.lax.dynamic_slice(genome_tensor, (start,), (self.group_size,))
            penalty = jnp.sum(self.penalty_matrix[group[:, None], group[None, :]])
            return fitness + penalty

        fitness = jax.lax.fori_loop(0, num_groups, body_fun, 0.0)
        return fitness
