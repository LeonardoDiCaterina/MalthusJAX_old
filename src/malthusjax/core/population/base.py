from abc import ABC, abstractmethod
from turtle import st
from typing import Callable, List, Optional, Tuple, Dict, Any, Generic, TypeVar

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import jax # type: ignore
from malthusjax.core.base import SerializationContext
from malthusjax.core.genome import AbstractGenome

T = TypeVar('T', bound=AbstractGenome)

class AbstractPopulation(ABC, Generic[T]):
    """Abstract base class for population management in genetic algorithms."""

    def __init__(self, genome_cls: type[T], pop_size: int, random_key: Optional[jar.PRNGKey] = None, context = None, random_init: bool = True, genome_init_params: Dict = {}, fitness_transform: Optional[Callable[[jax.Array], jax.Array]] = None) -> None:
        """Initialize population with maximum size."""


        self._genomes: Optional[jnp.ndarray] = None
        self._fitness_values: Optional[jnp.ndarray] = None

        self._genome_cls = genome_cls
        self._pop_size = pop_size
        self._random_key = random_key
        self._context = context
        self._genome_init_params = genome_init_params
        self._fitness_transform = fitness_transform

        
        #TODO: add defensive programming
        if random_init:
            self._random_init()


        '''def __init__(self, genome_cls, pop_size, random_init=False, random_key=None, genome_init_params=None):
            self._genome_cls = genome_cls
            self._pop_size = pop_size
            self._genome_init_params = genome_init_params if genome_init_params is not None else {}
            self._random_key = random_key
            self._genomes = None
            self._fitness_values = None
            
            if random_init and random_key is not None:
                stack = self.random_stack(
                    genome_cls=genome_cls,
                    stack_size=pop_size,
                    random_key=random_key,
                    genome_init_params=self._genome_init_params
                )
                self.set_solutions(stack)'''
        
    @classmethod
    def random_stack(cls, genome_cls: type[T], stack_size: int, random_key: jar.PRNGKey, genome_init_params: Dict = {}) -> jnp.ndarray:
        """Create a random stack of genomes.

        Args:
            genome_cls: The genome class to use for initialization.
            stack_size: The number of genomes to create.
            random_key: JAX PRNGKey for random operations.
            genome_init_params: Parameters for genome initialization.

        Returns:
            A JAX array containing the randomly initialized genomes.
        """
        keys = jar.split(random_key, stack_size)
        init_fn = genome_cls.get_random_initialization_jit(genome_init_params)
        return jax.vmap(init_fn)(keys)


# Internal methods for population initialization and management
    
    def _random_init(self) -> None:
        """Randomly initialize the population."""


        self._genomes = self.random_stack(
            genome_cls=self._genome_cls,
            stack_size=self._pop_size,
            random_key=self._random_key,
            genome_init_params=self._genome_init_params
        )
        
        
    
    def from_stack(self, stack: jnp.ndarray) -> 'AbstractPopulation':
        """Initialize population from a stack of genomes.

        Args:
            stack: JAX array of genomes to initialize the population.
        """
        if stack.shape[0] > self._pop_size:
            stack = stack[:self._pop_size]

        new_population = AbstractPopulation(
            genome_cls=self._genome_cls,
            pop_size=self._pop_size,
            random_key=self._random_key,
            context=self._context,
            random_init=False,
            genome_init_params=self._genome_init_params,
            fitness_transform=self._fitness_transform
        )
        new_population._genomes = stack
        return new_population
    
    def from_list(self, genomes: List[T]) -> 'AbstractPopulation':
        """Initialize population from a list of genome instances.

        Args:
            genomes: List of genome instances to initialize the population.
        """
        if len(genomes) > self._pop_size:
            genomes = genomes[:self._pop_size]

        stack = jnp.stack([genome.to_tensor() for genome in genomes])

        new_population = self.from_stack(stack)
        return new_population
    
    def from_array_of_indexes(self, indexes: jnp.ndarray) -> 'AbstractPopulation':
        """Create a new population from a JAX array of solution indexes.

        Args:
            indexes: JAX array of indexes to select solutions from the current population.

        Returns:
            A new AbstractPopulation instance with the selected solutions.
        """
        if self._genomes is None:
            raise ValueError("Current population is empty. Cannot select from it.")
        if jnp.any(indexes < 0) or jnp.any(indexes >= len(self)):
            raise ValueError("Indexes are out of bounds.")

        get_from_array_of_indexes_fn = self.get_from_array_of_indexes_jit()
        selected_genomes = get_from_array_of_indexes_fn(self._genomes, indexes)

        new_population = self.from_stack(selected_genomes)
        return new_population
    
# methods to convert population to different formats    

    def to_stack(self) -> jnp.ndarray:  
        """Get the population's genomes as a JAX array.

        Returns:
            JAX array of the population's genomes.
        """
        return self._genomes

    def to_list(self) -> List[T]:
        """Get the population's genomes and their fitness values as a list of genome instances.

        Returns:
            List of tuples containing genome instances and their fitness values.
        """
        return [self._genome_cls.from_tensor(tensor, self._genome_init_params, self._context) for tensor in self._genomes]





# JIT-compiled functions for performance


    def get_distance_matrix_function_jit(self) -> Callable[[jax.Array, jax.Array], float]:
        """
        Returns a JIT-compiled function to compute the distance between two genomes.

        Returns:
            Callable: A function that takes two JAX arrays and returns a float distance.
        """
        distance_function = self._genome_cls.get_distance_jit()
        @jax.jit
        def distance_matrix_fn(stack: jnp.ndarray) -> jnp.ndarray:
            def compute_row(i: int) -> jax.Array:
                return jax.vmap(lambda x: distance_function(stack[i], x))(stack)
            return jax.vmap(compute_row)(jnp.arange(stack.shape[0]))

        return distance_matrix_fn
    
    def get_autocorrection_function_jit(self) -> Callable[[jax.Array], jax.Array]:
        """Get JIT-compiled function that turns invalid genomes into valid ones."""
        auto_correction_function = self._genome_cls.get_autocorrection_jit(self._genome_init_params)
        return jax.jit(jax.vmap(auto_correction_function))
    
    def get_init_function_jit(self) -> Callable[[jar.PRNGKey], jax.Array]:
        """Get JIT-compiled function to initialize a single genome."""
        init_function = self._genome_cls.get_random_initialization_jit(self._genome_init_params)
        return jax.jit(init_function)
    
    
    @staticmethod
    def get_from_array_of_indexes_jit() ->Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
        """Create a new population from a JAX array of solution indexes.
        Args:
            indexes: JAX array of indexes to select solutions from the current population.
        Returns:
            Callable: A function that takes a JAX array (population stack) and a JAX array (indexes) and returns a JAX array of selected solutions.
        """
        def func(stack: jax.Array, indexes: jax.Array) -> jnp.ndarray:
            # Assumes all indexes are valid (0 <= i < stack.shape[0])
            population_indexes = jax.vmap(lambda i: stack[i])(indexes)
            return population_indexes

        return jax.jit(func)
    
    @staticmethod
    def get_sort_by_fitness_jit() -> Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]:
        """Get a JIT-compiled function to sort solutions by fitness values in descending order.

        Returns:
            Callable: A function that takes two JAX arrays (solutions and fitness values) and returns a tuple of sorted solutions and sorted fitness values.
        """
        def sort_fn(genomes: jnp.ndarray, fitness_values: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
            sorted_indices = jnp.argsort(-fitness_values)
            sorted_genomes = genomes[sorted_indices]
            sorted_fitness = fitness_values[sorted_indices]
            return sorted_genomes, sorted_fitness

        return jax.jit(sort_fn)

    @staticmethod
    def get_get_best_index_jit() -> Callable[[jnp.ndarray], int]:
        """Get a JIT-compiled function to find the index of the best fitness value.

        Returns:
            Callable: A function that takes a JAX array of fitness values and returns the index of the highest value.
        """
        def best_index_fn(fitness_values: jnp.ndarray) -> int:
            return jnp.argmax(fitness_values)

        return jax.jit(best_index_fn)

    @staticmethod
    def get_get_best_fitness_jit() -> Callable[[jnp.ndarray], float]:
        """Get a JIT-compiled function to find the best fitness value.

        Returns:
            Callable: A function that takes a JAX array of fitness values and returns the highest value.
        """
        def best_fitness_fn(fitness_values: jnp.ndarray) -> float:
            return jnp.max(fitness_values).astype(jnp.float32)

        return jax.jit(best_fitness_fn)

    # dunder methods for easier access
    def __len__(self) -> int:
        if self._genomes is None:
            return 0
        return self._genomes.shape[0]

    def __getitem__(self, index: int) -> T:
        if self._genomes is None:
            raise IndexError("Population is empty.")
        if abs(index) < 0 or abs(index) >= len(self):
            raise IndexError("Index out of range.")
        return self._genomes[index]

    def __iter__(self):
        if self._genomes is None:
            raise StopIteration
        for i in range(len(self)):
            yield self._genomes[i]
            
    def __str__(self) -> str:
        # print all the genome tensors in the population
        genomes_str = '\n'.join([str(self._genomes[i]) for i in range(len(self._genomes))])
        return f"Population(genome_cls={self._genome_cls.__name__}, pop_size={self._pop_size}, current_size={len(self)}):\n{genomes_str}"
    def __repr__(self) -> str:
        return f"Population(genome_cls={self._genome_cls.__name__}, pop_size={self._pop_size}, current_size={len(self)})"

    # methods to convert population to different formats
    def to_stack(self) -> jnp.ndarray:
        """Get the population's genomes as a JAX array.

        Returns:
            JAX array of the population's genomes.
        """
        return self._genomes

    def to_list(self) -> List[T]:
        """Get the population's genomes as a list of genome instances.

        Returns:
            List of genome instances.
        """
        return [self._genome_cls.from_tensor(tensor, self._genome_init_params) for tensor in self._genomes]

    # setters and getters
    def set_genomes(self, genomes: jnp.ndarray) -> None:
        """Set the population's genomes directly.

        Args:
            genomes: JAX array of genomes to set as the population's genomes.
        """
        if genomes.shape[0] > self._pop_size:
            raise ValueError(f"Number of genomes exceeds population size: {self._pop_size}")
        self._genomes = genomes
        self._best_genome = None

    def get_genomes(self) -> jnp.ndarray:
        """Get the population's genomes.

        Returns:
            JAX array of the population's genomes.
        """
        return self._genomes
    def get_genome_init_params(self) -> Dict:
        """Get the genome initialization parameters.

        Returns:
            Dictionary of genome initialization parameters.
        """
        return self._genome_init_params
    def get_random_key(self) -> Optional[jar.PRNGKey]:
        """Get the current random key.

        Returns:
            JAX PRNGKey used for random operations.
        """
        return self._random_key
    def get_fitness_values(self) -> jnp.ndarray:
        """Get fitness values of all genomes in the population.

        Returns:
            Array of fitness values for each genome.
        """
        if self._fitness_values.shape[0] != self._genomes.shape[0]:
            raise ValueError("Fitness values array size does not match number of genomes.")
        return self._fitness_values
    
    def set_fitness_values(self, fitness_values: jnp.ndarray) -> None:
        """Set fitness values for the population.

        Args:
            fitness_values: JAX array of fitness values to set.
        
        Raises:
            ValueError: If the size of fitness_values does not match the number of genomes.
        """
        if fitness_values.shape[0] != len(self):
            raise ValueError("Fitness values array size does not match number of genomes.")
        self._fitness_values = fitness_values
        self._best_genome = None

    def get_best_genome(self) -> T:
        """Get the best genome in the population.
        
        Returns:
            Genome with the highest fitness value.

        Raises:
            ValueError: If population is empty or fitness values are not set.
        """
        if self._genomes.shape[0] == 0:
            raise ValueError("Population is empty.")
        if self._fitness_values.shape[0] != self._genomes.shape[0]:
            raise ValueError("Fitness values are not set or do not match number of genomes.")

        if self._best_genome is None:
            best_index_fn = self.get_get_best_index_jit()
            best_index = best_index_fn(self._fitness_values)
            self._best_genome = self._genome_cls.from_tensor(self._genomes[best_index], self.context)

        best_genome = self._genome_cls.from_tensor(self._best_genome.to_tensor(), self.context)
        return best_genome, self._fitness_values[best_index]

        
