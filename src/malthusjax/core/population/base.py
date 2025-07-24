from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any, Generic, TypeVar

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore
import jax # type: ignore
from malthusjax.core.base import SerializationContext
from malthusjax.core.solution.base import AbstractSolution

T = TypeVar('T', bound=AbstractSolution)

class AbstractPopulation(ABC, Generic[T]):
    """Abstract base class for population management in genetic algorithms."""
    
    def __init__(self,solution_class: type[T], max_size: int, random_key: Optional[jar.PRNGKey] = None, context = None, random_init: bool = True, genome_init_params: Dict = {}, fitness_transform = None) -> None:
        """Initialize population with maximum size."""
        self._best_solution = None
        self.solution_class = solution_class
        self.max_size = max_size
        self.context = context
        self.genome_init_params = genome_init_params
        self.fitness_transform = fitness_transform
        self.random_key = random_key #if random_key is not None else jar.PRNGKey(0)
        print(f"genome_init_params{genome_init_params}")
        if random_init:
            self._random_init( genome_init_params, random_key=self.random_key, fitness_transform = fitness_transform)
    
    def generate_random_keys(self, n_keys: int) -> jnp.ndarray:
        """Generate an array of JAX random keys.
        
        Args:
            n_keys: Number of random keys to generate.
            base_key: Base key to use for generation. If None, uses the population's random key.
                If population key is also None, uses a default seed.
                
        Returns:
            Array of n_keys random keys.
        """
                
        # Split the base key into n_keys + 1 keys
        keys = jar.split(self.random_key, n_keys + 1)
        
        # Store the last key as the new population key for future use
        self.random_key = keys[-1]
        
        # Return the first n_keys
        return keys[:-1]
    
    def __len__(self) -> int:
        """Get the number of solutions in the population."""
        return self.size
    def __iter__(self):
        """Iterate over the solutions in the population."""
        return iter(self.get_solutions())
    def __next__(self):
        """Get the next solution in the population."""
        solutions = self.get_solutions()
        if not solutions:
            raise StopIteration
        return next(iter(solutions))
    
    def validate(self) -> bool:
        """Validate all solutions in the population."""
        solutions = self.get_solutions()
        if not solutions:
            return False
        return bool(jnp.all(jnp.array([solution.is_valid for solution in solutions])))
    
    def __getitem__(self, index: int) -> T:
        """Get a solution by index."""
        solutions = self.get_solutions()
        if index < 0 or index >= len(solutions):
            raise IndexError("Index out of range")
        return solutions[index]
    
    def __contains__(self, solution: T ) -> bool:
        """Check if a solution is in the population."""
        return solution in self.get_solutions()
    
    def _random_init(self, genome_init_params: Dict, random_key: Optional[jar.PRNGKey] = None, fitness_transform = None ) -> None:
        """Randomly initialize the population."""
        if random_key is None:
            random_key = self.random_key
        keys = self.generate_random_keys(self.max_size)
        
        for i in range(self.max_size):
            solution = self.solution_class(random_key=keys[i],
                                           random_init=True,
                                          genome_init_params = genome_init_params,
                                          fitness_transform = fitness_transform)
            self.add_solution(solution)
    
    @abstractmethod
    def add_solution(self, solution: T) -> None:
        """Add a solution to the population."""
        pass
    
    @abstractmethod
    def get_solutions(self) -> List[T]:
        """Get all solutions in the population."""
        pass

    @abstractmethod
    def get_fitness_values(self) -> jnp.ndarray:
        """Get fitness values of all solutions in the population."""
        pass

    @abstractmethod
    def get_best_solution(self) -> T:
        """Get the best solution in the population."""
        pass
    
    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics."""
        pass
    
    @abstractmethod
    def update_fitness(self, fitness_evaluator) -> None:
        """Update fitness of all solutions in the population."""
        pass
    
    @property
    @abstractmethod
    def size(self) -> int:
        """Get the current population size."""
        pass
    
    def from_stack (self,
                    stack: jnp.ndarray,
                    context: SerializationContext = None,
                    fitness_values: Optional[jnp.ndarray] = None,
                    ) -> "AbstractPopulation" :
        """Create a new population from a stacked array representation.
    
        Args:
            stack: JAX array containing stacked genome data for multiple solutions.
        
        Returns:
            A new Population instance with solutions created from the stack.
        """
        if fitness_values is not None and fitness_values.shape[0] != stack.shape[0]:
            raise ValueError("fitness_values must match the number of solutions in stack")
                
        class_type = type(self)
        
        size_of_stack = stack.shape[0]
        random_keys =  self.generate_random_keys(size_of_stack +1)
        
        new_population = class_type(
            solution_class= self.solution_class,
            max_size= size_of_stack,
            random_init = False,
            random_key = random_keys[-1]
        )
        solutions0 = self.solution_class(genome_init_params = self.genome_init_params, random_init=False)
        for i in range(size_of_stack):
            tensor = stack[i]
            random_key = random_keys[i]
            solution = solutions0.from_tensor(tensor = tensor,genome_init_params = context.genome_init_params)
            if fitness_values is not None:
                solution.raw_fitness = fitness_values[i]
            new_population.add_solution(solution)
        return new_population
    
    def from_solution_list (self,
                            solutions: List[T],
                            context: SerializationContext = None,
                            fitness_values: Optional[jnp.ndarray] = None) -> "AbstractPopulation":
        """Create a new population from a list of solutions.
        Args:
            solutions: List of solutions to initialize the population.
            context: Serialization context for genome initialization parameters.
            fitness_values: Optional array of fitness values corresponding to the solutions.
        Returns:
            A new Population instance with solutions created from the list.
        """ 
        if fitness_values is not None and len(fitness_values) != len(solutions):
            raise ValueError("fitness_values must match the number of solutions in solutions list")
        
        if context is None:
            context = self.context
        
        class_type = type(self)
        size_of_solutions = len(solutions)
        random_keys = self.generate_random_keys(1)  
        new_population = class_type(
            solution_class=self.solution_class,
            max_size=self.max_size,
            context=context,
            random_init=False,
            random_key=random_keys[0]
        )
        new_population._solutions = solutions
        if fitness_values is not None:
            if len(fitness_values) != size_of_solutions:
                raise ValueError("fitness_values must match the number of solutions in solutions list")
            for i, solution in enumerate(solutions):
                if not isinstance(solution, self.solution_class):
                    raise TypeError(f"Expected solution of type {self.solution_class}, got {type(solution)}")
                new_population[i].raw_fitness = fitness_values[i]
        
        return new_population

    def to_stack(self) -> jnp.ndarray:
        """Stack all genomes in the population into a single JAX array.

        Returns:
            JAX array where each row corresponds to the genome of a solution.
        """
        if not self._solutions:
            return jnp.array([])

        genomes = [s.to_tensor() for s in self._solutions]
        return jnp.stack(genomes)