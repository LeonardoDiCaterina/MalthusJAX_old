# Path: src/malthusjax/core/population/population.py

from typing import List, Dict, Any, Generic, TypeVar, Optional
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

from malthusjax.core.solution.base import AbstractSolution
from malthusjax.core.population.base import AbstractPopulation

T = TypeVar('T', bound=AbstractSolution)

class Population(AbstractPopulation[T], Generic[T]):
    """Standard population implementation for genetic algorithms."""
    
    def __init__(self, solution_class: type[T], max_size: int, random_key: Optional[jar.PRNGKey] = None, random_init: bool = True, genome_init_params: Dict = {}, fitness_transform = None) -> None:
        """Initialize the population with a solution class and maximum size.
        
        Args:
            solution_class: Class of solutions to be managed in the population.
            max_size: Maximum number of solutions in the population.
            random_key: Optional JAX random key for random initialization.
            random_init: Whether to randomly initialize the population.
        """
        self._solutions: List[T] = []
        self.genome_init_params = genome_init_params
        self._random_key = random_key,
        super().__init__(solution_class = solution_class, max_size = max_size, random_key = random_key, random_init = random_init, genome_init_params = genome_init_params, fitness_transform = fitness_transform)
        self._best_solution: Optional[T] = None

            
    def add_solution(self, solution: T) -> None:
        """Add a solution to the population.
        
        Args:
            solution: Solution to add to the population.
            
        Raises:
            ValueError: If population is already at max capacity.
        """
        if len(self._solutions) >= self.max_size:
            raise ValueError(f"Population already at maximum capacity: {self.max_size}")
        
        self._solutions.append(solution)
        self._best_solution = None

    def get_solutions(self) -> List[T]:
        """Get all solutions in the population.
        
        Returns:
            List of all solutions currently in the population.
        """
        return self._solutions
    
    def get_fitness_values(self) -> jnp.ndarray:
        """Get fitness values of all solutions in the population.
        
        Returns:
            Array of fitness values for each solution.
        """
        if not self._solutions:
            return jnp.array([])
        
        return jnp.array([s.fitness for s in self._solutions])
    
    def get_best_solution(self) -> T:
        """Get the best solution in the population.
        
        Returns:
            Solution with the highest fitness value.
            
        Raises:
            ValueError: If population is empty.
        """
        if not self._solutions:
            raise ValueError("Cannot get best solution from empty population")
        
        if self._best_solution is None:
            # This should only happen if solutions were added without fitness values
            # Sort and return the best
            return sorted(self._solutions, reverse=True)[0]
        
        return self._best_solution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get population statistics.
        
        Returns:
            Dictionary containing population statistics:
            - 'pop_size': Current population size
            - 'max_fitness': Fitness of best solution
            - 'min_fitness': Fitness of worst solution
            - 'avg_fitness': Average fitness of population
            - 'fitness_std': Standard deviation of fitness values
            -
        """
        if not self._solutions:
            return {
                'pop_size': 0,
                'max_fitness': None,
                'min_fitness': None,
                'avg_fitness': None,
                'fitness_std': None
            }
            
        fitness_values = jnp.array([s.fitness for s in self._solutions])
        
        return {
            'pop_size': len(self._solutions),
            'max_fitness': float(jnp.max(fitness_values)),
            'min_fitness': float(jnp.min(fitness_values)),
            'avg_fitness': float(jnp.mean(fitness_values)),
            'fitness_std': float(jnp.std(fitness_values)),
            '25th_percentile': float(jnp.percentile(fitness_values, 25)),
            '50th_percentile': float(jnp.percentile(fitness_values, 50)),
            '75th_percentile': float(jnp.percentile(fitness_values, 75))
        }
    
    def update_fitness(self, fitness_evaluator) -> None:
        """Update fitness of all solutions in the population.
        
        Args:
            fitness_evaluator: Fitness evaluator to use for updating fitness values.
        """
        if not self._solutions:
            return
        
        fitness_evaluator.evaluate_solutions(self._solutions)
        
        # Update best solution
        self._best_solution = sorted(self._solutions, reverse=True)[0]
    
    @property
    def size(self) -> int:
        """Get the current population size.
        
        Returns:
            Current number of solutions in the population.
        """
        return len(self._solutions)
    

    
    