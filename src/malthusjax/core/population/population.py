# Path: src/malthusjax/core/population/population.py

from typing import List, Dict, Any, Generic, TypeVar, Optional
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

from malthusjax.core.base import SerializationContext
from malthusjax.core.genome.base import AbstractGenome
from malthusjax.core.population.base import AbstractPopulation
#from malthusjax.core.fitness.base import AbstractFitnessEvaluator

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from malthusjax.core.fitness.base import AbstractFitnessEvaluator

T = TypeVar('T', bound=AbstractGenome)

class Population(AbstractPopulation[T], Generic[T]):
    """Standard population implementation for genetic algorithms."""

    def __init__(self, genome_cls: type[T], pop_size: int, random_key: Optional[jar.PRNGKey] = None, context: Optional[SerializationContext] = None, random_init: bool = True, genome_init_params: Dict = {}, fitness_transform = None) -> None:
        """Initialize the population with a genome class and maximum size.

        Args:
            genome_cls: Class of genomes to be managed in the population.
            pop_size: Maximum number of genomes in the population.
            random_key: Optional JAX random key for random initialization.
            random_init: Whether to randomly initialize the population.
        """
        self._genomes:jnp.ndarray = jnp.empty((0,), dtype= int)
        self._fitness_values: jnp.ndarray = jnp.array([])
        self.genome_init_params = genome_init_params
        self._random_key = random_key
        super().__init__(genome_cls=genome_cls, pop_size=pop_size, random_key=random_key, context=context, random_init=random_init, genome_init_params=genome_init_params, fitness_transform=fitness_transform)
        self._best_genome: Optional[T] = None

    def add_genome(self, genome: T) -> None:
        """Add a genome to the population.

        Args:
            genome: Genome to add to the population.
        
        Raises:
            ValueError: If population is already at max capacity.
        """
        if self._genomes is None:
            self._genomes = []
        elif len(self._genomes) >= self._pop_size:
            raise ValueError(f"Population already at maximum capacity: {self._pop_size}")

        self._genomes.append(genome)
        self._best_genome = None

    def get_genomes(self) -> List[T]:
        """Get all genomes in the population.

        Returns:
            List of all genomes currently in the population.
        """
        return self._genomes

    def get_fitness_values(self) -> jnp.ndarray:
        """Get fitness values of all genomes in the population.

        Returns:
            Array of fitness values for each genome.
        """
        if self._genomes is None or len(self._genomes) == 0:
            return jnp.array([])

        return self._fitness_values

    def get_best_genome(self) -> T:
        """Get the best solution in the population.
        
        Returns:
            Solution with the highest fitness value.
            
        Raises:
            ValueError: If population is empty.
        """
        if self._genomes is None or len(self._genomes) == 0:
            raise ValueError("Cannot get best solution from empty population")
        
        if len(self._genomes) != len(self._fitness_values):
            raise ValueError("Fitness values are not up to date with genomes")
        
        if self._best_genome is None:
            best_idx = jnp.argmax(self._fitness_values)
            best_genome = self._genomes[int(best_idx)]
            self._best_genome = self._genome_cls.from_tensor(best_genome, self.get_genome_init_params())
            self._best_genome._fitness = self._fitness_values[int(best_idx)]

        return self._best_genome
    
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
        if not self._genomes:
            return {
                'pop_size': 0,
                'max_fitness': None,
                'min_fitness': None,
                'avg_fitness': None,
                'fitness_std': None
            }
            
        fitness_values = jnp.array([s.fitness for s in self._solutions])
        
        return {
            'pop_size': len(self._genomes),
            'max_fitness': float(jnp.max(fitness_values)),
            'min_fitness': float(jnp.min(fitness_values)),
            'avg_fitness': float(jnp.mean(fitness_values)),
            'fitness_std': float(jnp.std(fitness_values)),
            '25th_percentile': float(jnp.percentile(fitness_values, 25)),
            '50th_percentile': float(jnp.percentile(fitness_values, 50)),
            '75th_percentile': float(jnp.percentile(fitness_values, 75))
        }

    def update_fitness(self, fitness_evaluator: 'AbstractFitnessEvaluator') -> None:
        """Update fitness of all solutions in the population.
        
        Args:
            fitness_evaluator: Fitness evaluator to use for updating fitness values.
        """
        if not self._genomes:
            return

        fitness_evaluator.evaluate_solutions(self._genomes)

        # Update best solution
        self._best_solution = sorted(self._genomes, reverse=True)[0]

    @property
    def size(self) -> int:
        """Get the current population size.
        
        Returns:
            Current number of solutions in the population.
        """
        if self._genomes is None:
            return 0
        return len(self._genomes)
    

    
    