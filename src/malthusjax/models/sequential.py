from typing import List, Optional, Dict, Any, Union, Callable
import jax
import jax.numpy as jnp
import jax.random as jar
from dataclasses import dataclass
import time
from abc import ABC, abstractmethod



from malthusjax.core.solution.base import AbstractSolution
from malthusjax.core.population.population import Population
from malthusjax.core.fitness.base_old import AbstractFitnessEvaluator
from malthusjax.operators.base import AbstractGeneticOperator
from malthusjax.models.callbacks import AbstractCallback


@dataclass
class EvolutionaryHistory:
    """History of evolutionary training."""
    best_fitness: List[float]
    average_fitness: List[float]
    generation_times: List[float]
    population_sizes: List[int]
    best_solutions: List[AbstractSolution]
    
@dataclass
class ModelConfig:
    """Configuration for the GeneticSequential model."""
    population_size: int = 100
    max_generations: int = 1000
    random_seed: int = 42
    verbose: int = 1
    early_stopping_patience: Optional[int] = None
    fitness_threshold: Optional[float] = None


class GeneticSequential:
    """
    A Keras-like sequential model for evolutionary algorithms.
    
    This class allows you to stack evolutionary operators sequentially
    and provides a clean interface for training evolutionary algorithms.
    
    Example:
        ```python
        model = GeneticSequential(name='my_optimizer')
        model.add(TournamentSelection(tournament_size=5))
        model.add(UniformCrossover(crossover_rate=0.7))
        model.add(BitFlipMutation(mutation_rate=0.01))
        
        model.build(solution_class=BinarySolution, genome_params={'array_size': 50, 'p': 0.5})
        model.compile(fitness_evaluator=BinarySumFitnessEvaluator())
        
        history = model.fit(generations=100, population_size=100)
        ```
    """
    
    def __init__(self, name: str = "GeneticSequential", config: Optional[ModelConfig] = None):
        self.name = name
        self.config = config or ModelConfig()
        self.operators: List[AbstractGeneticOperator] = []
        self.fitness_evaluator: Optional[AbstractFitnessEvaluator] = None
        self.solution_class: Optional[type] = None
        self.genome_params: Optional[Dict[str, Any]] = None
        self.population: Optional[Population] = None
        self.is_built = False
        self.is_compiled = False
        self.random_key = jar.PRNGKey(self.config.random_seed)
        self.callbacks: List[AbstractCallback] = []
        
        # Performance tracking
        self._compile_time: Optional[float] = None
        self._build_time: Optional[float] = None

    def add(self, operator: AbstractGeneticOperator) -> 'GeneticSequential':
        """
        Add an operator to the model.
        
        Args:
            operator: An evolutionary operator (selection, crossover, mutation, etc.)
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If model is already built
        """
        if self.is_built:
            raise ValueError("Cannot add operators after model is built. Create a new model.")

        if not isinstance(operator, AbstractGeneticOperator):
            raise TypeError(f"Expected AbstractGeneticOperator, got {type(operator).__name__}")

        self.operators.append(operator)
        
        if self.config.verbose > 1:
            print(f"Added {operator.__class__.__name__} to {self.name}")
            
        return self
    
    def build(self, 
              solution_class: type, 
              genome_params: Dict[str, Any],
              population_size: Optional[int] = None) -> 'GeneticSequential':
        """
        Build the model with specified solution class and genome parameters.
        
        Args:
            solution_class: The solution class to use
            genome_params: Parameters for genome initialization
            population_size: Override default population size
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If no operators have been added
        """
        start_time = time.time()
        
        if not self.operators:
            raise ValueError("Cannot build model without operators. Use add() to add operators first.")
            
        if not issubclass(solution_class, AbstractSolution):
            raise TypeError(f"solution_class must be a subclass of AbstractSolution")
            
        self.solution_class = solution_class
        self.genome_params = genome_params.copy()
        
        if population_size:
            self.config.population_size = population_size
            
        # Build operators
        for operator in self.operators:
            operator.build(solution_class=solution_class, genome_params=genome_params)
            
        self.is_built = True
        self._build_time = time.time() - start_time
        
        if self.config.verbose > 0:
            print(f"Model '{self.name}' built successfully in {self._build_time:.3f}s")
            print(f"  - Solution class: {solution_class.__name__}")
            print(f"  - Genome params: {genome_params}")
            print(f"  - Population size: {self.config.population_size}")
            print(f"  - Operators: {len(self.operators)}")
            
        return self
    
    def compile(self, 
                fitness_evaluator: AbstractFitnessEvaluator,
                callbacks: Optional[List[AbstractCallback]] = None) -> 'GeneticSequential':
        """
        Compile the model with a fitness evaluator.
        
        Args:
            fitness_evaluator: The fitness evaluator to use
            callbacks: Optional list of callbacks
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If model is not built
        """
        if not self.is_built:
            raise ValueError("Model must be built before compilation. Use build() first.")
            
        start_time = time.time()
        
        self.fitness_evaluator = fitness_evaluator
        self.callbacks = callbacks or []
        
        # Initialize population
        self.population = Population(
            solution_class=self.solution_class,
            max_size=self.config.population_size,
            random_key=self.random_key,
            random_init=True,
            genome_init_params=self.genome_params
        )
        
        # Initial fitness evaluation
        self.fitness_evaluator.evaluate_solutions(self.population.get_solutions())
        
        self.is_compiled = True
        self._compile_time = time.time() - start_time
        
        if self.config.verbose > 0:
            print(f"Model compiled successfully in {self._compile_time:.3f}s")
            print(f"  - Fitness evaluator: {fitness_evaluator.__class__.__name__}")
            print(f"  - Initial population created and evaluated")
            if self.callbacks:
                print(f"  - Callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")
                
        return self
    
    def fit(self, 
            generations: Optional[int] = None,
            population_size: Optional[int] = None,
            callbacks: Optional[List[AbstractCallback]] = None,
            verbose: Optional[int] = None) -> EvolutionaryHistory:
        """
        Train the evolutionary algorithm.
        
        Args:
            generations: Number of generations to run (overrides config)
            population_size: Population size (overrides config)
            callbacks: Additional callbacks for this run
            verbose: Verbosity level (overrides config)
            
        Returns:
            EvolutionaryHistory object containing training history
            
        Raises:
            ValueError: If model is not compiled
        """
        if not self.is_compiled:
            raise ValueError("Model must be compiled before training. Use compile() first.")
            
        # Override config if provided
        max_generations = generations or self.config.max_generations
        if population_size and population_size != self.config.population_size:
            raise ValueError("Cannot change population size during fit. Rebuild model instead.")
            
        if verbose is not None:
            original_verbose = self.config.verbose
            self.config.verbose = verbose
        else:
            original_verbose = None
            
        # Combine callbacks
        all_callbacks = self.callbacks + (callbacks or [])
        
        # Initialize history
        history = EvolutionaryHistory()
        
        # Initialize callbacks
        for callback in all_callbacks:
            callback.on_train_begin(self, history)
            
        start_time = time.time()
        
        try:
            for generation in range(max_generations):
                generation_start = time.time()
                
                # Callback: on_generation_begin
                for callback in all_callbacks:
                    callback.on_generation_begin(generation, self, history)
                
                # Apply operators sequentially
                current_population = self.population
                
                for i, operator in enumerate(self.operators):
                    if self.config.verbose > 2:
                        print(f"  Applying {operator.__class__.__name__}...")
                        
                    # Apply operator
                    current_population = operator.apply(current_population, self.random_key)
                    
                    # Update random key
                    self.random_key, _ = jar.split(self.random_key)
                
                # Update population
                self.population = current_population
                
                # Evaluate fitness
                self.fitness_evaluator.evaluate_solutions(self.population.get_solutions())
                
                # Record statistics
                stats = self.population.get_statistics()
                history.record_generation(
                    generation=generation,
                    max_fitness=stats['max_fitness'],
                    min_fitness=stats['min_fitness'],
                    avg_fitness=stats['avg_fitness'],
                    fitness_std=stats['fitness_std'],
                    best_solution=self.population.get_best_solution(),
                    generation_time=time.time() - generation_start
                )
                
                # Callback: on_generation_end
                for callback in all_callbacks:
                    if callback.on_generation_end(generation, self, history):
                        if self.config.verbose > 0:
                            print(f"Early stopping triggered by {callback.__class__.__name__}")
                        break
                else:
                    # Continue if no callback requested early stopping
                    continue
                break
                
                # Progress reporting
                if self.config.verbose > 0 and (generation + 1) % 10 == 0:
                    best_fitness = stats['max_fitness']
                    avg_fitness = stats['avg_fitness']
                    print(f"Generation {generation + 1:4d}: "
                          f"Best={best_fitness:.4f}, "
                          f"Avg={avg_fitness:.4f}, "
                          f"Time={time.time() - generation_start:.3f}s")
                    
        except KeyboardInterrupt:
            if self.config.verbose > 0:
                print(f"\nTraining interrupted at generation {generation}")
                
        finally:
            # Restore original verbose setting
            if original_verbose is not None:
                self.config.verbose = original_verbose
                
            # Callback: on_train_end
            for callback in all_callbacks:
                callback.on_train_end(self, history)
                
            # Finalize history
            history.finalize(total_time=time.time() - start_time)
            
            if self.config.verbose > 0:
                summary = history.get_summary()
                print(f"\nTraining completed:")
                print(f"  - Total generations: {summary['total_generations']}")
                print(f"  - Best fitness: {summary['best_fitness']:.4f}")
                print(f"  - Total time: {summary['total_time']:.2f}s")
                print(f"  - Avg time per generation: {summary['avg_generation_time']:.3f}s")
                
        return history
    
    def predict(self, population: Optional[Population] = None) -> Population:
        """
        Apply the evolutionary operators to a population without fitness evaluation.
        
        Args:
            population: Population to process (uses model's population if None)
            
        Returns:
            Processed population
        """
        if not self.is_compiled:
            raise ValueError("Model must be compiled before prediction.")
            
        target_population = population or self.population
        current_population = target_population
        
        for operator in self.operators:
            current_population = operator.apply(current_population, self.random_key)
            self.random_key, _ = jar.split(self.random_key)
            
        return current_population
    
    def evaluate(self, population: Optional[Population] = None) -> Dict[str, float]:
        """
        Evaluate a population and return statistics.
        
        Args:
            population: Population to evaluate (uses model's population if None)
            
        Returns:
            Dictionary of fitness statistics
        """
        if not self.is_compiled:
            raise ValueError("Model must be compiled before evaluation.")
            
        target_population = population or self.population
        self.fitness_evaluator.evaluate_solutions(target_population.get_solutions())
        
        return target_population.get_statistics()
    
    def get_best_solution(self):
        """Get the best solution from the current population."""
        if not self.population:
            raise ValueError("No population available. Train the model first.")
        return self.population.get_best_solution()
    
    def summary(self):
        """Print a summary of the model architecture."""
        print(f"Model: {self.name}")
        print("=" * 50)
        
        if not self.is_built:
            print("Model not built yet.")
            return
            
        print(f"Solution Class: {self.solution_class.__name__}")
        print(f"Genome Params: {self.genome_params}")
        print(f"Population Size: {self.config.population_size}")
        print(f"Built: {self.is_built}, Compiled: {self.is_compiled}")
        
        if self._build_time:
            print(f"Build Time: {self._build_time:.3f}s")
        if self._compile_time:
            print(f"Compile Time: {self._compile_time:.3f}s")
            
        print("\nOperator Stack:")
        print("-" * 30)
        
        for i, operator in enumerate(self.operators):
            print(f"{i+1:2d}. {operator.__class__.__name__}")
            if hasattr(operator, 'get_config'):
                config = operator.get_config()
                config_str = ", ".join([f"{k}={v}" for k, v in config.items()])
                print(f"    Config: {config_str}")
                
        if self.fitness_evaluator:
            print(f"\nFitness Evaluator: {self.fitness_evaluator.__class__.__name__}")
            
        if self.callbacks:
            print(f"Callbacks: {[cb.__class__.__name__ for cb in self.callbacks]}")
            
        print("=" * 50)
    
    def save_weights(self, filepath: str):
        """Save the current population to a file."""
        if not self.population:
            raise ValueError("No population to save.")
            
        # Implementation depends on your serialization strategy
        raise NotImplementedError("Weight saving not implemented yet")
    
    def load_weights(self, filepath: str):
        """Load population from a file."""
        raise NotImplementedError("Weight loading not implemented yet")
    
    def clone(self) -> 'GeneticSequential':
        """Create a copy of this model."""
        new_model = GeneticSequential(name=f"{self.name}_copy", config=self.config)
        
        # Copy operators (they should be immutable or cloneable)
        for operator in self.operators:
            new_model.add(operator)
            
        if self.is_built:
            new_model.build(self.solution_class, self.genome_params)
            
        if self.is_compiled:
            new_model.compile(self.fitness_evaluator, self.callbacks)
            
        return new_model
    
    def __repr__(self):
        return f"GeneticSequential(name='{self.name}', operators={len(self.operators)}, built={self.is_built}, compiled={self.is_compiled})"