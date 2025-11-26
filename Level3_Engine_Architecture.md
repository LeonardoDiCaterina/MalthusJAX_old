# MalthusJAX Level 3 Engine Architecture

## Overview

The **Level 3 Engine** is MalthusJAX's evolutionary computation orchestrator that provides a modular, abstract architecture for genetic algorithms. It implements a clean 2-tier system with a universal abstract base class and concrete genetic algorithm implementations optimized for JAX/JIT compilation.

## Architecture Philosophy

### Design Principles

1. **Modularity**: Swappable engine implementations through abstract base classes
2. **Performance**: JIT-compiled JAX operations with `jax.lax.scan` optimization
3. **Abstraction**: Clean separation between interface and implementation
4. **Type Safety**: Flax struct dataclasses for JIT compatibility
5. **Extensibility**: Easy to add new engine implementations

### 3-Level Hierarchical Structure

```
Level 1: Core Components (Genomes, Fitness Evaluators)
    ↓
Level 2: Genetic Operators (Selection, Crossover, Mutation)
    ↓
Level 3: Evolution Engines (Production, Research, Legacy)
```

## Core Architecture Components

### Abstract Base Classes (`base.py`)

#### `AbstractEngineParams`
```python
@flax.struct.dataclass
class AbstractEngineParams:
    pop_size: int = flax.struct.field(pytree_node=False)
    elitism: int = flax.struct.field(pytree_node=False)
    num_generations: int = flax.struct.field(pytree_node=False)
```

**Key Features:**
- **Immutable Configuration**: Uses `@flax.struct.dataclass` with `pytree_node=False`
- **Static Parameters**: All fields marked static for JIT compilation
- **Type Safety**: Proper typing for parameter validation

#### `AbstractEvolutionState`
```python
@flax.struct.dataclass
class AbstractEvolutionState:
    generation: int
    best_fitness: jax.Array
    stagnation_counter: int
    rng_key: jax.Array
```

**Key Features:**
- **JAX Pytree**: Automatic Pytree registration for JIT compatibility
- **Immutable**: All state mutations return new instances
- **Minimal Interface**: Contains only essential fields required by all engines
- **RNG Management**: Proper JAX PRNG key handling

#### `AbstractGenerationOutput`
```python
@flax.struct.dataclass
class AbstractGenerationOutput:
    best_fitness: jax.Array
    mean_fitness: jax.Array
    generation: jax.Array
    
    @classmethod
    def get_kpi_names(cls) -> List[str]:
        return list(cls.__dataclass_fields__.keys())
```

**Key Features:**
- **KPI Foundation**: Base class for generation-wise metrics
- **Dynamic Discovery**: `get_kpi_names()` enables universal visualization
- **Extensible**: Subclasses can add domain-specific metrics

#### `AbstractEngine`
```python
class AbstractEngine(ABC):
    @abstractmethod
    def init_state(self, rng_key: jnp.ndarray, params: AbstractEngineParams) -> AbstractEvolutionState:
        pass
        
    @abstractmethod
    def step(self, key: jnp.ndarray, state: AbstractEvolutionState, 
             params: AbstractEngineParams) -> Tuple[jnp.ndarray, AbstractEvolutionState, AbstractGenerationOutput]:
        pass
        
    def run(self, initial_state: AbstractEvolutionState, params: AbstractEngineParams, 
            time_it: bool = False, compile: bool = True, verbose: bool = False):
        # JAX scan-based evolution loop with compilation caching
        pass
```

**Key Features:**
- **Abstract Interface**: Defines contract all engines must implement
- **JAX Scan Integration**: Built-in scan-based evolution loop
- **Compilation Caching**: Intelligent JIT compilation with parameter-based caching
- **Timing & Monitoring**: Built-in performance measurement and progress tracking

## Engine Implementations

### GeneticEngine (`basic_engine.py`)

**Purpose**: Complete genetic algorithm implementation using the AbstractEngine interface.

#### State Components

##### `GeneticEngineParams`
```python
@flax.struct.dataclass  
class GeneticEngineParams(AbstractEngineParams):
    mutation_rate: float = flax.struct.field(pytree_node=False, default=0.01)
    crossover_rate: float = flax.struct.field(pytree_node=False, default=0.8)
    tournament_size: int = flax.struct.field(pytree_node=False, default=3)
```

##### `GeneticEvolutionState`
```python
@flax.struct.dataclass  
class GeneticEvolutionState(AbstractEvolutionState):
    current_population: jnp.ndarray
    current_fitness: jnp.ndarray
    best_genome: jnp.ndarray
    ema_delta_fitness: jnp.ndarray
```

##### `GeneticGenerationOutput`
```python
@flax.struct.dataclass  
class GeneticGenerationOutput(AbstractGenerationOutput):
    std_fitness: jnp.ndarray
    min_fitness: jnp.ndarray
    best_genome: jnp.ndarray
    ema_delta_fitness: jnp.ndarray
```

#### Implementation Features

**Constructor**:
```python
class GeneticEngine(AbstractEngine):
    def __init__(self,
                 genome_representation: AbstractGenome,
                 fitness_evaluator: AbstractFitnessEvaluator,
                 selection_operator: AbstractSelectionOperator,
                 crossover_operator: AbstractCrossover,
                 mutation_operator: AbstractMutation):
        # Stores Level 1 & 2 components
        # Extracts pure functions for JIT compilation
```

**Evolution Step**:
Implements standard genetic algorithm pipeline:
1. **Elitism**: Preserve best individuals
2. **Selection**: Tournament selection for parent pairs
3. **Crossover**: Generate offspring with crossover operator
4. **Mutation**: Apply mutations to offspring
5. **Evaluation**: Fitness assessment of new population
6. **Metrics**: Update convergence and diversity tracking

**Key Characteristics:**
- ✅ **Modular Composition**: Uses Level 1 & 2 components
- ✅ **JIT Optimized**: All operations vectorized and JIT-compiled
- ✅ **Rich Metrics**: Comprehensive generation-wise analytics
- ✅ **Elitism Support**: Configurable elite preservation
- ✅ **Convergence Tracking**: EMA-based improvement monitoring

**Use Cases:**
- Standard genetic algorithm applications
- Binary and real-valued optimization
- Research and educational demonstrations
- Production genetic algorithm deployments

## Usage Examples

### Direct Engine Instantiation

```python
from malthusjax.engine.basic_engine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.binary import UniformCrossover
from malthusjax.operators.mutation.binary import BitFlipMutation

# Create genetic algorithm engine
engine = GeneticEngine(
    genome_representation=BinaryGenome(array_shape=(100,), p=0.1),
    fitness_evaluator=BinarySumFitnessEvaluator(),
    selection_operator=TournamentSelection(number_of_choices=80, tournament_size=3),
    crossover_operator=UniformCrossover(crossover_rate=0.8, n_outputs=1),
    mutation_operator=BitFlipMutation(mutation_rate=0.02)
)

# Configure parameters
params = GeneticEngineParams(
    pop_size=100,
    elitism=3,
    num_generations=200,
    mutation_rate=0.02,
    crossover_rate=0.8,
    tournament_size=3
)

# Initialize and run evolution
rng_key = jar.PRNGKey(42)
initial_state = engine.init_state(rng_key, params)
final_state, history, elapsed_time = engine.run(
    initial_state, 
    params, 
    compile=True, 
    time_it=True, 
    verbose=True
)

print(f"Best fitness: {final_state.best_fitness}")
print(f"Evolution time: {elapsed_time:.3f}s")
```

### Using Built-in Compilation Caching

```python
# Pre-compile for multiple runs (recommended for experiments)
engine.compile_evolution(params)  # Compile once

# Multiple runs use cached compilation
results = []
for seed in range(10):
    rng = jar.PRNGKey(seed)
    state = engine.init_state(rng, params)
    final_state, history, _ = engine.run(state, params, compile=True)
    results.append(final_state.best_fitness)
    
print(f"Mean performance: {jnp.mean(jnp.array(results))}")
```

## State System Hierarchy

```
AbstractEvolutionState (base interface)
└── GeneticEvolutionState (genetic algorithm implementation)

AbstractGenerationOutput (base KPI interface)
└── GeneticGenerationOutput (genetic algorithm metrics)

AbstractEngineParams (base configuration)
└── GeneticEngineParams (genetic algorithm parameters)
```

### State Transitions

The engine follows a clean state transition pattern:

```python
# Generation Loop (via jax.lax.scan)
initial_state → step_fn → new_state → step_fn → ... → final_state
               ↓        ↓           ↓
           kpi_output₁ kpi_output₂  ...
```

**GeneticEngine**: Collects comprehensive generation metrics in `GeneticGenerationOutput`

## JAX Integration & Performance

### JIT Compilation Strategy

1. **Level 1**: Individual functions (fitness evaluation, genome initialization)
2. **Level 2**: Operator functions (selection, crossover, mutation) 
3. **Level 3**: Complete evolution step and scan-based evolution loop

### Pure Function Architecture

GeneticEngine extracts pure functions from Level 1 & 2 components:

```python
class GeneticEngine(AbstractEngine):
    def __init__(self, genome_representation, fitness_evaluator, 
                 selection_operator, crossover_operator, mutation_operator):
        # Extract pure functions for JIT compilation
        self.init_genome_fn = self.genome.get_random_initialization_pure()
        self.evaluate_fn = self.fitness.get_pure_fitness_function()
        self.select_fn = self.selection.get_pure_function()
        self.crossover_fn = self.crossover.get_pure_function()
        self.mutation_fn = self.mutation.get_pure_function()
```

### Scan-Based Evolution with Compilation Caching

The AbstractEngine provides intelligent compilation caching:

```python
def run(self, initial_state, params, compile=True, time_it=False, verbose=False):
    if compile:
        # Check if compiled function exists for these parameters
        if self.is_compiled(params):
            evolution_fn = self._compiled_evolution_fn  # Use cached
        else:
            self.compile_evolution(params)  # Compile and cache
            evolution_fn = self._compiled_evolution_fn
    
    # Execute with JAX scan
    (final_key, final_state), history = evolution_fn(init_carry)
```

## File Structure

```
src/malthusjax/engine/
├── __init__.py          # Empty module file
├── base.py              # AbstractEngine, AbstractEngineParams, AbstractEvolutionState
├── basic_engine.py      # GeneticEngine implementation
└── genetic.py           # Empty (placeholder)
```

## Extension Points

The abstract architecture enables easy extension:

### Custom Engine Implementation
```python
class MyCustomEngine(AbstractEngine):
    def init_state(self, rng_key, params):
        # Custom initialization logic
        return MyCustomEvolutionState(...)
    
    def step(self, key, state, params):
        # Custom evolution step
        return population, new_state, metrics
        
    # Inherits run() method with scan-based execution and compilation caching
```

### Custom Parameter and State Classes
```python
@flax.struct.dataclass
class MyCustomEngineParams(AbstractEngineParams):
    custom_param: float = flax.struct.field(pytree_node=False, default=1.0)

@flax.struct.dataclass
class MyCustomEvolutionState(AbstractEvolutionState):
    custom_state: jax.Array
    
@flax.struct.dataclass
class MyCustomGenerationOutput(AbstractGenerationOutput):
    custom_metric: jax.Array
```

## Conclusion

The Level 3 Engine architecture provides a solid foundation for evolutionary computation in MalthusJAX:

- **Clean Abstraction**: Well-defined AbstractEngine interface with concrete GeneticEngine implementation
- **JAX Native**: Full JIT compilation with intelligent compilation caching
- **Performance Optimized**: Scan-based evolution loop with static parameter compilation
- **Extensible**: Easy to add new engine implementations through abstract base classes
- **Type Safe**: Flax struct dataclasses ensure JIT compatibility
- **Rich Metrics**: Comprehensive KPI tracking with universal visualization support

This architecture successfully provides both high-performance evolutionary computation and a clean foundation for extending MalthusJAX with new algorithm implementations.