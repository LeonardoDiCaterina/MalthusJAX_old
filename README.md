# MalthusJAX

A JAX-based framework for evolutionary computation

![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## Key Features

- **JAX-Native**: Built from the ground up with JAX for automatic differentiation, JIT compilation, and GPU acceleration
- **Tensorized Operations**: Vectorized genetic operators for massive parallel processing
- **4-Level Hierarchical Architecture**: Modular design from core components to rich visualizations
- **Research-Ready**: Extensible framework for evolutionary computation research
- **Performance-Optimized**: JIT-compiled evolutionary steps with orders of magnitude speedup

## Project Structure

The MalthusJAX framework is organized into 4 hierarchical levels:

```
src/malthusjax/
├── core/                    # Level 1: Core Components
│   ├── genome/             # Genome representations (Binary, Real, Categorical, Tree)
│   ├── fitness/            # Fitness evaluators
│   └── base.py             # JAXTensorizable, AbstractGenome, AbstractFitnessEvaluator
├── operators/              # Level 2: Genetic Operators
│   ├── selection/          # Selection operators (Tournament, Roulette)
│   ├── crossover/          # Crossover operators (Uniform, Average, Cycle)
│   ├── mutation/           # Mutation operators (BitFlip, Ball, Scramble)
│   └── base.py             # AbstractGeneticOperator, factory pattern
├── engine/                 # Level 3: Evolution Engines
│   ├── base.py             # AbstractEngine, AbstractEngineParams, AbstractEvolutionState
│   ├── basic_engine.py     # GeneticEngine, GeneticEngineParams
│   └── genetic.py          # Legacy implementations
└── visualization/          # Level 4: Rich Analytics & Visualization
    ├── base.py             # AbstractVisualizer, VisualizationConfig
    ├── single_run.py       # EvolutionVisualizer, GeneticAlgorithmVisualizer
    └── multi_run.py        # EngineComparator, FunctionalDataAnalyzer

```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- JAX 0.7+

### Install from source
```bash
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX
pip install -e .
```

### Development installation
```bash
pip install -e ".[dev]"
```

## Quick Start

### Level 1: Basic Genome and Fitness

```python
import jax
import jax.numpy as jnp
import jax.random as jar

from malthusjax.core.genome.binary import BinaryGenome, BinaryGenomeConfig
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator

# Create genome configuration
config = BinaryGenomeConfig(array_shape=(10,), p=0.5)

# Get JIT-compiled initialization function
init_fn = BinaryGenome.get_random_initialization_pure_from_config(config)
jit_init_fn = jax.jit(init_fn)

# Create population
key = jar.PRNGKey(42)
keys = jar.split(key, 100)
population = jax.vmap(jit_init_fn)(keys)

# Evaluate fitness
evaluator = BinarySumFitnessEvaluator()
fitness_values = evaluator.evaluate_batch(population)
print(f"Population fitness: {fitness_values}")
```

### Level 2: Genetic Operators

```python
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.mutation.binary import BitFlipMutation
from malthusjax.operators.crossover.binary import UniformCrossover

# Selection
selector = TournamentSelection(number_of_choices=50, tournament_size=3)
select_fn = jax.jit(selector.get_pure_function())
selected_indices = select_fn(key, fitness_values)

# Mutation  
mutator = BitFlipMutation(mutation_rate=0.1)
mutate_fn = jax.jit(mutator.get_pure_function())
mutated_pop = jax.vmap(mutate_fn, in_axes=(0, 0))(keys, population)

# Crossover
crossover = UniformCrossover(crossover_rate=0.8, n_outputs=2)
cross_fn = jax.jit(crossover.get_pure_function())
```

### Level 3: Complete Evolution Engine

```python
from malthusjax.engine.basic_engine import GeneticEngine, GeneticEngineParams
from malthusjax.core.genome.real import RealGenome
from malthusjax.core.fitness.real import SphereFitnessEvaluator
from malthusjax.operators.selection.tournament import TournamentSelection
from malthusjax.operators.crossover.real import AverageCrossover
from malthusjax.operators.mutation.real import BallMutation

# Configure the evolution engine
engine = GeneticEngine(
    genome_representation=RealGenome(array_shape=(10,), min_val=-5.0, max_val=5.0),
    fitness_evaluator=SphereFitnessEvaluator(),
    selection_operator=TournamentSelection(number_of_choices=198, tournament_size=3),
    crossover_operator=AverageCrossover(blend_rate=0.8, n_outputs=1),
    mutation_operator=BallMutation(mutation_rate=0.1, mutation_strength=0.1)
)

# Configure evolution parameters
params = GeneticEngineParams(
    pop_size=200,
    elitism=2,
    num_generations=100,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=3
)

# Run evolution
key = jar.PRNGKey(42)
initial_state = engine.init_state(key, params)
final_state, history, elapsed_time = engine.run(
    initial_state, 
    params, 
    compile=True, 
    time_it=True
)

print(f"Best fitness: {final_state.best_fitness}")
print(f"Best genome: {final_state.best_genome}")
print(f"Runtime: {elapsed_time:.3f}s")
```

### Level 4: Rich Visualization and Analytics

```python
from malthusjax.visualization import EvolutionVisualizer, VisualizationConfig

# Configure visualization styling
config = VisualizationConfig(
    figsize=(15, 10),
    style='seaborn-v0_8',
    color_palette='husl'
)

# Create comprehensive visualization
visualizer = EvolutionVisualizer(config)
visualizer.plot_evolution_summary(
    history, 
    title="Evolution Analysis",
    save_path="evolution_results.png"
)

# Advanced analytics
from malthusjax.visualization import FunctionalDataAnalyzer
analyzer = FunctionalDataAnalyzer()
convergence_metrics = analyzer.analyze_convergence_patterns([history])
print(f"Convergence rate: {convergence_metrics['convergence_rate']:.3f}")
```

## 🎨 Level 4: Visualization & Analytics

MalthusJAX includes a comprehensive **Level 4 visualization system** for research-grade analysis:

### Core Visualization Components

- **`EvolutionVisualizer`**: Single-run evolution analysis with fitness plots, diversity tracking, and convergence metrics
- **`GeneticAlgorithmVisualizer`**: GA-specific visualizations including selection pressure and genetic diversity
- **`EngineComparator`**: Multi-algorithm comparison with statistical significance testing
- **`FunctionalDataAnalyzer`**: Advanced analytics for convergence patterns and performance characterization

### Key Features

- **Configuration-Driven**: Unified `VisualizationConfig` for consistent styling across all plots
- **Publication-Ready**: High-DPI output with customizable themes for research papers
- **Interactive Analytics**: Rich KPI extraction and statistical analysis
- **Multi-Run Support**: Comparative analysis across different algorithms and parameter sets
- **Extensible Framework**: Abstract base classes for custom visualization development

### Example: Complete Evolution Analysis

```python
from malthusjax.visualization import EvolutionVisualizer, EngineComparator

# Single evolution run analysis
visualizer = EvolutionVisualizer()
visualizer.plot_evolution_summary(history, title="GA Performance")
visualizer.plot_diversity_analysis(history)
visualizer.plot_convergence_metrics(history)

# Multi-algorithm comparison
comparator = EngineComparator()
results = comparator.compare_algorithms([ga_history, de_history, pso_history])
comparator.plot_comparative_analysis(results)
```

## Performance Benchmarks

MalthusJAX achieves significant speedups through JIT compilation:

| Problem Type | Population Size | Generations | Time | Speedup vs Pure Python |
|--------------|-----------------|-------------|------|------------------------|
| Binary (50-bit) | 1000 | 100 | 0.8s | ~50x |
| Real (10D Sphere) | 1000 | 100 | 1.2s | ~40x |
| Real (20D Rastrigin) | 2000 | 100 | 2.1s | ~35x |

## Supported Genome Types

- **BinaryGenome**: Binary string representation for combinatorial optimization
- **RealGenome**: Real-valued vectors for continuous optimization  
- **CategoricalGenome**: Discrete categorical variables
- **PermutationGenome**: Permutation-based representation for ordering problems

## Supported Problem Types

### Fitness Functions
- **Binary Problems**: Binary sum, Knapsack problem
- **Real-Valued**: Sphere, Rastrigin, Rosenbrock, Ackley functions
- **Permutation**: TSP, sorting problems
- **Custom**: Easy to implement your own fitness functions

### Genetic Operators (Level 2)
- **Selection**: Tournament, Roulette wheel, Ranking selection
- **Crossover**: Uniform, Single-point, Average (real), Cycle (permutation)
- **Mutation**: Bit-flip, Ball mutation, Swap, Scramble

### Evolution Engines (Level 3)
- **GeneticEngine**: Production-ready genetic algorithm with rich configuration
- **AbstractEngine**: Base class for custom algorithm implementations
- **State Management**: Type-safe evolution state with comprehensive KPI tracking

### Visualization & Analytics (Level 4)
- **EvolutionVisualizer**: Single-run analysis with fitness evolution and diversity plots
- **GeneticAlgorithmVisualizer**: GA-specific visualization with selection pressure analysis
- **EngineComparator**: Multi-algorithm comparison with statistical significance testing
- **FunctionalDataAnalyzer**: Advanced convergence analysis and performance characterization
- **VisualizationConfig**: Unified configuration system for publication-ready plots

## Examples and Tutorials

Check out the comprehensive examples in the [`examples/`](examples/) directory:

- [`Demo_LV_1.ipynb`](examples/Demo_LV_1.ipynb): Level 1 - Core genomes and fitness functions
- [`Demo_LV_2.ipynb`](examples/Demo_LV_2.ipynb): Level 2 - Genetic operators and population dynamics
- [`Demo_LV_3_Lite.ipynb`](examples/Demo_LV_3_Lite.ipynb): Level 3 - Complete evolution engines and performance analysis
- [`Demo_Level4_Visualization.ipynb`](examples/Demo_Level4_Visualization.ipynb): Level 4 - Rich visualization and comparative analytics

## Research Applications

MalthusJAX is designed for research in:

- **Evolutionary Algorithms**: Genetic algorithms, evolution strategies
- **Genetic Programming**: Tree-based and linear GP systems  
- **Multi-objective Optimization**: Pareto-based evolutionary approaches
- **Neuroevolution**: Evolution of neural network architectures and weights
- **Hyperparameter Optimization**: Evolutionary hyperparameter tuning

## 🧪 Testing & Quality Assurance

MalthusJAX uses comprehensive testing and quality tools:

```bash
# Run complete quality check pipeline
make check-all          # Runs lint, format, type-check, test

# Individual quality checks  
make test               # pytest with 80%+ coverage requirement
make lint               # Ruff linting (replaces flake8, black, isort) 
make format             # Ruff code formatting
make type-check         # mypy strict type checking

# Performance testing
make test-performance   # Benchmark JIT compilation and runtime
```

### Testing Strategy
- **Unit Tests**: Each component thoroughly tested in isolation
- **Integration Tests**: Full evolution loops and engine composition
- **Performance Tests**: JIT compilation and runtime benchmarks
- **Type Safety**: Comprehensive mypy coverage with strict settings
- **Reproducibility**: Deterministic testing with fixed random seeds

### Code Quality Standards
- **80% Test Coverage**: Minimum coverage threshold enforced
- **Type Annotations**: Full type hints for all public interfaces
- **Documentation**: Comprehensive docstrings following NumPy style
- **Code Formatting**: Consistent formatting with Ruff
- **Lint-Free**: Zero warnings policy with comprehensive rules

## 🤝 Contributing

We welcome contributions! MalthusJAX is designed to be extensible and research-friendly.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX
make install-dev        # Install with all dev dependencies

# Run quality checks before submitting
make check-all         # Must pass before PR submission
```

### Contribution Areas
- **New Genome Types**: Extend Level 1 with novel representations
- **Genetic Operators**: Add Level 2 selection, crossover, mutation operators  
- **Evolution Engines**: Implement Level 3 custom algorithms
- **Fitness Functions**: Domain-specific problem implementations
- **Visualization**: Enhanced analytics and plotting capabilities
- **Performance**: GPU/TPU optimizations and large-scale improvements

### Guidelines
- Follow the 4-level architecture principles
- Implement proper JAX/JIT compatibility
- Include comprehensive tests and documentation
- Use factory pattern for all genetic operators
- Extend abstract base classes appropriately

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **JAX Team**: Built on the excellent [JAX](https://github.com/google/jax) library for high-performance computing
- **Research Community**: Inspired by decades of evolutionary computation research  
- **Contributors**: Special thanks to all contributors advancing the framework
- **Educational Support**: Developed with support from academic research institutions

## 📞 Support & Community

- 📧 **Email**: leonardo.dicaterina@mail.polimi.it, 20240485@novaims.unl.pt
- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/LeonardoDiCaterina/MalthusJAX/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/LeonardoDiCaterina/MalthusJAX/discussions)
- 📚 **Documentation**: [Full API Documentation](docs/build/html/index.html)
- 🎓 **Research**: Citation information and academic usage guidelines available

---

**MalthusJAX**: High-performance evolutionary computation at the speed of JAX!

*Evolving solutions through modular composition, JAX optimization, and research-grade extensibility.* 