# MalthusJAX

A JAX-based framework for evolutionary computation with emphasis on high-performance, tensorized operations and research-grade experimentation.

![JAX](https://img.shields.io/badge/JAX-0.4+-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![License](https://img.shields.io/badge/License-MIT-red.svg)

## Key Features

- **JAX-Native**: Built from the ground up with JAX for automatic differentiation, JIT compilation, and GPU acceleration
- **Tensorized Operations**: Vectorized genetic operators for massive parallel processing
- **Modular Architecture**: 3-level hierarchical design from core genomes to complete evolutionary systems
- **Research-Ready**: Extensible framework for evolutionary computation research
- **Performance-Optimized**: JIT-compiled evolutionary steps with orders of magnitude speedup

## Project Structure

The MalthusJAX framework is organized into 6 hierarchical levels:

```
src/malthusjax/
├── core/                    # Level 1: Core Components
│   ├── genome/             # Genome representations (Binary, Real, Categorical)
│   ├── fitness/            # Fitness evaluators
│   └── base.py             # Base abstractions
├── operators/              # Level 2: Genetic Operators
│   ├── selection/          # Selection operators (Tournament, Roulette)
│   ├── crossover/          # Crossover operators
│   ├── mutation/           # Mutation operators
│   └── base.py             # Operator abstractions
└── engine/                 # Level 3: Evolution Engines
    ├── BasicMalthusEngine.py
    └── state.py            # Evolution state management

```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- JAX 0.7+

### Install from source
```bash
git clone https://github.com/yourusername/MalthusJAX.git
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
from malthusjax.engine.BasicMalthusEngine import BasicMalthusEngine
from malthusjax.core.genome.real import RealGenome
from malthusjax.core.fitness.real import SphereFitnessEvaluator

# Configure the evolution engine
engine = BasicMalthusEngine(
    genome_representation=RealGenome(array_shape=(10,), min_val=-5.0, max_val=5.0),
    fitness_evaluator=SphereFitnessEvaluator(),
    selection_operator=TournamentSelection(100, 3),
    crossover_operator=AverageCrossover(0.8, 1),
    mutation_operator=BallMutation(0.1, 0.1),
    elitism=2
)

# Run evolution
key = jar.PRNGKey(42)
final_state, history = engine.run(
    key, 
    num_generations=100, 
    pop_size=200
)

print(f"Best fitness: {final_state.best_fitness}")
print(f"Best genome: {final_state.best_genome}")
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

### Genetic Operators
- **Selection**: Tournament, Roulette wheel
- **Crossover**: Uniform, Single-point, Average (real), Cycle (permutation)
- **Mutation**: Bit-flip, Ball mutation, Swap, Scramble

## Examples and Tutorials

Check out the comprehensive examples in the [`examples/`](examples/) directory:

- [`Demo_LV_1.ipynb`](examples/Demo_LV_1.ipynb): Core genomes and fitness functions
- [`Demo_LV_2.ipynb`](examples/Demo_LV_2.ipynb): Genetic operators and population dynamics
- [`Demo_LV_3.ipynb`](examples/Demo_LV_3.ipynb): Complete evolution engines and performance analysis
- [`demos/`](examples/demos/): Detailed component demonstrations

## Research Applications

MalthusJAX is designed for research in:

- **Evolutionary Algorithms**: Genetic algorithms, evolution strategies
- **Genetic Programming**: Tree-based and linear GP systems  
- **Multi-objective Optimization**: Pareto-based evolutionary approaches
- **Neuroevolution**: Evolution of neural network architectures and weights
- **Hyperparameter Optimization**: Evolutionary hyperparameter tuning

## Testing

Run the test suite:

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run linting
make lint

# Run type checking  
make type-check
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/LeonardoDiCaterina/MalthusJAX.git
cd MalthusJAX

# Install in development mode
make install-dev

# Run quality checks
make check-all
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use MalthusJAX in your research, please cite:

```bibtex
@software{malthusjax2024,
  title={MalthusJAX: A JAX-based Framework for Evolutionary Computation},
  author={Leonardo DiCaterina},
  year={2024},
  url={https://github.com/yourusername/MalthusJAX}
}
```

## Acknowledgments

- Built on top of the excellent [JAX](https://github.com/google/jax) library
- Inspired by modern evolutionary computation research
- Special thanks to the JAX development team for the foundational framework

## Support

- 📧 Email: [leonardo.dicaterina@mail.polimi.it], [20240485@novaims.unl.pt]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/MalthusJAX/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/MalthusJAX/discussions)

---

**MalthusJAX**: Evolving solutions at the speed of JAX! 