"""
MalthusJAX: A JAX-based evolutionary computation framework.
"""

__version__ = "0.2.0"

# === CLEAN API: LEVEL 1 COMPONENTS ===
# Expose core genomes at top level for clean imports
from .core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig, BinaryPopulation
from .core.genome.real_genome import RealGenome, RealGenomeConfig, RealPopulation
from .core.genome.categorical_genome import CategoricalGenome, CategoricalGenomeConfig, CategoricalPopulation
from .core.genome.linear import LinearGenome, LinearGenomeConfig, LinearPopulation

# Fitness evaluators - organized by genome type
from .core.fitness.binary_evaluators import BinarySumEvaluator, BinarySumConfig, KnapsackEvaluator, KnapsackConfig
from .core.fitness.real_evaluators import SphereEvaluator, SphereConfig, GriewankEvaluator, GriewankConfig
from .core.fitness.linear_gp_evaluator import LinearGPEvaluator

# === CLEAN API: LEVEL 2 OPERATORS ===
# Namespace organization eliminates "import hell"

class mutation:
    """Mutation operators organized by genome type."""
    # Binary mutations
    from .operators.mutation.binary import BitFlipMutation as BitFlip
    from .operators.mutation.binary import ScrambleMutation as Scramble
    
    # Real mutations  
    from .operators.mutation.real import GaussianMutation as Gaussian
    from .operators.mutation.real import BallMutation as Ball
    from .operators.mutation.real import PolynomialMutation as Polynomial
    
    # Categorical mutations
    from .operators.mutation.categorical import CategoricalFlipMutation as CategoricalFlip
    from .operators.mutation.categorical import RandomCategoryMutation as RandomCategory
    
    # Linear GP mutations
    from .operators.mutation.linear import LinearMutation as Linear
    from .operators.mutation.linear import LinearPointMutation as LinearPoint

class crossover:
    """Crossover operators organized by genome type."""
    # Binary crossover
    from .operators.crossover.binary import UniformCrossover as Uniform
    from .operators.crossover.binary import SinglePointCrossover as SinglePoint
    
    # Real crossover
    from .operators.crossover.real import BlendCrossover as Blend
    from .operators.crossover.real import SimulatedBinaryCrossover as SBX
    
    # Linear GP crossover
    from .operators.crossover.linear import LinearCrossover as Linear

class selection:
    """Selection operators (genome-agnostic)."""
    from .operators.selection.tournament import TournamentSelection as Tournament
    from .operators.selection.roulette import RouletteWheelSelection as Roulette

# === CLEAN USAGE EXAMPLES ===
"""
Example of the new clean API:

import malthusjax as mjx

# Level 1: Clean genome and fitness setup
config = mjx.BinaryGenomeConfig(length=100)
evaluator = mjx.BinarySumEvaluator(mjx.BinarySumConfig(maximize=True))
population = mjx.BinaryPopulation.init_random(key, config, size=50)

# Level 2: Clean operator access
mutator = mjx.mutation.BitFlip(num_offspring=1, mutation_rate=0.01)  
crossover = mjx.crossover.Uniform(num_offspring=2, crossover_rate=0.7)
selector = mjx.selection.Tournament(num_selections=30, tournament_size=3)

# Level 3: Clean evolution composition (coming soon)
engine = mjx.EvolutionEngine(mutator=mutator, crossover=crossover, selector=selector)
result = engine.evolve(population, evaluator, num_generations=100)
"""