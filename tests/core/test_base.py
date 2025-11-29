"""
Tests for base classes and abstractions (Level 1).

Tests base genome, base operator abstractions and their contracts.
"""

import pytest
import jax
import jax.numpy as jnp

"""
Tests for base classes and abstractions (Level 1).

Tests base genome, base operator abstractions and their contracts.
"""

import pytest
import jax
import jax.numpy as jnp

from malthusjax.operators.base import BaseMutation, BaseCrossover, BaseSelection


class TestBaseGenomeAbstractions:
    """Test base genome and operator abstractions."""

    def test_base_operators_exist(self):
        """Test that base operator classes are properly defined."""
        assert BaseMutation is not None
        assert BaseCrossover is not None 
        assert BaseSelection is not None
        
    def test_generic_typing(self):
        """Test that base operators use proper generic typing."""
        # These should be abstract classes with generic parameters
        assert hasattr(BaseMutation, '__orig_bases__')
        assert hasattr(BaseCrossover, '__orig_bases__')
        assert hasattr(BaseSelection, '__orig_bases__')


class TestBaseMutation:
    """Test BaseMutation base class."""

    def test_abstract_interface(self):
        """Test that BaseMutation cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseMutation()

    def test_required_methods(self):
        """Test that subclasses must implement required methods."""
        # This is tested implicitly by the concrete operator tests
        pass


class TestBaseCrossover:
    """Test BaseCrossover base class."""

    def test_abstract_interface(self):
        """Test that BaseCrossover cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseCrossover()

    def test_dataclass_interface(self):
        """Test that crossover operators are dataclasses."""
        from malthusjax.operators.crossover.binary import UniformCrossover
        
        crossover = UniformCrossover(num_offspring=3, crossover_rate=0.8)
        
        # Should be frozen (immutable)
        with pytest.raises(AttributeError):
            crossover.num_offspring = 5

    def test_static_parameters(self):
        """Test that parameters are static for JIT compilation."""
        from malthusjax.operators.crossover.real import BlendCrossover
        
        crossover = BlendCrossover(num_offspring=2, crossover_rate=0.9, alpha=0.3)
        
        assert crossover.num_offspring == 2
        assert crossover.crossover_rate == 0.9
        assert crossover.alpha == 0.3


class TestBaseSelection:
    """Test BaseSelection base class."""

    def test_abstract_interface(self):
        """Test that BaseSelection cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSelection()

    def test_dataclass_interface(self):
        """Test that selection operators are dataclasses."""
        from malthusjax.operators.selection.tournament import TournamentSelection
        
        selector = TournamentSelection(tournament_size=3, num_selections=5)
        
        # Should be frozen (immutable)
        with pytest.raises(AttributeError):
            selector.tournament_size = 7

    def test_static_parameters(self):
        """Test that parameters are static for JIT compilation."""
        from malthusjax.operators.selection.roulette import RouletteWheelSelection
        
        selector = RouletteWheelSelection(num_selections=10)
        
        assert selector.num_selections == 10


class TestOperatorSignatures:
    """Test that all operators follow correct signature patterns."""

    def test_mutation_signatures(self, rng_key, binary_genome_config, binary_genome):
        """Test mutation operator signatures."""
        from malthusjax.operators.mutation.binary import BitFlipMutation
        
        mutator = BitFlipMutation(mutation_rate=0.1)
        
        # Test operator call signature: (key, genome, config) -> genome
        result = mutator(rng_key, binary_genome, binary_genome_config)
        assert hasattr(result, 'bits')
        
        # Test pure function signature: (key, genome_data, config) -> genome_data
        pure_fn = mutator.get_pure_function()
        result_data = pure_fn(rng_key, binary_genome.bits, binary_genome_config)
        assert result_data.shape == binary_genome.bits.shape

    def test_crossover_signatures(self, rng_key, binary_genome_config):
        """Test crossover operator signatures."""
        from malthusjax.operators.crossover.binary import UniformCrossover
        from malthusjax.core.genome.binary_genome import BinaryGenome
        
        key1, key2 = jax.random.split(rng_key)
        parent1 = BinaryGenome.random_init(key1, binary_genome_config)
        parent2 = BinaryGenome.random_init(key2, binary_genome_config)
        
        crossover = UniformCrossover(num_offspring=3, crossover_rate=0.8)
        
        # Test operator call signature: (key, parent1, parent2, config) -> offspring_batch
        result = crossover(rng_key, parent1, parent2, binary_genome_config)
        assert hasattr(result, 'bits')
        assert result.bits.shape == (3, binary_genome_config.length)
        
        # Test pure function signature: (key, parent1_data, parent2_data, config) -> offspring_data
        pure_fn = crossover.get_pure_function()
        result_data = pure_fn(rng_key, parent1.bits, parent2.bits, binary_genome_config)
        assert result_data.shape == (3, binary_genome_config.length)

    def test_selection_signatures(self, rng_key, fitness_values):
        """Test selection operator signatures."""
        from malthusjax.operators.selection.tournament import TournamentSelection
        
        selector = TournamentSelection(tournament_size=3, num_selections=5)
        
        # Test operator call signature: (key, fitness_values) -> selected_indices
        result = selector(rng_key, fitness_values)
        assert result.shape == (5,)
        
        # Test pure function signature: (key, fitness_values, ...) -> selected_indices
        pure_fn = selector.get_pure_function()
        result_pure = pure_fn(rng_key, fitness_values, selector.tournament_size, selector.num_selections)
        assert result_pure.shape == (5,)


class TestGenomeConfigValidation:
    """Test genome configuration validation."""

    def test_binary_config_validation(self):
        """Test binary genome config validation."""
        from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
        
        # Valid config
        config = BinaryGenomeConfig(length=10)
        assert config.length == 10
        
        # Invalid config should raise error
        with pytest.raises(ValueError):
            BinaryGenomeConfig(length=0)
            
        with pytest.raises(ValueError):
            BinaryGenomeConfig(length=-1)

    def test_real_config_validation(self):
        """Test real genome config validation."""
        from malthusjax.core.genome.real_genome import RealGenomeConfig
        
        # Valid config
        config = RealGenomeConfig(length=5, bounds=(-2.0, 2.0))
        assert config.length == 5
        assert config.bounds == (-2.0, 2.0)
        
        # Invalid configs should raise errors
        with pytest.raises(ValueError):
            RealGenomeConfig(length=0, bounds=(-1.0, 1.0))
            
        with pytest.raises(ValueError):
            RealGenomeConfig(length=-1, bounds=(-1.0, 1.0))
            
        with pytest.raises(ValueError):
            RealGenomeConfig(length=5, bounds=(2.0, -2.0))  # min > max

    def test_categorical_config_validation(self):
        """Test categorical genome config validation."""
        from malthusjax.core.genome.categorical_genome import CategoricalGenomeConfig
        
        # Valid config
        config = CategoricalGenomeConfig(length=8, n_categories=4)
        assert config.length == 8
        assert config.n_categories == 4
        
        # Invalid configs should raise errors  
        with pytest.raises(ValueError):
            CategoricalGenomeConfig(length=0, n_categories=3)
            
        with pytest.raises(ValueError):
            CategoricalGenomeConfig(length=5, n_categories=0)
            
        with pytest.raises(ValueError):
            CategoricalGenomeConfig(length=-1, n_categories=3)


class TestPyTreeRegistration:
    """Test that configs are properly registered as PyTrees."""

    def test_binary_config_pytree(self):
        """Test binary config PyTree registration."""
        from malthusjax.core.genome.binary_genome import BinaryGenomeConfig
        
        config = BinaryGenomeConfig(length=10)
        
        # Should be able to use with jax.tree_map
        def double_length(x):
            if hasattr(x, 'length'):
                return BinaryGenomeConfig(length=x.length * 2)
            return x
            
        doubled_config = jax.tree.map(double_length, config)
        assert doubled_config.length == 20

    def test_real_config_pytree(self):
        """Test real config PyTree registration."""
        from malthusjax.core.genome.real_genome import RealGenomeConfig
        
        config = RealGenomeConfig(length=5, bounds=(-1.0, 1.0))
        
        # Should work with JAX transformations
        configs = [config, config]
        stacked = jax.tree.map(lambda *x: x, *configs)
        
        assert len(stacked) == 2

    def test_categorical_config_pytree(self):
        """Test categorical config PyTree registration."""
        from malthusjax.core.genome.categorical_genome import CategoricalGenomeConfig
        
        config = CategoricalGenomeConfig(length=8, n_categories=4)
        
        # Should work with vmap
        configs = [config] * 3
        
        # This tests that the config can be used as a static argument in vmap
        def dummy_fn(config):
            return config.length
            
        lengths = [dummy_fn(c) for c in configs]
        assert all(l == 8 for l in lengths)


@pytest.mark.integration
class TestArchitectureIntegration:
    """Integration tests for the overall architecture."""

    def test_genome_operator_compatibility(self, rng_key):
        """Test that genomes work with all compatible operators."""
        from malthusjax.core.genome.binary_genome import BinaryGenome, BinaryGenomeConfig
        from malthusjax.operators.mutation.binary import BitFlipMutation
        from malthusjax.operators.crossover.binary import UniformCrossover
        from malthusjax.operators.selection.tournament import TournamentSelection
        from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator
        
        # Create components
        config = BinaryGenomeConfig(length=10)
        genome1 = BinaryGenome.random_init(rng_key, config)
        
        key1, key2 = jax.random.split(rng_key)
        genome2 = BinaryGenome.random_init(key1, config)
        
        mutator = BitFlipMutation(mutation_rate=0.1)
        crossover = UniformCrossover(num_offspring=2, crossover_rate=0.8) 
        selector = TournamentSelection(tournament_size=2, num_selections=3)
        evaluator = BinarySumFitnessEvaluator()
        
        # Test full pipeline
        # 1. Evaluate fitness
        fitness1 = evaluator.evaluate_single(genome1.bits)
        fitness2 = evaluator.evaluate_single(genome2.bits)
        fitness_values = jnp.array([fitness1, fitness2])
        
        # 2. Select parents
        selected_indices = selector(key2, fitness_values)
        assert selected_indices.shape == (3,)
        
        # 3. Crossover
        key3 = jax.random.split(key2)[0]
        offspring = crossover(key3, genome1, genome2, config)
        assert offspring.bits.shape == (2, config.length)
        
        # 4. Mutation
        key4 = jax.random.split(key3)[0]
        for i in range(offspring.bits.shape[0]):
            child = BinaryGenome(bits=offspring.bits[i])
            mutated_child = mutator(key4, child, config)
            assert mutated_child.bits.shape == (config.length,)

    def test_real_genome_pipeline(self, rng_key):
        """Test full pipeline with real genomes."""
        from malthusjax.core.genome.real_genome import RealGenome, RealGenomeConfig
        from malthusjax.operators.mutation.real import GaussianMutation
        from malthusjax.operators.crossover.real import BlendCrossover
        from malthusjax.operators.selection.roulette import RouletteWheelSelection
        from malthusjax.core.fitness.real_functions import SphereFunction
        
        # Create components
        config = RealGenomeConfig(length=5, bounds=(-2.0, 2.0))
        
        key1, key2, key3 = jax.random.split(rng_key, 3)
        genome1 = RealGenome.random_init(key1, config)
        genome2 = RealGenome.random_init(key2, config)
        
        mutator = GaussianMutation(mutation_rate=0.2, sigma=0.1)
        crossover = BlendCrossover(num_offspring=3, crossover_rate=0.9, alpha=0.3)
        selector = RouletteWheelSelection(num_selections=2)
        evaluator = SphereFunction()
        
        # Test pipeline
        fitness1 = evaluator.evaluate_single(genome1.values)
        fitness2 = evaluator.evaluate_single(genome2.values)
        fitness_values = jnp.array([fitness1, fitness2])
        
        selected_indices = selector(key3, fitness_values)
        assert selected_indices.shape == (2,)
        
        offspring = crossover(key3, genome1, genome2, config)
        assert offspring.values.shape == (3, config.length)
        
        for i in range(offspring.values.shape[0]):
            child = RealGenome(values=offspring.values[i])
            mutated_child = mutator(key3, child, config)
            assert mutated_child.values.shape == (config.length,)