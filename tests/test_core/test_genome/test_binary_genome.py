"""
Tests for the BinaryGenome class in MalthusJAX.
"""

import pytest
import jax.numpy as jnp # type: ignore
from jax import random as jar # type: ignore

from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.base import Compatibility, ProblemTypes


class TestBinaryGenome:
    def test_initialization_default(self):
        genome = BinaryGenome(array_size=5, p=0.5)
        assert genome.array_size == 5
        assert genome.p == 0.5
        assert not hasattr(genome, 'genome') or not hasattr(genome, 'genome')  # Shouldn't be initialized yet
        
    def test_initialization_invalid_p(self):
        with pytest.raises(AssertionError):
            BinaryGenome(array_size=5, p=1.5)
        with pytest.raises(AssertionError):
            BinaryGenome(array_size=5, p=-0.1)
    
    def test_random_init(self):
        genome = BinaryGenome(array_size=100, p=0.3, random_init=True, random_key=jar.PRNGKey(42))
        assert len(genome) == 100
        assert jnp.issubdtype(genome.genome.dtype, jnp.bool_)
        # Check approximate proportion of 1s
        proportion_ones = jnp.mean(genome.genome)
        assert jnp.isclose(proportion_ones, 0.3, atol=0.1)

    def test_random_init_reproducible(self):
        key = jar.PRNGKey(42)
        genome1 = BinaryGenome(array_size=100, p=0.3, random_init=True, random_key=key)
        genome2 = BinaryGenome(array_size=100, p=0.3, random_init=True, random_key=key)
        assert jnp.array_equal(genome1.genome, genome2.genome)
    
    def test_len(self):
        genome = BinaryGenome(array_size=50, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        # Test length without initialization
        assert len(genome) == 50
    
    def test_clone(self):
        genome = BinaryGenome(array_size=10, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        clone = genome.clone()
        assert clone is not genome
        assert jnp.array_equal(clone.genome, genome.genome)
        assert clone.array_size == genome.array_size
        assert clone.p == genome.p
    def test_mutate(self):
        genome = BinaryGenome(array_size=100, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        original_genome = genome.genome.copy()
        
        # Test mutation with low rate
        mutated = genome.mutate(rate=0.1, random_key=jar.PRNGKey(123))
        
        # Should return a new genome
        assert mutated is not genome
        assert mutated.array_size == genome.array_size
        assert mutated.p == genome.p
        
        # Some bits should have changed (with high probability)
        changes = jnp.sum(original_genome != mutated.genome)
        assert changes > 0  # Very likely with 100 bits and 10% mutation rate
    
    def test_crossover(self):
        parent1 = BinaryGenome(array_size=20, p=0.2, random_init=True, random_key=jar.PRNGKey(42))
        parent2 = BinaryGenome(array_size=20, p=0.8, random_init=True, random_key=jar.PRNGKey(123))
        
        # Test crossover
        child1, child2 = parent1.crossover(parent2, random_key=jar.PRNGKey(456))
        
        # Children should have same size as parents
        assert len(child1) == 20
        assert len(child2) == 20
        assert child1.array_size == 20
        assert child2.array_size == 20
        
        # Children should be different objects
        assert child1 is not parent1
        assert child2 is not parent2
        assert child1 is not child2
    
    '''def test_compatibility(self):
        genome = BinaryGenome(array_size=10, p=0.5)
        
        # Test compatibility with different problem types
        assert genome.get_compatibility(ProblemTypes.BINARY) == Compatibility.NATIVE
        assert genome.get_compatibility(ProblemTypes.CONTINUOUS) == Compatibility.INCOMPATIBLE
        assert genome.get_compatibility(ProblemTypes.DISCRETE) == Compatibility.COMPATIBLE'''
    
    def test_edge_cases(self):
        # Test with p=0 (all zeros)
        genome_zeros = BinaryGenome(array_size=10, p=0.0, random_init=True, random_key=jar.PRNGKey(42))
        assert jnp.all(genome_zeros.genome == False)
        
        # Test with p=1 (all ones)
        genome_ones = BinaryGenome(array_size=10, p=1.0, random_init=True, random_key=jar.PRNGKey(42))
        assert jnp.all(genome_ones.genome == True)
        
    def test_len(self):
        genome = BinaryGenome(array_size=50, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        # Test length without initialization
        assert len(genome) == 50
    
    def test_size_and_shape_properties(self):
        """Test size and shape properties from AbstractGenome."""
        genome = BinaryGenome(array_size=15, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        assert genome.size == 15
        assert genome.shape == (15,)
        
        # Test that size equals len
        assert genome.size == len(genome)
    
    def test_metadata_property(self):
        """Test metadata getter/setter from AbstractGenome."""
        genome = BinaryGenome(array_size=5, p=0.5, custom_param="test_value")
        
        # Check initial metadata
        assert "custom_param" in genome.metadata
        assert genome.metadata["custom_param"] == "test_value"
        
        # Test metadata setter
        new_metadata = {"new_key": "new_value", "number": 42}
        genome.metadata = new_metadata
        assert genome.metadata == new_metadata
    
    def test_compatibility_property(self):
        """Test compatibility getter/setter from AbstractGenome."""
        genome = BinaryGenome(array_size=5, p=0.5)
        
        # Check default compatibility
        assert genome.compatibility is not None
        assert genome.compatibility.problem_type == ProblemTypes.DISCRETE_OPTIMIZATION
        
        # Test compatibility setter
        new_compat = Compatibility(problem_type=ProblemTypes.BINARY)
        genome.compatibility = new_compat
        assert genome.compatibility == new_compat
    
    def test_is_valid_property_and_invalidate(self):
        """Test validation system from AbstractGenome."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        
        # Should be valid after proper initialization
        assert genome.is_valid == True
        
        # Test invalidation
        genome.invalidate()
        # Should recalculate validity
        assert genome.is_valid == True
        
        # Test with invalid genome (manually corrupt)
        genome.genome = jnp.array([0, 1, 2, 1, 0])  # 2 is invalid for binary
        genome.invalidate()
        assert genome.is_valid == False
    
    def test_tensor_conversion_methods(self):
        """Test to_tensor and from_tensor methods."""
        original = BinaryGenome(array_size=8, p=0.3, random_init=True, random_key=jar.PRNGKey(42))
        
        # Test to_tensor
        tensor = original.to_tensor()
        assert tensor.dtype == jnp.int32
        assert tensor.shape == (8,)
        assert jnp.all((tensor == 0) | (tensor == 1))
        
        # Test from_tensor
        context = original.get_serialization_context()
        reconstructed = BinaryGenome.from_tensor(
            tensor, 
            genome_init_params=context.genome_init_params
        )
        
        assert jnp.array_equal(original.genome, reconstructed.genome)
        assert original.array_size == reconstructed.array_size
        assert original.p == reconstructed.p
    
        '''    def test_serialization_context(self):
        """Test get_serialization_context method."""
        genome = BinaryGenome(array_size=10, p=0.7, custom_data="test")
        context = genome.get_serialization_context()
        
        assert context.genome_class == BinaryGenome
        assert context.genome_init_params['array_size'] == 10
        assert context.genome_init_params['p'] == 0.7
        assert 'custom_data' in context.__dict__'''
    
    def test_distance_method(self):
        """Test distance calculation between genomes."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(123))
        
        # Distance should be non-negative
        distance = genome1.distance(genome2)
        assert distance >= 0
        
        # Distance to self should be 0
        assert genome1.distance(genome1) == 0
        
        # Distance should be symmetric
        assert genome1.distance(genome2) == genome2.distance(genome1)
        
        # Test distance with different sizes (should be inf)
        genome3 = BinaryGenome(array_size=10, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        assert genome1.distance(genome3) == float('inf')
        
        # Test distance with non-BinaryGenome (should be inf)
        assert genome1.distance("not_a_genome") == float('inf')
    
    def test_semantic_key(self):
        """Test semantic key generation."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        
        # Same random key should produce same genome and same semantic key
        assert genome1.semantic_key() == genome2.semantic_key()
        
        # Different genomes should have different keys
        genome3 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(123))
        assert genome1.semantic_key() != genome3.semantic_key()
        
        # Key should be consistent
        key1 = genome1.semantic_key()
        key2 = genome1.semantic_key()
        assert key1 == key2
    
    def test_jax_tree_operations(self):
        """Test JAX tree flatten/unflatten operations."""
        original = BinaryGenome(array_size=6, p=0.4, random_init=True, random_key=jar.PRNGKey(42))
        original.metadata = {"test_key": "test_value"}
        
        # Test tree_flatten
        children, aux_data = original.tree_flatten()
        assert len(children) == 1  # Should contain the genome array
        assert jnp.array_equal(children[0], original.genome)
        assert aux_data['array_size'] == 6
        assert aux_data['p'] == 0.4
        
        # Test tree_unflatten
        reconstructed = BinaryGenome.tree_unflatten(aux_data, children)
        assert jnp.array_equal(original.genome, reconstructed.genome)
        assert original.array_size == reconstructed.array_size
        assert original.p == reconstructed.p
    
    def test_update_from_tensor(self):
        """Test update_from_tensor method."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        original_genome = genome.genome.copy()
        
        # Create new tensor
        new_tensor = jnp.array([True, False, True, True, False])
        
        # Update genome
        genome.update_from_tensor(new_tensor, validate=True)
        assert jnp.array_equal(genome.genome, new_tensor)
        assert not jnp.array_equal(genome.genome, original_genome)
        
        # Test with wrong shape
        wrong_tensor = jnp.array([True, False])
        with pytest.raises(ValueError, match="incompatible"):
            genome.update_from_tensor(wrong_tensor)
    
    def test_equality_and_hashing(self):
        """Test __eq__, __ne__, and __hash__ methods."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        genome3 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(123))
        
        # Same random key should produce equal genomes
        assert genome1 == genome2
        assert not (genome1 != genome2)
        
        # Different genomes should not be equal
        assert genome1 != genome3
        assert not (genome1 == genome3)
        
        # Test with non-genome object
        assert genome1 != "not_a_genome"
        assert genome1 != 42
        
        # Test hashing
        assert hash(genome1) == hash(genome2)  # Same semantic key
        genome_set = {genome1, genome2, genome3}
        assert len(genome_set) == 2  # genome1 and genome2 should be deduplicated
    
    def test_subtraction_operator(self):
        """Test __sub__ operator for distance calculation."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(42))
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=jar.PRNGKey(123))
        
        # Test subtraction operator
        distance = genome1 - genome2
        assert distance == genome1.distance(genome2)
        assert distance >= 0
        
        # Test with invalid operand
        with pytest.raises(TypeError):
            genome1 - "not_a_genome"
    
    def test_jit_methods(self):
        """Test JIT-compiled methods."""
        # Test random initialization JIT
        init_params = {'array_size': 10, 'p': 0.3}
        init_fn = BinaryGenome.get_random_initialization_jit(init_params)
        
        key = jar.PRNGKey(42)
        result = init_fn(key)
        assert result.shape == (10,)
        assert jnp.issubdtype(result.dtype, jnp.bool_)
        
        # Test distance JIT
        distance_fn = BinaryGenome.get_distance_jit()
        sol1 = jnp.array([True, False, True, False])
        sol2 = jnp.array([False, False, True, True])
        distance = distance_fn(sol1, sol2)
        assert distance == 2  # Should be Hamming distance
        
        # Test autocorrection JIT
        correction_fn = BinaryGenome.get_autocorrection_jit(init_params)
        invalid_sol = jnp.array([2, -1, 0.5, 1, 0, 1, 1, 0, 1, 1, 1])  # Too long with invalid values
        corrected = correction_fn(invalid_sol).astype(jnp.int32)
        assert corrected.shape == (10,)  # Should be truncated
        assert jnp.all((corrected == 0) | (corrected == 1))  # Should be valid binary
         