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
        """Test initialization with required parameters."""
        genome = BinaryGenome(array_size=5, p=0.5)
        assert genome.array_size == 5
        assert genome.p == 0.5
        assert not hasattr(genome, 'genome') or not hasattr(genome, 'genome')  # Shouldn't be initialized yet
        assert isinstance(genome.compatibility, Compatibility)
        assert genome.compatibility.problem_type == ProblemTypes.DISCRETE_OPTIMIZATION
        
    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        genome = BinaryGenome(array_size=10, p=0.3)
        assert genome.array_size == 10
        assert genome.p == 0.3
        
    def test_initialization_invalid_p(self):
        """Test initialization with invalid probability."""
        with pytest.raises(AssertionError):
            BinaryGenome(array_size=5, p=1.5)  # p > 1
            
        with pytest.raises(AssertionError):
            BinaryGenome(array_size=5, p=-0.1)  # p < 0
        
    def test_random_init(self):
        """Test random initialization."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=42)
        assert hasattr(genome, 'genome')
        assert genome.genome.shape == (5,)
        assert genome.is_valid
        
    def test_random_init_reproducible(self):
        """Test that random initialization is reproducible with same key."""
        genome1 = BinaryGenome(array_size=10, p=0.3, random_init=True, random_key=42)
        genome2 = BinaryGenome(array_size=10, p=0.3, random_init=True, random_key=42)

        assert jnp.array_equal(genome1.genome, genome2.genome)  # Should be equal

    def test_validate_valid_genome(self):
        """Test validation with valid genome."""
        genome = BinaryGenome(array_size=5, p=0.5)
        genome.genome = jnp.array([0, 1, 1, 0, 1])
        genome._validate()
        assert genome.is_valid is True
        
    def test_validate_invalid_genome_values(self):
        """Test validation with invalid genome values."""
        genome = BinaryGenome(array_size=5, p=0.5)
        genome.genome = jnp.array([0, 2, 1, 0, 1])  # Contains 2, which is invalid
        assert genome._validate() is False
        assert genome.is_valid is False
        
    def test_validate_invalid_genome_shape(self):
        """Test validation with invalid genome shape."""
        genome = BinaryGenome(array_size=5, p=0.5)
        genome.genome = jnp.array([0, 1, 1])  # Wrong size
        assert genome._validate() is False
        
    def test_validate_missing_genome(self):
        """Test validation when genome is not set."""
        genome = BinaryGenome(array_size=5, p=0.5)
        assert genome._validate() is False
        
    def test_distance_calculation(self):
        """Test distance calculation between genomes."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome2.genome = jnp.array([1, 1, 0, 0, 1])
        
        # Hamming distance should be 2 (different at positions 0 and 2)
        assert genome1.distance(genome2) == 2.0
        
        # Identical genomes should have distance 0
        assert genome1.distance(genome1) == 0.0
        
    def test_distance_incompatible_types(self):
        """Test distance calculation with incompatible genome types."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        # Test with non-BinaryGenome
        assert genome1.distance("not a genome") == float('inf')
        
    def test_distance_incompatible_sizes(self):
        """Test distance calculation with incompatible sizes."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome2 = BinaryGenome(array_size=3, p=0.5, random_init=False)
        genome2.genome = jnp.array([1, 1, 0])
        
        assert genome1.distance(genome2) == float('inf')
            
    def test_semantic_key(self):
        """Test semantic key generation."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome.genome = jnp.array([0, 1, 1, 0, 1])
        
        # The semantic key should be consistent
        key1 = genome.semantic_key()
        key2 = genome.semantic_key()
        assert key1 == key2
        assert isinstance(key1, str)
        
        # Different genomes should have different keys
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome2.genome = jnp.array([1, 0, 0, 1, 0])
        assert genome.semantic_key() != genome2.semantic_key()
        
    def test_to_tensor(self):
        """Test conversion to tensor."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome.genome = jnp.array([0, 1, 1, 0, 1], dtype=jnp.bool_)
        
        tensor = genome.to_tensor()
        assert tensor.dtype == jnp.int32
        assert jnp.array_equal(tensor, jnp.array([0, 1, 1, 0, 1], dtype=jnp.int32))
        
    def test_from_tensor(self):
        """Test creation from tensor."""
        tensor = jnp.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 1])
        genome_init_params = {'array_size': 10, 'p': 0.5}
        
        genome = BinaryGenome.from_tensor(tensor, genome_init_params=genome_init_params)
        assert genome.array_size == 10
        assert jnp.array_equal(genome.to_tensor(), tensor)
        assert genome.is_valid
        
            
    def test_get_serialization_context(self):
        """Test serialization context generation."""
        genome = BinaryGenome(array_size=8, p=0.3)
        context = genome.get_serialization_context()
        
        assert context.genome_class == BinaryGenome
        assert context.genome_init_params['array_size'] == 8
        assert context.genome_init_params['p'] == 0.3
        assert context.compatibility == genome.compatibility
        
    def test_clone_shallow(self):
        """Test shallow copy."""
        original = BinaryGenome(array_size=5, p=0.3, random_init=True, random_key=42)
        original._metadata = {"test": "metadata"}
        
        copy = original.clone(deep=False)
        
        # Check that it's a different object but with the same data
        assert copy is not original
        assert copy.array_size == original.array_size
        assert copy.p == original.p
        assert jnp.array_equal(copy.genome, original.genome)
        assert copy._metadata == original._metadata
        assert copy._is_valid == original._is_valid
        
    def test_clone_deep(self):
        """Test deep copy."""
        original = BinaryGenome(array_size=5, p=0.3, random_init=True, random_key=42)
        original._metadata = {"test": "metadata", "nested": {"key": "value"}}
        
        copy = original.clone(deep=True)
        
        # Check that it's a different object but with the same data
        assert copy is not original
        assert copy.array_size == original.array_size
        assert copy.p == original.p
        assert jnp.array_equal(copy.genome, original.genome)
        assert copy._metadata == original._metadata
        assert copy._metadata is not original._metadata  # Should be a new dict
        
    def test_update_from_tensor(self):
        """Test updating genome from tensor."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=42)
        original_genome = genome.genome.copy()
        
        new_tensor = jnp.array([1, 0, 1, 0, 1])
        genome.update_from_tensor(new_tensor, validate=True)
        
        assert jnp.array_equal(genome.to_tensor(), new_tensor)
        assert not jnp.array_equal(genome.genome, original_genome)
        
    def test_update_from_tensor_invalid_shape(self):
        """Test updating genome with wrong shape tensor."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=42)
        
        wrong_tensor = jnp.array([1, 0, 1])  # Wrong size
        with pytest.raises(ValueError, match="Tensor shape .* incompatible with array_size"):
            genome.update_from_tensor(wrong_tensor)

    def test_tree_flatten_unflatten(self):
        """Test JAX tree flattening and unflattening."""
        original = BinaryGenome(array_size=5, p=0.3, random_init=True, random_key=42)
        original._metadata = {"test": "metadata"}
        
        children, aux_data = original.tree_flatten()
        reconstructed = BinaryGenome.tree_unflatten(aux_data, children)
        
        assert reconstructed.array_size == original.array_size
        assert reconstructed.p == original.p
        assert jnp.array_equal(reconstructed.genome, original.genome)
        assert reconstructed._metadata == original._metadata
        
    def test_equality(self):
        """Test genome equality comparison."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=False) 
        genome2.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome3 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome3.genome = jnp.array([1, 0, 0, 1, 0])
        
        assert genome1 == genome2
        assert genome1 != genome3
        assert genome1 != "not a genome"
        
    def test_hash(self):
        """Test genome hashing."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome2.genome = jnp.array([0, 1, 1, 0, 1])
        
        # Equal genomes should have same hash
        assert hash(genome1) == hash(genome2)
        
        # Should be usable in sets
        genome_set = {genome1, genome2}
        assert len(genome_set) == 1  # Should be deduplicated
        
    def test_subtraction_operator(self):
        """Test distance calculation via subtraction operator."""
        genome1 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome1.genome = jnp.array([0, 1, 1, 0, 1])
        
        genome2 = BinaryGenome(array_size=5, p=0.5, random_init=False)
        genome2.genome = jnp.array([1, 1, 0, 0, 1])
        
        distance = genome1 - genome2
        assert distance == 2.0
        
        with pytest.raises(TypeError):
            genome1 - "not a genome"
            
    def test_string_representations(self):
        """Test string and repr methods."""
        genome = BinaryGenome(array_size=5, p=0.3, random_init=True, random_key=42)
        
        str_repr = str(genome)
        assert "BinaryGenome" in str_repr
        assert "size=5" in str_repr
        assert "valid=" in str_repr
        
        repr_str = repr(genome)
        assert "BinaryGenome" in repr_str
        assert "array_size=5" in repr_str
        assert "p=0.3" in repr_str
        
    def test_invalidate_cache(self):
        """Test cache invalidation."""
        genome = BinaryGenome(array_size=5, p=0.5, random_init=True, random_key=42)
        
        # Access is_valid to cache it
        assert genome.is_valid is True
        
        # Invalidate cache
        genome.invalidate()
        assert genome._is_valid is None
        
        # Should recompute on next access
        assert genome.is_valid is True
        
    def test_metadata_handling(self):
        """Test metadata getter and setter."""
        genome = BinaryGenome(array_size=5, p=0.5, test_meta="value")
        
        assert genome.metadata["test_meta"] == "value"
        
        new_metadata = {"new_key": "new_value"}
        genome.metadata = new_metadata
        assert genome.metadata == new_metadata
        
    def test_size_and_shape_properties(self):
        """Test size and shape properties."""
        genome = BinaryGenome(array_size=8, p=0.5, random_init=True, random_key=42)
        
        assert genome.size == 8
        assert genome.shape == (8,)