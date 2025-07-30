"""
Binary solution implementation that simplifies usage of AbstractSolution for binary problems.
"""

from typing import Any, Optional, Dict, Tuple, List, Callable
import jax  # type: ignore
import jax.random as jar  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import Array  # type: ignore
import pytest

from malthusjax.core.solution.binary_solution import BinarySolution
from malthusjax.core.genome.binary import BinaryGenome
from malthusjax.core.fitness.binary_ones import BinarySumFitnessEvaluator, KnapsackFitnessEvaluator


class TestBinarySolution:
    """Test class for BinarySolution functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.array_size = 10
        self.p = 0.5
        self.genome_init_params = {'array_size': self.array_size, 'p': self.p}
        self.random_key = jar.PRNGKey(42)
    
    def test_initialization_with_params(self):
        """Test basic initialization with genome parameters."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        
        assert solution.array_size == self.array_size
        assert solution.p == self.p
        assert isinstance(solution.genome, BinaryGenome)
        assert solution.binary_array.shape == (self.array_size,)
    
    def test_initialization_without_params_raises_error(self):
        """Test that initialization without genome_init_params raises ValueError."""
        with pytest.raises(ValueError, match="genome_init_params must be provided"):
            BinarySolution()
    
    def test_properties_access(self):
        """Test property getters and setters."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=False
        )
        
        # Test getters
        assert solution.array_size == self.array_size
        assert solution.p == self.p
        
        # Test setters
        new_size = 15
        new_p = 0.7
        solution.array_size = new_size
        solution.p = new_p
        
        assert solution.array_size == new_size
        assert solution.p == new_p
    
    def test_from_binary_array(self):
        """Test creation from binary array."""
        binary_data = jnp.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        self.genome_init_params['array_size'] = len(binary_data)  # Ensure size matches
        solution = BinarySolution.from_binary_array(
            binary_array=binary_data,
            genome_init_params=self.genome_init_params
        )
        
        assert jnp.array_equal(solution.binary_array, binary_data)
        assert solution.array_size == len(binary_data)
    
    def test_from_binary_array_size_mismatch_raises_error(self):
        """Test that array size mismatch raises ValueError."""
        binary_data = jnp.array([1, 0, 1])  # Size 3, but genome_init_params expects 10
        self.genome_init_params['array_size'] = 10  # Expecting size 10
        with pytest.raises(ValueError, match="Binary array size .* does not match"):
            BinarySolution.from_binary_array(
                binary_array=binary_data,
                genome_init_params=self.genome_init_params
            )
    
    def test_tensor_operations(self):
        """Test tensor conversion and reconstruction."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        
        # Convert to tensor
        tensor = solution.to_tensor()
        assert tensor.shape == (self.array_size,)
        
        # Reconstruct from tensor
        reconstructed = BinarySolution.from_tensor(
            tensor=tensor,
            genome_init_params=self.genome_init_params
        )
        
        assert jnp.array_equal(solution.binary_array, reconstructed.binary_array)
        assert reconstructed.array_size == solution.array_size
        assert reconstructed.p == solution.p
    
    def test_tensor_size_mismatch_raises_error(self):
        """Test that tensor size mismatch raises ValueError."""
        wrong_size_tensor = jnp.array([1, 0, 1])  # Size 3, but expecting 10
        
        with pytest.raises(ValueError, match="Tensor shape .* does not match expected array size"):
            BinarySolution.from_tensor(
                tensor=wrong_size_tensor,
                genome_init_params=self.genome_init_params
            )
    
    def test_flip_bit(self):
        """Test bit flipping functionality."""
        binary_data = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        solution = BinarySolution.from_binary_array(
            binary_array=binary_data,
            genome_init_params=self.genome_init_params
        )
        
        # Flip bit at index 1 (from 0 to 1)
        flipped_solution = solution.flip_bit(1)
        expected = binary_data.at[1].set(1)
        
        assert jnp.array_equal(flipped_solution.binary_array, expected)
        assert not jnp.array_equal(solution.binary_array, flipped_solution.binary_array)
    
    def test_flip_bit_out_of_bounds_raises_error(self):
        """Test that flipping bit out of bounds raises ValueError."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True
        )
        
        with pytest.raises(ValueError, match="Index .* out of bounds"):
            solution.flip_bit(self.array_size)  # Index equals array size (out of bounds)
        
        with pytest.raises(ValueError, match="Index .* out of bounds"):
            solution.flip_bit(-1)  # Negative index
    
    def test_hamming_distance(self):
        """Test Hamming distance calculation."""
        binary1 = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        binary2 = jnp.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0])
        
        solution1 = BinarySolution.from_binary_array(binary1, self.genome_init_params)
        solution2 = BinarySolution.from_binary_array(binary2, self.genome_init_params)
        
        # Expected differences at positions: 0, 3, 5, 8 = 4 differences
        expected_distance = 4
        assert solution1.hamming_distance(solution2) == expected_distance
        assert solution2.hamming_distance(solution1) == expected_distance
    
    def test_hamming_distance_same_solution(self):
        """Test Hamming distance with identical solutions."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        
        assert solution.hamming_distance(solution) == 0
    
    def test_hamming_distance_type_error(self):
        """Test that Hamming distance with non-BinarySolution raises TypeError."""
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True
        )
        
        with pytest.raises(TypeError, match="other must be a BinarySolution"):
            solution.hamming_distance("not a solution")
    
    def test_hamming_distance_size_mismatch_raises_error(self):
        """Test that Hamming distance with different sizes raises ValueError."""
        solution1 = BinarySolution(
            genome_init_params={'array_size': 5, 'p': 0.5},
            random_init=True
        )
        solution2 = BinarySolution(
            genome_init_params={'array_size': 8, 'p': 0.5},
            random_init=True
        )
        
        with pytest.raises(ValueError, match="Solutions must have same array size"):
            solution1.hamming_distance(solution2)
    
    def test_fitness_operations(self):
        """Test fitness setting and comparison."""
        solution1 = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True
        )
        solution2 = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True
        )
        
        # Set fitness values
        solution1.raw_fitness = 0.8
        solution2.raw_fitness = 0.6
        
        assert solution1.has_fitness
        assert solution2.has_fitness
        assert solution1 > solution2  # Higher fitness
        assert solution2 < solution1  # Lower fitness
    
    def test_fitness_transform(self):
        """Test fitness transformation."""
        transform = lambda x: x * 2  # Double the fitness
        solution = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            fitness_transform=transform
        )
        
        solution.raw_fitness = 5.0
        assert solution.raw_fitness == 5.0
        assert solution.fitness == 10.0  # Transformed
    
    def test_clone_functionality(self):
        """Test solution cloning."""
        original = BinarySolution(
            genome_init_params=self.genome_init_params,
            random_init=True,
            random_key=self.random_key
        )
        original.raw_fitness = 42.0
        
        cloned = original.clone()
        
        # Should be equal but not the same object
        assert original == cloned
        assert original is not cloned
        assert original.genome is not cloned.genome
        assert cloned.raw_fitness == 42.0
        
        # Modifying clone shouldn't affect original
        cloned.raw_fitness = 100.0
        assert original.raw_fitness == 42.0
    
    def test_string_representations(self):
        """Test string representations."""
        binary_data = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        solution = BinarySolution.from_binary_array(binary_data, self.genome_init_params)
        solution.raw_fitness = 3.14159
        
        str_repr = str(solution)
        assert "BinarySolution" in str_repr
        assert "3.1416" in str_repr  # Rounded fitness
        assert "1010101010" in str_repr  # Binary string
        
        repr_str = repr(solution)
        assert "array_size=10" in repr_str
        assert "p=0.5" in repr_str
        assert "1010101010" in repr_str
    
    def test_to_binary_string(self):
        """Test binary string conversion."""
        binary_data = jnp.array([1, 0, 1, 1, 0])
        genome_params = {'array_size': 5, 'p': 0.5}
        solution = BinarySolution.from_binary_array(binary_data, genome_params)
        
        assert solution.to_binary_string() == "10110"
    
    def test_long_binary_string_truncation(self):
        """Test that long binary strings are truncated in string representation."""
        long_params = {'array_size': 25, 'p': 0.5}
        solution = BinarySolution(
            genome_init_params=long_params,
            random_init=True,
            random_key=self.random_key
        )
        
        str_repr = str(solution)
        # Should contain truncation indicator
        assert "..." in str_repr
    
    def test_equality_and_hashing(self):
        """Test equality comparison and hashing based on genome."""
        binary_data = jnp.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        
        solution1 = BinarySolution.from_binary_array(binary_data, self.genome_init_params)
        solution2 = BinarySolution.from_binary_array(binary_data, self.genome_init_params)
        
        # Different fitness but same genome should be equal
        solution1.raw_fitness = 10.0
        solution2.raw_fitness = 20.0
        
        assert solution1 == solution2
        assert hash(solution1) == hash(solution2)
        
    def test_integration_with_fitness_evaluator(self):
        """Test integration with fitness evaluators."""
        # Create some solutions
        solutions = []
        for i in range(3):
            solution = BinarySolution(
                genome_init_params=self.genome_init_params,
                random_init=True,
                random_key=jar.PRNGKey(i)
            )
            solutions.append(solution)
        
        # Use BinarySumFitnessEvaluator
        evaluator = BinarySumFitnessEvaluator()
        evaluator.evaluate_solutions(solutions)
        
        # All solutions should now have fitness
        for solution in solutions:
            assert solution.has_fitness
            assert isinstance(solution.raw_fitness, (int, float))
            assert solution.raw_fitness >= 0  # Sum of binary values is non-negative