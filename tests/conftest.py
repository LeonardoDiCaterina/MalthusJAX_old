# File: MalthusJAX/tests/conftest.py
"""Shared test configuration and fixtures for MalthusJAX tests."""

import pytest
import jax.random as jr # type: ignore[import-untyped]


@pytest.fixture
def random_key():
    """Provide a random JAX PRNG key for tests."""
    return jr.PRNGKey(42)


@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    return {
        'test_value': 42,
        'test_array': [1, 2, 3, 4]
    }