"""
Defines the core state for the MalthusJAX engine.
This state object is a Pytree and serves as the "carry"
for the jax.lax.scan loop, passing all necessary information
from one generation to the next.
"""

import flax.struct # type: ignore
from jax import Array # type: ignore
from jax.random import PRNGKey # type: ignore
import jax.numpy as jnp # type: ignore


@flax.struct.dataclass
class MalthusState_callbacks:
    """
    The complete, immutable state of the Genetic Algorithm at any
    given generation.
    """
    
    # --- Core GA Data ---
    
    population: Array
    """
    The JAX array of all genomes.
    Shape: (population_size, *genome_shape)
    """
    
    fitness: Array
    """
    The JAX array of fitness values for the population.
    Shape: (population_size,)
    """
    
    # --- Elitism & Tracking ---
    
    best_genome: Array
    """
    The single best genome found so far in the run.
    Shape: (*genome_shape)
    """
    
    best_fitness: float
    """The fitness value of the single best genome."""
    
    # --- Loop State ---
    
    key: PRNGKey
    """
    The JAX PRNGKey. This is the MOST CRITICAL part.
    It *must* be part of the state and updated every step
    to ensure reproducible randomness.
    """
    
    generation: int
    """A simple integer counter for the current generation."""
    
    
    #--- Additional Callbacks State ---

    generation_start_population: jnp.ndarray
    
    generation_start_metrics: jnp.ndarray
    
    post_evaluation_metrics: jnp.ndarray
    
    post_selection_population: jnp.ndarray
    
    post_selection_metrics: jnp.ndarray
    
    post_crossover_population: jnp.ndarray
    
    post_crossover_metrics: jnp.ndarray
    
    post_mutation_population: jnp.ndarray
    
    post_mutation_metrics: jnp.ndarray
    
    post_elitism_population: jnp.ndarray
    
    post_elitism_metrics: jnp.ndarray