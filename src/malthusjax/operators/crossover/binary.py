from malthusjax.operators.crossover.base import AbstractCrossover
from functools import partial # this is needed for JIT compilation as it allows us to pass static arguments
from typing import Callable

import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore


class UniformCrossover(AbstractCrossover):
    """Uniform crossover operator that randomly selects genes from two parents.
    
    This operator uses a uniform mask to determine which genes to take from each parent.
    """
    
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function for uniform crossover."""
        crossover_rate = self.crossover_rate  # Capture as closure
        
        @partial(jax.jit, static_argnames=())
        def crossover(genome_data: jax.Array, random_key: jax.Array) -> jax.Array:  # Only 2 args
            """Perform uniform crossover on a single genome."""
            # Generate a random mask based on the crossover rate
            mask = jar.bernoulli(random_key, p=crossover_rate, shape=genome_data.shape)
            
            # Create a second genome by flipping the mask
            flipped_mask = jnp.logical_not(mask)
            
            # Combine the two genomes using the mask
            crossed_genome = jnp.where(mask, genome_data, flipped_mask)
            return crossed_genome
        
        return crossover
    
class CycleCrossover(AbstractCrossover):
    """Cycle crossover operator that creates offspring by following cycles in the parent genomes.
    
    This operator is more complex and can be optimized for JIT compilation.
    """
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function for cycle crossover."""
        crossover_rate = self.crossover_rate  # Capture as closure variable
        
        def crossover(genome_data: jax.Array, random_key: jax.Array) -> jax.Array:  # Only 2 args
            """Perform cycle crossover on a single genome.
            
            Args:
                genome_data: Input genome array  
                random_key: Random key for crossover decisions
                
            Returns:
                Modified genome after crossover
            """
            # Your cycle crossover logic here using crossover_rate
            # Generate mask based on crossover_rate
            mask = jax.random.bernoulli(random_key, p=crossover_rate, shape=genome_data.shape)
            
            # Apply cycle crossover logic
            crossed_genome = jnp.where(mask, 1 - genome_data, genome_data)
            return crossed_genome
        
        return crossover