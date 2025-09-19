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
        def crossover(genome_data1: jax.Array, genome_data2: jax.Array, random_key: jax.Array) -> jax.Array:  
            """Perform uniform crossover on a single genome."""
            # Generate a random mask based on the crossover rate
            mask = jar.bernoulli(random_key, p=self.crossover_rate, shape=genome_data1.shape)
            # Combine the two genomes using the mask
            crossed_genome = jnp.where(mask, genome_data1, genome_data2)
            return crossed_genome

        
        return crossover
    
class CycleCrossover(AbstractCrossover):
    """Cycle crossover operator that creates offspring by following cycles in the parent genomes.
    
    This operator is more complex and can be optimized for JIT compilation.
    """
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function for cycle crossover."""
        crossover_rate = self.crossover_rate  # Capture as closure variable
        
        def crossover(genome_data1: jax.Array, genome_data2: jax.Array, random_key: jax.Array) -> jax.Array:  # Only 2 args
            """Perform cycle crossover on a single genome.
            
            Args:
                genome_data1: Input genome array 
                genome_data2: Second genome array for crossover
                random_key: JAX random key for stochastic operations 
                
            Returns:
                Modified genome after crossover
            """
            start_pos = jar.randint(random_key, (), 0, genome_data1.shape[0])
            def body_fun(carry, _):
                idx, offspring = carry
                next_idx = jnp.where(genome_data1 == genome_data2[idx], size=1)[0][0]
                new_offspring = offspring.at[next_idx].set(genome_data2[next_idx])
                return (next_idx, new_offspring), None
            
            
            initial_offspring = genome_data1.at[start_pos].set(genome_data2[start_pos])
            (_, final_offspring), _ = jax.lax.scan(
                lambda c, x: body_fun(c, x),
                (start_pos, initial_offspring),
                jnp.arange(genome_data1.shape[0]),
            )

            return final_offspring

        return crossover
    
    

class SinglePointCrossover(AbstractCrossover):
    """Single point crossover operator that randomly selects a crossover point.
    
    This operator uses a single crossover point to exchange genetic material between two parents.
    """
    
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function for single point crossover."""
        crossover_rate = self.crossover_rate  # Capture as closure
        @partial(jax.jit, static_argnames=())
        def crossover(genome_data1: jax.Array, genome_data2: jax.Array, random_key: jax.Array) -> jax.Array:
            """Perform single point crossover on two genomes."""
            # Select a random crossover point
            crossover_point = jar.randint(random_key, (), 0, genome_data1.shape[0])
            # Create a mask for the crossover point
            mask = jnp.arange(genome_data1.shape[0]) < crossover_point
            # Combine the two genomes using the mask
            crossed_genome = jnp.where(mask, genome_data1, genome_data2)

            return crossed_genome

        return crossover




class PillarCrossover(AbstractCrossover):
    """Pillar point crossover operator keeps the genes in common between two parents and shuffles the rest.
    
    """
    
    def _create_crossover_function(self) -> Callable:
        """Create the core crossover function for single point crossover."""
        crossover_rate = self.crossover_rate  # Capture as closure
        @partial(jax.jit, static_argnames=())
        def crossover(genome_data1: jax.Array, genome_data2: jax.Array, random_key: jax.Array) -> jax.Array:
            """Perform pillar point crossover on two genomes."""
            length = genome_data1.shape[0]
            indices = jnp.arange(length)
            
            # Create mask where genomes are equal
            mask = genome_data1 == genome_data2
            
            # Create permuted indices for shuffling
            perm_indices = jax.random.permutation(random_key, indices)
            
            # Create the output array starting with genome_data1
            output = genome_data1
            
            # Function to update a single index
            def update_fn(i, val):
                should_update = ~mask[i]
                # Find the corresponding permuted value
                perm_idx = perm_indices[i]
                new_val = jnp.where(should_update, genome_data1[perm_idx], val)
                return new_val
            
            # Apply updates using scan to avoid dynamic indexing
            output = jax.lax.fori_loop(0, length, 
                                    lambda i, acc: acc.at[i].set(update_fn(i, acc[i])), 
                                    output)
            
            return output

        return crossover