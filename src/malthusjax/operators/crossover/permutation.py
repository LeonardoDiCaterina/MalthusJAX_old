from malthusjax.operators.crossover.base import AbstractCrossover
from functools import partial
from typing import Callable
import jax # type: ignore
import jax.numpy as jnp # type: ignore
import jax.random as jar # type: ignore

# --- Cycle Crossover (CX) ---

class CycleCrossover(AbstractCrossover):
    """
    Cycle Crossover (CX) for permutation genomes.
    
    This implementation finds one cycle starting from a random point
    for each output required.
    """
    
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compiled function for cycle crossover.
        Ignores crossover_rate, as CX is typically an all-or-nothing operation.
        """
        return partial(
            _cycle_crossover,
            n_outputs=self.n_outputs
        )

@partial(jax.jit, static_argnames=["n_outputs"])
def _cycle_crossover(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array,
    n_outputs: int
) -> jax.Array:
    """Pure JAX cycle crossover, vmapped for n_outputs."""
    
    # Create a batch of random start positions, one for each output offspring
    start_pos_tensor = jar.randint(key, (n_outputs,), 0, parent1.shape[0])
    
    # Vmap the single-cycle crossover function over the batch of start positions
    return jax.vmap(
        _cycle_crossover_single,
        in_axes=(None, None, 0) # Same parents, new start_pos for each
    )(parent1, parent2, start_pos_tensor)

@jax.jit
def _cycle_crossover_single(
    parent1: jax.Array,
    parent2: jax.Array,
    start_pos: int
) -> jax.Array:
    """Performs a single cycle crossover starting at start_pos."""
    
    def body_fun(carry, _):
        # carry = (current_index, offspring_in_progress)
        idx, offspring = carry
        
        # Find the value in parent1 at the current index
        val_at_idx = parent1[idx]
        
        # Find the index of that value in parent2
        next_idx = jnp.argmin(jnp.abs(parent2 - val_at_idx)) # More robust than jnp.where
        
        # Set the gene in the offspring
        new_offspring = offspring.at[next_idx].set(parent2[next_idx])
        return (next_idx, new_offspring), None

    # Start with parent1, but with the first gene from parent2 at start_pos
    initial_offspring = parent1.at[start_pos].set(parent2[start_pos])
    
    # Run the scan to follow the cycle
    (_, final_offspring), _ = jax.lax.scan(
        body_fun,
        (start_pos, initial_offspring),
        jnp.arange(parent1.shape[0]) # Run for max genome length
    )
    return final_offspring

# --- Pillar Crossover (PX) ---

class PillarCrossover(AbstractCrossover):
    """
    Pillar Crossover (PX) for permutation genomes.
    Keeps genes (pillars) that are common between parents
    and shuffles the remaining genes.
    """
    
    def get_pure_function(self) -> Callable:
        """
        Returns a JIT-compiled function for pillar crossover.
        Ignores crossover_rate.
        """
        return partial(
            _pillar_crossover,
            n_outputs=self.n_outputs
        )

@partial(jax.jit, static_argnames=["n_outputs"])
def _pillar_crossover(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array,
    n_outputs: int
) -> jax.Array:
    """Pure JAX pillar crossover, vmapped for n_outputs."""
    
    # Create a batch of keys, one for each output offspring
    keys = jar.split(key, n_outputs)
    
    # Vmap the single pillar crossover function
    return jax.vmap(
        _pillar_crossover_single,
        in_axes=(0, None, None) # New key for each, same parents
    )(keys, parent1, parent2)

@jax.jit
def _pillar_crossover_single(
    key: jax.Array,
    parent1: jax.Array,
    parent2: jax.Array
) -> jax.Array:
    """Performs a single pillar crossover."""
    
    length = parent1.shape[0]
    indices = jnp.arange(length)
    
    # 1. Find pillars (common genes)
    mask = (parent1 == parent2)
    
    # 2. Get indices of non-pillar genes from parent1
    non_pillar_indices = jnp.where(~mask, indices, length)
    
    # 3. Get the values of non-pillar genes from parent1
    non_pillar_values = jnp.where(~mask, parent1, 0)
    
    # 4. Create a shuffled version of the non-pillar values
    shuffled_non_pillar_values = jar.permutation(key, non_pillar_values)

    # 5. Create the offspring
    # Start with parent1 (which already has the correct pillars)
    # and "fill in" the non-pillar spots with the shuffled values.
    # We use jnp.where to place the shuffled values.
    
    # This is tricky. A simpler, more JAX-friendly way:
    # 1. Find pillars (common genes)
    mask = (parent1 == parent2)
    
    # 2. Get values from parent1 that are *not* pillars
    non_pillar_values = parent1[~mask]
    
    # 3. Shuffle them
    shuffled_values = jar.permutation(key, non_pillar_values)
    
    # 4. Create the offspring by filling in the blanks
    # Use jax.lax.scan to fill in the non-pillar indices one by one
    
    def fill_body(carry, i):
        (offspring, shuffled_idx) = carry
        
        def update_fn(offspring, shuffled_idx):
            # If this is a non-pillar index, fill it
            val_to_fill = shuffled_values[shuffled_idx]
            offspring = offspring.at[i].set(val_to_fill)
            return offspring, shuffled_idx + 1
            
        def no_update_fn(offspring, shuffled_idx):
            # It's a pillar, do nothing
            return offspring, shuffled_idx

        return jax.lax.cond(mask[i], no_update_fn, update_fn, offspring, shuffled_idx), None

    # Start with parent1 (which has the correct pillars) and 0s elsewhere
    initial_offspring = jnp.where(mask, parent1, 0)
    (final_offspring, _), _ = jax.lax.scan(
        fill_body,
        (initial_offspring, 0),
        indices
    )
    
    return final_offspring