"""
Example usage of MalthusJAX fitness functions.

Demonstrates how to use the new Binary and Real genome fitness
evaluators in simple evolution experiments.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import malthusjax as mjx


def binary_optimization_example():
    """Example: Binary optimization with BinarySum and Knapsack."""
    print("🔹 Binary Optimization Examples")
    print("="*50)
    
    key = jr.PRNGKey(42)
    
    # Example 1: OneMax (BinarySum) optimization
    print("\n📊 OneMax (BinarySum) Problem:")
    config = mjx.BinaryGenomeConfig(length=20)
    evaluator = mjx.BinarySumEvaluator(mjx.BinarySumConfig(maximize=True))
    
    # Create initial population
    population = mjx.BinaryPopulation.init_random(key, config, size=50)
    initial_fitness = evaluator.evaluate_batch(population)
    best_initial = max(initial_fitness)
    
    print(f"  ✓ Initial population: {len(population)} individuals")
    print(f"  ✓ Best initial fitness: {best_initial}/20 (ones)")
    print(f"  ✓ Average fitness: {sum(initial_fitness)/len(initial_fitness):.1f}")
    
    # Example 2: Knapsack Problem
    print("\n🎒 Knapsack Problem:")
    knapsack_config = mjx.KnapsackEvaluator.create_random_problem(jr.PRNGKey(123), 15)
    knapsack_evaluator = mjx.KnapsackEvaluator(knapsack_config)
    
    knapsack_genome_config = mjx.BinaryGenomeConfig(length=15)
    knapsack_population = mjx.BinaryPopulation.init_random(jr.PRNGKey(999), knapsack_genome_config, size=30)
    
    knapsack_fitness = knapsack_evaluator.evaluate_batch(knapsack_population)
    best_knapsack = max(knapsack_fitness)
    best_idx = knapsack_fitness.index(best_knapsack)
    best_genome = knapsack_population[best_idx]
    
    print(f"  ✓ Problem: {len(knapsack_config.weights)} items, capacity={knapsack_config.capacity:.1f}")
    print(f"  ✓ Best solution: {best_genome.count_ones()} items, fitness={best_knapsack:.1f}")
    print(f"  ✓ Total value possible: {jnp.sum(knapsack_config.values):.1f}")


def real_optimization_example():
    """Example: Real-valued optimization with Sphere and Griewank."""
    print("\n🔹 Real-Valued Optimization Examples")
    print("="*50)
    
    key = jr.PRNGKey(42)
    
    # Example 1: Sphere function optimization
    print("\n📈 Sphere Function Minimization:")
    config = mjx.RealGenomeConfig(length=10, bounds=(-5.0, 5.0))
    sphere_evaluator = mjx.SphereEvaluator(mjx.SphereConfig(minimize=True))
    
    # Create initial population
    population = mjx.RealPopulation.init_random(key, config, size=40)
    initial_fitness = sphere_evaluator.evaluate_batch(population)
    best_initial = max(initial_fitness)  # Maximizing negative values
    best_idx = initial_fitness.index(best_initial)
    best_genome = population[best_idx]
    
    print(f"  ✓ Problem: Minimize f(x) = sum(x_i^2), 10 dimensions")
    print(f"  ✓ Global minimum: f(0,0,...,0) = 0")
    print(f"  ✓ Best initial: f={-best_initial:.3f} (fitness={best_initial:.3f})")
    print(f"  ✓ Best genome magnitude: {best_genome.magnitude():.3f}")
    
    # Example 2: Griewank function (multimodal)
    print("\n🌊 Griewank Function (Multimodal):")
    griewank_config = mjx.RealGenomeConfig(length=6, bounds=(-600.0, 600.0))
    griewank_evaluator = mjx.GriewankEvaluator(mjx.GriewankConfig(minimize=True))
    
    griewank_population = mjx.RealPopulation.init_random(jr.PRNGKey(999), griewank_config, size=30)
    griewank_fitness = griewank_evaluator.evaluate_batch(griewank_population)
    best_griewank = max(griewank_fitness)
    
    print(f"  ✓ Problem: Griewank function, 6 dimensions")
    print(f"  ✓ Global minimum: f(0,0,...,0) = 0")
    print(f"  ✓ Best found: f={-best_griewank:.3f} (fitness={best_griewank:.3f})")
    print(f"  ✓ Fitness range: [{min(griewank_fitness):.1f}, {max(griewank_fitness):.1f}]")


def main():
    """Run all fitness function examples."""
    print("🚀 MalthusJAX Fitness Functions Examples")
    print("=========================================")
    
    binary_optimization_example()
    real_optimization_example()
    
    print("\n\n🎯 Summary:")
    print("  ✅ Binary genome fitness functions ready for:")
    print("     - Combinatorial optimization (OneMax, TSP, etc.)")
    print("     - Constrained problems (Knapsack, scheduling)")
    print("  ✅ Real genome fitness functions ready for:")
    print("     - Function optimization (Sphere, Rastrigin, etc.)")
    print("     - Multimodal optimization (Griewank, Ackley)")
    print("     - Constrained optimization (Box constraints)")
    print("  ⚡ All functions JIT-compiled for high performance")
    print("\n🎉 Ready to implement evolution algorithms! 🧬")


if __name__ == "__main__":
    main()