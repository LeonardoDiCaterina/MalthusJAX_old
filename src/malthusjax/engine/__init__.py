"""
Level 3: The JIT-compiled Genetic Algorithm Engine.

This package provides modular, swappable engines for different use cases:

- ProductionEngine: Lean, high-performance engine optimized for deployment
- ResearchEngine: Full-featured engine with complete introspection capabilities
- MalthusEngine: Factory function for easy engine selection

The state system is now hierarchical:
- AbstractState: Base state interface
- ProductionState: Lean state for production use
- ResearchState: Rich state with callback metrics
- MalthusState: Legacy compatibility (alias to AbstractState)
"""

from .state import MalthusState
from .base import AbstractState, AbstractEngine
from .ProductionEngine import ProductionEngine, ProductionState
from .ResearchEngine import ResearchEngine, ResearchState, CallbackMetrics, FullIntermediateState


def MalthusEngine(genome_representation,
                 fitness_evaluator,
                 selection_operator,
                 crossover_operator,
                 mutation_operator,
                 elitism: int,
                 engine_type: str = "production"):
    """
    Factory function to create the appropriate MalthusJAX engine.
    
    Args:
        genome_representation: The genome representation object
        fitness_evaluator: The fitness evaluator object  
        selection_operator: The selection operator object
        crossover_operator: The crossover operator object
        mutation_operator: The mutation operator object
        elitism: Number of elite individuals to retain each generation
        engine_type: Either "production" for lean performance or "research" for full introspection
        
    Returns:
        ProductionEngine or ResearchEngine instance based on engine_type
        
    Examples:
        # For deployment/production use
        engine = MalthusEngine(..., engine_type="production")
        final_state, fitness_history = engine.run(key, 100, 50)
        
        # For research/development use  
        engine = MalthusEngine(..., engine_type="research")
        final_state, all_intermediates = engine.run(key, 100, 50)
    """
    
    if engine_type == "production":
        return ProductionEngine(
            genome_representation=genome_representation,
            fitness_evaluator=fitness_evaluator,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            elitism=elitism
        )
    elif engine_type == "research":
        return ResearchEngine(
            genome_representation=genome_representation,
            fitness_evaluator=fitness_evaluator,
            selection_operator=selection_operator,
            crossover_operator=crossover_operator,
            mutation_operator=mutation_operator,
            elitism=elitism
        )
    else:
        raise ValueError(f"Unknown engine_type: {engine_type}. Must be 'production' or 'research'.")


__all__ = [
    # Legacy compatibility
    "MalthusState",
    
    # Abstract base classes
    "AbstractState", 
    "AbstractEngine",
    
    # Production engine
    "ProductionEngine", 
    "ProductionState",
    
    # Research engine  
    "ResearchEngine", 
    "ResearchState", 
    "CallbackMetrics", 
    "FullIntermediateState",
    
    # Factory function
    "MalthusEngine"
]