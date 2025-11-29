# Project MalthusJAX: Development Recap & Architectural Decisions

## Document 1: Visual Identity & Theoretical Positioning
**Theme:** Defining the "MalthusJax" Brand and Competitive Edge.

### 1. Aesthetic Direction: "Clean Tech Dark Mode"
We moved away from "Cyberpunk Neon" (too distracting) and "Academic Beige" (too boring) to a hybrid professional aesthetic:
* **Background:** Deep Charcoal (`#1E1E2E`) to reduce eye strain and signal "Developer Tool."
* **Palette:** Functional Syntax Highlighting colors (Coral for Ops, Emerald for Data, Cyan for Structure).
* **Graphics:** Flat, 2D schematic diagrams with subtle glows, avoiding complex 3D wireframes to emphasize architectural clarity.

### 2. The "Hardware Lottery" Argument
We defined the core narrative for why MalthusJax exists:
* **The Problem:** Standard Genetic Programming (Tree-based) fails on GPUs due to **Warp Divergence** (threads waiting for neighbors) and **Pointer Chasing** (memory latency).
* **The Competitors:**
    * **TensorGP:** Uses "Data Parallelism" (1 Tree vs N Pixels). Fails at population scaling due to memory fragmentation and lack of JIT (Eager execution).
    * **Kozax:** Uses a "Matrix-of-Trees." Efficient, but suffers from **Sparse Utilization** (padding empty nodes) and enforces a rigid Tree Topology.
* **The MalthusJax Solution:** **Population Parallelism** via Linear Genomes. We treat the GPU as a massive, lockstep Stack Machine. 100% memory utilization, 0% warp divergence.

### 3. Theoretical Innovations
* **The Symbiotic Genome:** The genome is not one solution, but a container of $L$ potential solutions (Atomic Trees).
* **Topological Constraints:** We enforce a Directed Acyclic Graph (DAG) structure where instruction $i$ can only reference memory indices $< i$.

---

## Document 2: Core Architecture (The "Great Pytree Refactor")
**Theme:** Enforcing JAX-Correctness and Functional Purity.

### 1. The "Russian Doll" Class Hierarchy
We restructured the code to separate concerns and ensure every object is a valid JAX Pytree (`@struct.dataclass`).
* **`GenomeConfig`:** Static blueprints (Length, Inputs, Ops). Passed to functions to determine array shapes.
* **`BaseGenome` (Abstract):** Defines the interface (`distance`, `autocorrect`, `random_init`).
* **`LinearGenome` / `ContinuousGenome` (Concrete):** The actual data containers.
    * **Linear:** `int32` arrays for Ops/Args.
    * **Continuous:** `float32` arrays for Santa 2025 ($x, y, \theta$).
* **`BasePopulation` (Abstract):** A generic container for batches of genomes.
* **`EvolutionState`:** The carrier object passed through the `jax.lax.scan` loop, holding the Population, RNG Key, and Hall of Fame (Global Best).

### 2. The "Structure of Arrays" (SoA) Pattern
We resolved the ambiguity between "Individual" and "Population":
* A **Genome** object holds the data.
* A **Population** object wraps a *batch* of Genomes.
* We implemented "Kebab-Friendly" magic methods (`__getitem__`, `__len__`, `__iter__`) on the Population class so users can interact with it like a Python list, while JAX treats it as a contiguous tensor.

### 3. Initialization & Autocorrection
* **Topological Init:** We wrote a vectorized factory that generates random genomes guaranteed to be valid DAGs.
* **Autocorrect:** A repair mechanism using `jnp.clip` to fix invalid references after mutation, ensuring the graph never cycles or crashes.

---

## Document 3: The Execution Engine & Operators
**Theme:** High-Performance Compilation and Operator Logic.

### 1. The Interpreter (`LinearGPEvaluator`)
* **`jax.lax.scan`:** We replaced Python loops with XLA-compiled loops to execute linear programs on the GPU.
* **Symbiotic Evaluation:** Instead of returning just the final result, the evaluator returns the **History** of every instruction's output. This allows us to evaluate 50 sub-solutions for the price of 1 execution.
* **Flexible Architecture:** We separated the **Config** (Context) from the **Data** (Input Batch), allowing the same evaluator to handle Regression, Knapsack, or Geometry.

### 2. The Operator Paradigm (Batch-First)
We standardized all genetic operators (Mutation, Crossover, Selection) to follow a strict JAX pattern:
* **Static Parameters:** (`num_offspring`, `tournament_size`) are marked `pytree_node=False`. Changing them triggers re-compilation.
* **Dynamic Parameters:** (`mutation_rate`, `mixing_ratio`) are traced arrays. They can be annealed/changed at runtime without penalty.
* **Batch Output:** Crossover was refactored to return a **Batch** `(N, L)` instead of a Tuple, removing "glue code" from the engine loop.

### 3. Selection Strategies
* **Symbiotic Tournament:** Selects parents based on their best *internal* components (Top $K$ atomic trees), preserving genetic diversity.
* **Standard Tournament:** Selects based on total fitness (used for interdependent problems like Santa 2025).

---

## Document 4: Use Cases & Demonstrations
**Theme:** Proving Versatility: From Code Synthesis to Tree Packing.

### 1. Use Case A: Symbolic Regression (Linear GP)
* **Goal:** Find a mathematical formula to fit data.
* **Genome:** Integers (ADD, SUB, Inputs, Registers).
* **Operators:** Bit-flip mutation, Uniform Crossover.
* **Result:** Successfully compiled a loop running millions of evaluations per second to solve regression tasks.

### 2. Use Case B: Santa 2025 (Continuous Optimization)
* **Goal:** Pack 50 Christmas trees into the smallest bounding box without overlap.
* **Genome:** Floats ($x, y, \theta$).
* **Evaluator:** A custom **Differentiable Geometric Engine** (`JaxSantaEvaluator`).
    * Replaced `shapely` with pure JAX matrix rotations and bounding box logic.
    * Implemented a collision penalty using vector distances.
* **Operators:** `GaussianMutation` (nudge position) and `ContinuousCrossover` (swap trees).
* **Result:** A fully JIT-compiled solver (`run_santa_solver.py`) that optimizes geometric packing on the GPU.

### 3. The Notebooks (Levels 1 & 2)
* We reviewed and refined `Level_1_Demo.ipynb` (Genomes/Fitness) and `Level_2_Demo.ipynb` (Operators).
* We verified the "Clean Pipeline" where Selection $\to$ Crossover $\to$ Mutation flows seamlessly without shape mismatches.