# Dynamic Topology Networks

Graph-based neural networks with evolving topology for neuroevolution (GA-exclusive).

## Overview

This directory implements dynamic networks where the architecture itself evolves through mutations. Unlike standard feedforward or recurrent networks with fixed structure, these networks can grow and prune nodes and connections during training.

## Files

### Core Evolution
- **evolution.py** - Per-network topology evolution
  - `Node` - Network node (input/hidden/output roles)
  - `NodeList` - Organized collection of nodes
  - `Net` - Complete network with mutation operations
  - Mutations: `grow_node`, `grow_connection`, `prune_node`, `prune_connection`
  - Per-network logic (not batched) due to branching complexity
  - Frozen random weights, no gradient updates

### Population Computation
- **main.py** - Production population wrapper
  - `DynamicNetPopulation` - Batched population manager
  - Implements population interface for optimizer integration
  - GPU-parallel forward passes across population
  - Handles variable topologies via padding and masking
  - Rebuilds computation infrastructure after mutations/selection

### Testing
- **compute_test.py** - Example batched computation
  - `WelfordRunningStandardizer` - Online z-score standardization
  - `barebone_run()` - Detailed example with verbose output
  - Demonstrates index mapping and batched evaluation

## Key Concepts

**Variable Topology:** Each network can have different number of nodes and connections. Batching requires careful index management.

**Graph-Based Recurrence:** Cycles in the graph create recurrence. Networks may need multiple forward passes per input (`num_network_passes_per_input`).

**Frozen Weights:** Connection weights are randomly initialized when connections form, then never updated. Only topology evolves.

**Online Standardization:** Welford algorithm enables incremental mean/variance computation for z-score normalization during execution.

**GA-Exclusive:** Topology mutations are non-differentiable, so these networks only work with genetic algorithms, not SGD.

**Per-Network Evolution:** Unlike standard populations where parameters are mutated in parallel, topology changes happen per-network due to branching logic.

## Usage

```python
from ne.models.dynamic import DynamicNetPopulation

# Create population
pop = DynamicNetPopulation(
    input_size=4,
    output_size=2,
    pop_size=50,
    device="cuda"
)

# Forward pass
logits = pop.forward_batch(observations)  # [pop_size, N, output_size]

# Evaluate fitness
fitness = pop.evaluate(observations, actions)  # [pop_size]

# Evolution step
pop.select_simple_ga(fitness)  # Selection
pop.mutate()  # Mutation
```

## Integration

Used by `ne/optimizers/ga.py` for evolving-topology genetic algorithm experiments. Provides standard population interface but with dynamic architecture.
