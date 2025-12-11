# Dynamic Network Models

Graph-based neural networks with evolving topology for evolutionary optimization.

## What it's for

Provides dynamic networks where both weights and topology can evolve. Networks start with minimal structure and grow/prune nodes and connections through mutation. GA-exclusive (not compatible with SGD) because topology changes are not differentiable.

## What it contains

### Main Class
- `DynamicNetPopulation` - Population wrapper implementing batched GPU computation interface

### Key Methods
- `forward_batch()` - Batched forward pass across entire population (parallel GPU computation)
- `get_probs_batch()` - Returns softmax probabilities for all networks
- `set_population()` - Updates population with new networks after mutation/selection
- `get_population()` - Returns current network population

## Key Details

Wraps common/dynamic_net/evolution.py's Net class to provide platform-compatible interface. Uses batched computation from common/dynamic_net/computation.py to evaluate entire populations in parallel on GPU. Networks initialize with minimal structure (input/output nodes only) then apply `initial_mutations` (default 5) to create non-trivial starting architectures. Each network has variable topology (different numbers of nodes and connections) requiring careful batching via padding and masking. Networks can have cycles (recurrence through graph structure) and multiple forward passes per input. Supports Welford running standardization for internal activations. Only works with evolutionary optimizers (platform/optimizers/genetic.py for recurrent, platform/optimizers/evolution.py as fallback).
