# Dynamic Network Evolution

Graph-based neural network evolution with topology mutations (grow/prune nodes and connections).

## What it's for

Implements evolving neural network graphs where topology changes through mutation. Networks start minimal (input/output nodes only) and grow hidden nodes and connections. Per-network evolution logic with complex branching that's more efficient than population-wide operations.

## What it contains

### Core Classes
- `Node` - Network node/neuron (input, hidden, or output role)
- `NodeList` - Ordered collection of nodes with mutable/immutable tracking
- `Net` - Complete network with mutation operations

### Node Properties
- Input nodes: Forward input signals, non-parametric, no incoming connections
- Hidden nodes: Mutable, ≤3 incoming connections, frozen random weights, no biases, apply standardization
- Output nodes: Inherit hidden node properties, fixed count matching output dimension

### Network Mutations
- `mutate()` - Randomly applies one of: grow node, grow connection, prune node, prune connection
- `grow_node()` - Adds hidden node with random connections
- `grow_connection()` - Adds connection between existing nodes (respects max 3 incoming per node)
- `prune_node()` - Removes hidden node and reconnects graph
- `prune_connection()` - Removes single connection

## Key Details

Uses dual UID system: mutable_uid (changes when topology changes) and immutable_uid (permanent, tracks total nodes ever created). Weights are randomly initialized when connections form, then frozen - no gradient-based weight updates. Node standardization uses statistics tracked in n_mean_m2_x_z tensors (Welford algorithm). Networks can have cycles (recurrent) requiring multiple passes per input (num_network_passes_per_input). Mutations maintain graph connectivity and respect constraints (output nodes never pruned, input nodes never accept incoming connections). The evolution logic is per-network (not batched) because branching complexity makes vectorization inefficient. Used by platform/models/dynamic.py which wraps populations and handles batched computation via common/dynamic_net/computation.py.
