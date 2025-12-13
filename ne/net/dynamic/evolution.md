# evolution.py

## Purpose
Implements graph-based neural network evolution with topology mutations (grow/prune nodes and connections). Networks start minimal and evolve through architectural changes.

## Contents

### `Node` Class
Network node/neuron with three distinct roles:

**Input Nodes:**
- Forward input signals
- Non-parametric, no incoming connections
- Count matches input dimension

**Hidden Nodes:**
- Mutable parametric nodes
- ≤3 incoming connections with frozen random weights
- No biases
- Apply standardization: `standardize(weights · in_nodes' outputs)`

**Output Nodes:**
- Inherit hidden node properties
- Count matches output dimension
- Never pruned

**Node Attributes:**
- `mutable_uid` - Position-dependent ID (changes with topology)
- `immutable_uid` - Permanent lifetime tracking ID
- `in_nodes` / `out_nodes` - Connection lists
- `weights` - Frozen random weights (3 max)

**Node Methods:**
- `sample_nearby_node()` - Local connectivity-biased node sampling
- `connect_to()` / `disconnect_from()` - Connection management

### `NodeList` Class
Dataclass organizing nodes by role and status:
- `all`, `input`, `hidden`, `output` - Nodes by role
- `receiving`, `emitting` - Connection tracking (nodes appear once per connection)
- `being_pruned` - Prevents infinite pruning loops

### `Net` Class
Complete evolving network with mutation operations.

**Network Attributes:**
- `nodes` - NodeList containing all nodes
- `weights_list` - All mutable nodes' weights
- `n_mean_m2_x_z` - Welford standardization parameters (n, mean, m2, x, z) per node
- `avg_num_grow_mutations` / `avg_num_prune_mutations` - Co-evolving mutation rates
- `num_network_passes_per_input` - Multiple passes for recurrent graphs
- `local_connectivity_probability` - Bias toward local connections
- `in_nodes_indices` - Tensor mapping for computation
- `weights` - Tensor of all weights for batched computation

**Initialization:**
- `initialize_architecture()` - Creates input and output nodes (minimal start)

**Growth Operations:**
- `grow_node()` - Adds hidden node with 2 incoming, 1 outgoing connection
  - Prioritizes connecting isolated input/output nodes
  - Uses local connectivity bias via `sample_nearby_node()`
- `grow_connection()` - Adds edge between nodes

**Pruning Operations:**
- `prune_node()` - Removes hidden node and all connections
  - Cascades: disconnected hidden nodes also pruned
  - Updates all `mutable_uid` values
- `prune_connection()` - Removes single edge, prunes orphaned nodes

**Mutation:**
- `mutate()` - Main evolution method combining:
  1. **Parameter Perturbation** - Mutates mutation rates, connectivity bias, num passes
  2. **Architecture Perturbation** - Prunes then grows nodes (chained mutations)
  3. **Computation Tensor Generation** - Rebuilds `in_nodes_indices` and `weights` tensors

**Cloning & Serialization (NEW):**
- `clone()` - Create deep copy of network with proper tensor cloning (replaces deepcopy)
- `get_state_dict()` - Serialize complete graph structure for checkpointing
  - Saves node structure (roles, UIDs, weights, connections by immutable_uid)
  - Saves all scalar attributes and tensors (moved to CPU for pickling)
- `load_state_dict(state)` - Restore network from serialized state
  - Two-pass reconstruction: create nodes, then reconnect (handles cycles)
  - Rebuilds computation tensors

**Device Handling (NEW):**
- `device` parameter added to `__init__()` for consistent GPU/CPU tensor creation
- All tensors (n_mean_m2_x_z, in_nodes_indices, weights) created with explicit device

## Key Design Decisions

**Per-Network Logic:** Evolution runs per-network (not batched) due to branching complexity in topology changes.

**Frozen Weights:** Connection weights set randomly and never trained - only topology evolves.

**Dual UIDs:** `mutable_uid` changes with topology, `immutable_uid` tracks lifetime.

**Graph Recurrence:** Networks can have cycles, requiring multiple forward passes per input.

**Chained Mutations:** Growing nodes reuses previous node for local structure building.

**Computation Tensors:** After mutation, generates `in_nodes_indices` and `weights` tensors for batched forward passes (used by compute_test.py).
