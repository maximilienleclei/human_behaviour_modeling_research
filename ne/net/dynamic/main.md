# main.py

## Purpose
Production wrapper for dynamic network populations with batched GPU-parallel computation. Implements the population interface for integration with neuroevolution optimizers. GA-exclusive (not differentiable).

## Contents

### `DynamicNetPopulation` Class
Complete population manager for evolving-topology networks.

**Initialization:**
- Creates `pop_size` networks via `evolution.Net`
- Applies one mutation to each network
- Builds batched computation infrastructure

**Batched Computation Infrastructure** (`_build_computation_infrastructure()`):
Constructs all tensors needed for efficient GPU-parallel forward passes following `compute_test.py` pattern:

1. **`n_mean_m2_x_z`** - Welford standardization state (concatenated from all nets, 0-padded at front)
2. **`wrs`** - WelfordRunningStandardizer instance
3. **Index Mappings:**
   - `input_nodes_start_indices` - Where each network's inputs begin in flat tensor
   - `input_nodes_indices` - All input positions (flattened across population)
   - `output_nodes_indices` - All output positions (flattened across population)
   - `mutable_nodes_indices` - All hidden/output nodes
   - `flat_in_nodes_indices` - Flattened mapping of which nodes feed into each mutable node
4. **`weights`** - Concatenated connection weights from all networks
5. **Pass Mask:**
   - `num_network_passes_per_input_mask` - Boolean mask for multi-pass execution (handles variable recurrence depth)

Rebuilt after every mutation or selection (topology changes invalidate indices).

**Forward Pass** (`forward_batch()`):
Batched evaluation for all networks simultaneously:
1. For each observation:
   - Flatten observation → inject at input indices
   - Initialize `out` tensor with previous z-scores
   - Standardize via WelfordRunningStandardizer
   - For each network pass (up to max):
     - Gather inputs using `flat_in_nodes_indices`
     - Weight & sum: `(mapped_inputs * weights).sum(dim=1)`
     - Update mutable nodes (masked by pass requirement)
     - Standardize
   - Extract outputs at output indices
2. Stack outputs → `[pop_size, N, output_size]`

Returns raw logits (no softmax).

**Fitness Evaluation** (`evaluate()`):
Computes cross-entropy loss for each network:
1. Get logits via `forward_batch()`
2. Expand actions to `[pop_size, N]`
3. Flatten and compute per-sample CE
4. Reshape and mean per network → `[pop_size]` fitness values

Lower is better (minimization).

**Evolution Operations:**
- `mutate()` - Calls `net.mutate()` on all networks, then rebuilds infrastructure
- `select_and_duplicate()` - Top 50% selection (implements StructuralNetwork protocol):
  - Sort networks by fitness
  - Keep top half
  - Duplicate survivors using `net.clone()` to fill population (no longer uses deepcopy)
  - Rebuild infrastructure

**Checkpointing (FULLY IMPLEMENTED):**
- `get_state_dict()` - Serializes all networks using `net.get_state_dict()`
  - Saves complete graph structure for each network
  - Includes population metadata (input_size, output_size, pop_size, device)
- `load_state_dict(state)` - Reconstructs all networks from serialized states
  - Creates Net instances and calls `net.load_state_dict()` for each
  - Rebuilds computation infrastructure
- **Enables save/resume for dynamic network training!**

**Protocol Compliance:** Implements StructuralNetwork protocol from ne/net/protocol.py, enabling integration with Population adapter.

## Key Design Decisions

**GA-Exclusive:** Topology mutations are not differentiable → incompatible with gradient-based methods (SGD). Only works with genetic algorithms.

**Rebuild After Changes:** Any operation changing network structure (mutation, selection) requires rebuilding the computation infrastructure. This is expensive but necessary.

**Per-Observation Loop:** Unlike feedforward/recurrent populations that batch across observations, dynamic networks process one observation at a time due to stateful standardization and variable passes. Still batches across population dimension.

**Device Management:** Accepts `device` parameter, ensures all tensors on correct device.

**Welford Standardization:** Maintains running statistics across forward passes, crucial for stable training with random frozen weights.

## Integration

Used by GA optimizers in `ne/optimizers/ga.py`. Provides same interface as feedforward/recurrent populations:
- `forward_batch()` - Get logits
- `evaluate()` - Get fitness
- `mutate()` - Evolve topology
- `select_simple_ga()` - Selection pressure

Differs from other populations by having dynamic architecture and per-network evolution (not batched parameter mutations).
