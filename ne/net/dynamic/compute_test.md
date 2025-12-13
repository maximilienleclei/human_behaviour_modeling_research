# compute_test.py

## Purpose
Test/example code demonstrating batched population-wide forward computation for dynamic networks with varying topologies. Shows how to handle online standardization and variable-structure networks efficiently on GPU.

## Contents

### `WelfordRunningStandardizer` Class
Online z-score standardization using Welford's algorithm.

**Purpose:** Compute running mean and variance incrementally without storing all samples, enabling real-time standardization during network execution.

**State:** `n_mean_m2_x_z` tensor with columns:
- `n` - Sample count
- `mean` - Running mean
- `m2` - Sum of squared deviations (for variance)
- `x` - Previous raw value
- `z` - Previous z-score

**Algorithm:**
1. **Update Mask** - Only update stats for new raw values (not recycled z-scores or zeros)
2. **Welford Update** - Incremental mean/variance calculation:
   - `n += 1`
   - `delta = x - mean`
   - `mean += delta / n`
   - `m2 += delta * (x - mean)`
3. **Z-Score** - `(x - mean) / sqrt(m2/n)` for valid samples (n ≥ 2)
4. **Pass-Through** - Old z-scores and zeros pass unchanged
5. **State Storage** - Update `x` and `z` columns for next call

**Key Feature:** Selective updating - only processes new raw values, preserving already-standardized values across network passes.

### `barebone_run()` Function
Complete working example demonstrating batched forward pass for population of dynamic networks.

**Setup:**
- Creates 4 networks with 3 inputs, 2 outputs
- Mutates networks 5 times to grow varied topologies
- Sets different `num_network_passes_per_input` per network (handles cycles)

**Index Computation:**
Builds tensors mapping variable-topology networks to flat batched structure:
1. `nets_num_nodes` - Node count per network
2. `input_nodes_start_indices` - Where each network's inputs start in flat tensor
3. `input_nodes_indices` - All input node positions (flattened)
4. `output_nodes_indices` - All output node positions (flattened)
5. `mutable_nodes_indices` - All hidden/output nodes (non-input)
6. `in_nodes_indices` - Per mutable node, indices of its 3 input nodes (padded with 0)
7. `weights` - Concatenated weights from all networks
8. `num_network_passes_per_input_mask` - Boolean mask for multi-pass execution

**Forward Pass (per observation):**
1. Flatten observations → `flat_obs`
2. Initialize `out` with previous z-scores, inject `flat_obs` at input indices
3. Standardize via `wrs(out)`
4. For each network pass (up to max):
   - Gather inputs: `mapped_out = out[in_nodes_indices]`
   - Weight & sum: `(mapped_out * weights).sum(dim=1)`
   - Update mutable nodes (masked by pass requirement)
   - Standardize via `wrs(out)`
5. Extract actions from output indices

**Key Techniques:**
- **Index 0 Trick** - Padding with 0 (always outputs 0) handles missing connections
- **torch.gather()** - Efficiently maps node outputs to inputs via indices
- **Pass Masking** - Different networks can have different recurrence depths
- **Batched Standardization** - WRS handles full population simultaneously

**Verbose Output:** Prints all intermediate tensors for educational purposes.

## Usage
Run `barebone_run()` to see step-by-step batched computation with detailed logging. This is reference code showing how to integrate evolution.py networks into a batched forward pass system.

## Implementation Note
This is test/example code. Production usage would extract the core patterns into a proper `DynamicNetPopulation` class (see main.py).
