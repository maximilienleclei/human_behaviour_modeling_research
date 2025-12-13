# es.py

## Purpose
Evolution Strategy optimizer for neuroevolution with soft selection (rank-based weighting).

## Contents

### `optimize_es(nets, optim_data, test_data, eval_fn, ...)`
Main ES optimization loop.

**Algorithm:**
1. Evaluate fitness on training data
2. Rank-based soft selection (all networks contribute weighted by rank)
3. Mutate all networks
4. Repeat until time limit

**Difference from GA:** Soft selection instead of hard truncation - better networks get higher weight but all contribute.

**Generic:** Works with any batched network via `eval_fn` callback.

### `select_simple_es(nets, fitness)`
Rank-based soft selection. Delegates to network's `select_es()` method.

### `save_checkpoint(...)`
Saves generation, histories, time, and network state.

**Returns:** Dict with fitness_history, test_loss_history, final_generation.
