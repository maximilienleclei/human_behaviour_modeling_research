# ga.py

## Purpose
Simple Genetic Algorithm optimizer for neuroevolution with hard selection (top 50% survive).

## Contents

### `optimize_ga(nets, optim_data, test_data, eval_fn, ...)`
Main GA optimization loop.

**Algorithm:**
1. Evaluate fitness on training data
2. Select top 50%, duplicate to fill population
3. Mutate all networks
4. Repeat until time limit

**Features:**
- Checkpoint resume/save every 5 minutes
- Periodic test evaluation
- Progress logging

**Generic:** Works with any batched network (feedforward/recurrent/dynamic) via `eval_fn` callback.

### `select_simple_ga(nets, fitness)`
Hard selection - top 50% survive, bottom 50% replaced. Delegates to network's `select_ga()` method.

### `save_checkpoint(...)`
Saves generation, histories, time, and network state to disk.

**Returns:** Dict with fitness_history, test_loss_history, final_generation.
