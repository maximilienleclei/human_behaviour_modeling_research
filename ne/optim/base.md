# base.py

## Purpose
Shared optimization loop for all evolutionary algorithms (GA, ES, CMA-ES). Eliminates 95% code duplication across optimizer implementations.

**Key architectural change**: Optimizer receives fitness functions (closures), never sees observations/actions directly. This enables clean separation between evaluation and optimization.

## Contents

### `optimize()`
Main training loop shared by all evolutionary algorithms. Handles:
- Checkpoint save/load with resumption
- Time-based training (max_time limit)
- Periodic test evaluation (eval_interval)
- Fitness tracking and logging
- Recurrent state persistence across generations (optional)

**New signature:**
```python
def optimize(
    population,              # Population wrapper (not raw nets)
    fitness_fn,             # Closure: () -> fitness [num_nets]
    test_fitness_fn,        # Closure: () -> fitness [num_nets]
    selection_fn,           # (population, fitness) -> None
    algorithm_name: str,
    ...
)
```

**Key design**: `fitness_fn` and `test_fitness_fn` are closures created by eval layer. They capture obs/actions internally, so optimizer never sees raw data.

**Flow:**
1. Resume from checkpoint if exists
2. Loop until time limit:
   - Restore hidden states (if persisting across generations)
   - Call `fitness_fn()` - no data passed!
   - Call `selection_fn(population, fitness)` - algorithm-specific selection
   - Call `population.mutate()` - apply mutations
   - Reset/save hidden states based on config
   - Periodic test eval & logging
   - Checkpoint every 5 minutes
3. Final checkpoint and return history

### `StatePersistenceConfig`
Configuration for state persistence and transfer modes in continual learning.

**Note:** `StatePersistenceConfig` is now defined in `config/state.py` (shared between dl/ and ne/ modules).

Used to control:
- Hidden state persistence across generations
- Episode/environment transfer modes
- Continual fitness accumulation

See `config/state.py` for full documentation.

### `save_checkpoint()`
Saves optimization state including:
- Generation number, fitness/test histories
- Elapsed time, algorithm name
- Population state dict (includes network state)
- Hidden states (if persisting)
- **CMA-ES state (NEW):** If using CMA-ES, saves complete algorithm state (mean, sigma, C_diag, evolution paths, samples)

**Checkpoint Resume:** When resuming from checkpoint, base.optimize() restores:
- Fitness/test histories and generation number
- Population state (network parameters)
- Hidden states (if state persistence enabled)
- **CMA-ES state (NEW):** Recreates CMAESState object with all parameters for seamless continuation

## Integration

Each optimizer (GA/ES/CMA-ES) defines selection logic and wraps base.optimize():

```python
# ne/optim/ga.py
def select_ga(population, fitness):
    # GA-specific selection logic HERE
    # Operates on population, modifies in-place
    pass

def optimize_ga(population, fitness_fn, test_fitness_fn, ...):
    return optimize(
        population=population,
        fitness_fn=fitness_fn,
        test_fitness_fn=test_fitness_fn,
        selection_fn=select_ga,  # Defined above
        algorithm_name="ga",
        ...
    )
```

**Clean separation:**
- eval/ layer creates fitness_fn closures (captures data)
- optim/ layer calls fitness_fn() (never sees data)
- pop/ layer bridges networks and optimizers

**Benefits:**
- Optimizer is data-agnostic (works with any fitness function)
- Easy to add new evaluation modes (env-based, adversarial, etc.)
- Single source of truth for training loop
- ~500 lines → ~150 lines total across all optimizers
