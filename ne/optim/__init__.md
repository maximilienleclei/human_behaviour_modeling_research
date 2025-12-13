# Evolutionary Optimization Algorithms

Neuroevolution optimizers for training neural networks via evolutionary algorithms.

## Files

### Shared Base
- **base.py** - Shared optimization loop for all algorithms
  - `optimize()` - Common training loop (checkpointing, time tracking, evaluation)
  - `StatePersistenceConfig` - Recurrent state persistence configuration
  - Eliminates 95% code duplication across GA/ES/CMA-ES

### Genetic Algorithm
- **ga.py** - Simple GA with hard selection (top 50% survive, duplicate to fill population)
- **ga.md** - Documentation

### Evolution Strategy
- **es.py** - Simple ES with soft selection (rank-based weighting, all networks contribute)
- **es.md** - Documentation

### CMA-ES
- **cmaes.py** - Covariance Matrix Adaptation ES (adapts search distribution, diagonal approximation)
- **cmaes.md** - Documentation

## Design

### Unified Interface
All optimizers follow same interface:
- `optimize_*(nets, optim_data, test_data, eval_fn, ..., state_config=None)`
- Return dict with fitness_history, test_loss_history, final_generation
- Support checkpointing with resume
- Support state persistence for recurrent networks (optional)
- Work with any batched network via `eval_fn` callback

### Architecture
Each optimizer wraps `base.optimize()` with algorithm-specific selection:
```python
# ga.py
def optimize_ga(...):
    return optimize(..., selection_fn=select_and_mutate_ga, algorithm_name="ga")
```

**Selection:** GA=hard truncation, ES=rank-weighted soft, CMA-ES=distribution adaptation.

**Generic:** Optimizers are network-agnostic. Networks implement algorithm-specific methods (select_ga, select_es, update_cmaes).

### State Persistence (NEW)
Optional recurrent network state persistence via `StatePersistenceConfig`:
- `persist_across_generations`: Save/restore hidden states between optimization steps
- `persist_across_episodes`: Maintain state during evaluation (used in env rollouts)
- `reset_on_selection`: Reset states after selection, before mutation
