# Neuroevolution Module

Efficient GPU-parallel evolutionary optimization for neural networks with TorchRL integration, state persistence, and adversarial learning support.

## Structure

### net/
Network architectures with batched implementations:
- **feedforward.py** - BatchedFeedforward (multi-layer MLPs with configurable depth)
  - State dict methods for checkpointing: `get_state_dict()`, `load_state_dict()`
- **recurrent.py** - BatchedRecurrent (stacked RNNs with all layers recurrent, reservoir or trainable rank-1)
  - Hidden state persistence: `save_hidden_states()`, `restore_hidden_states()`, `reset_hidden_states()`
  - State dict methods including hidden states for full checkpoint support
- **dynamic/** - DynamicNetPopulation (evolving topology with graph mutations)

### eval/
Evaluation layer with low-level utilities and high-level orchestration:
- **env.py** - TorchRL environment evaluation with continual learning support
  - `evaluate_env_episodes()` for episode rollouts with env/mem/fit transfer modes
  - `evaluate_env_batch()` for offline evaluation on pre-recorded trajectories
- **imitation.py** - Imitation learning and adversarial training
  - `evaluate_imitation_episode()` for GAN-style generator-discriminator evaluation
  - `train_imitation()` for high-level imitation learning training
- **environment.py** - High-level train_environment() orchestration
- **supervised.py** - High-level train_supervised() orchestration

### pop/
Network-agnostic population utilities:
- **population.py** - Population wrapper with tracking attributes for continual learning
  - Tracking: curr_episode_score, curr_eval_score, total_num_steps, continual_fitness
  - Environment state storage: saved_env, saved_env_out
  - Adapter methods for optimizers: select_networks(), get/set_parameters_flat()

### optim/
Evolutionary optimization algorithms with unified base:
- **base.py** - Shared `optimize()` function handling checkpointing, time tracking, and state persistence
  - `StatePersistenceConfig` for state persistence and transfer modes (env/mem/fit transfer)
- **ga.py** - Simple Genetic Algorithm (hard selection, top 50%)
- **es.py** - Evolution Strategy (soft selection, rank-based)
- **cmaes.py** - CMA-ES (adaptive search distribution)

All optimizers accept `state_config` parameter for state persistence and continual learning modes.

## Design Principles

**Batched Computation:** All networks stored as [num_nets, ...] tensors for parallel GPU operations.

**Separation of Concerns:**
- net/ = network-specific forward/mutate
- pop/ = generic utilities
- optim/ = optimization algorithms

**Adaptive Mutation:** Per-parameter mutation strength (sigma) co-evolves with parameters.

## Key Features

### Continual Learning and Transfer Modes
`StatePersistenceConfig` now supports both hidden state persistence and environment/fitness transfer modes from old gen_transfer.py:

**Hidden State Persistence** (for recurrent networks):
- **persist_across_generations**: Save/restore states between optimization steps
- **persist_across_episodes**: Maintain states across episode boundaries (meta-learning)
- **reset_on_selection**: Control state reset after selection phase

**Transfer Modes** (for continual learning):
- **env_transfer**: Save/restore environment state across generations
  - Episodes can span multiple generations
  - Environment continues from where previous generation left off
- **mem_transfer**: Keep agent memory (hidden states) between episodes
  - Don't reset agent when episode ends (done=True)
- **fit_transfer**: Accumulate fitness across all generations
  - Optimize continual_fitness instead of per-generation score
  - Fitness accumulates across entire evolutionary run

### TorchRL Environment Integration
`eval/env.py` provides:
- Episode rollout evaluation with continual learning support
- Environment/memory/fitness transfer modes
- Configurable metrics: episode return or cross-entropy
- State persistence for recurrent networks across episodes
- Offline evaluation on pre-recorded trajectories
- Population tracking attributes (scores, steps, continual fitness)

### Imitation Learning and Adversarial Training
`eval/imitation.py` implements GAN-style training from old_imitate.py:
- Two-population training: generator (imitator) and discriminator
- Alternating evaluation: generator vs discriminator, then target vs discriminator
- Flexible element hiding for discriminator (hide_fn or hide_indices for privileged info like score/lives)
- Merge mode for fair fitness scaling
- Supports same transfer modes as environment evaluation
- High-level `train_imitation()` orchestration function

### Optimizer Refactoring
All optimizers (GA/ES/CMA-ES) now share common `base.optimize()` implementation:
- Eliminates 95% code duplication
- Consistent checkpointing, time tracking, and state persistence
- Algorithm-specific selection strategies injected via `selection_fn`
