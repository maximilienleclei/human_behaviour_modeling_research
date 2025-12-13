# State Persistence Configuration

Configuration for hidden state persistence and transfer modes in continual learning experiments.

## Purpose
Controls how recurrent network hidden states and environment states are managed across generations in evolutionary optimization, enabling continual learning experiments.

## Contents
- `StatePersistenceConfig` - Dataclass defining:
  - Hidden state persistence flags (persist_across_generations, persist_across_episodes, reset_on_selection)
  - Transfer mode flags (env_transfer, mem_transfer, fit_transfer)
  - Used by ne/optim/*.py optimizers to control state management
