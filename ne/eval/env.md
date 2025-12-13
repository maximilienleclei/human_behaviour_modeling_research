# TorchRL Environment Evaluation

TorchRL-based environment evaluation with continual learning support via environment/memory/fitness transfer modes.

## Purpose

Provides evaluation of populations on TorchRL environments with support for:
- Environment state transfer (env_transfer): Save/restore environment across generations
- Memory transfer (mem_transfer): Control agent hidden state reset
- Fitness transfer (fit_transfer): Accumulate fitness across all generations
- Episode/eval tracking via population attributes

Based on old gen_transfer.py architecture.

## Contents

### evaluate_env_episodes()
**Now accepts `population` instead of `nets`** to access tracking attributes.

Evaluates population on environment with continual learning support. Three transfer modes:

**env_transfer**: Save/restore environment state
- When True: Episodes span multiple generations
- Environment state and observation saved between generations
- Evaluation continues from where previous generation left off

**mem_transfer**: Keep agent memory between episodes
- When True: Don't reset hidden states when episode ends (done=True)
- Agent memory persists through episode boundaries

**fit_transfer**: Accumulate fitness across generations
- When True: Optimize continual_fitness instead of curr_eval_score
- Fitness accumulates across entire evolutionary run

**Tracking Updates**:
- `population.curr_eval_score`: Score in current evaluation
- `population.curr_eval_num_steps`: Steps in current evaluation
- `population.total_num_steps`: Total steps across all evaluations
- If env_transfer: `population.curr_episode_score`, `population.curr_episode_num_steps`
- If fit_transfer: `population.continual_fitness`

**Reset Behavior**:
- env_transfer=False: Reset environment each generation
- mem_transfer=False: Reset agent when done=True
- fit_transfer=True: Return continual_fitness instead of curr_eval_score

**Flow:**
1. Pre-eval reset based on transfer modes
2. Evaluation loop with tracking updates
3. Handle done=True based on env_transfer
4. Post-eval: Save environment state if needed
5. Return fitness (continual_fitness or curr_eval_score)

Currently executes action from first network due to single environment instance.

### evaluate_env_batch()
Offline evaluation on pre-recorded trajectories. Networks predict actions for recorded observations, fitness computed as cross-entropy against ground truth actions. Works with both feedforward and recurrent networks (recurrent treats batch as sequence).

## Usage

```python
from ne.eval.env import evaluate_env_episodes
from ne.optim.base import StatePersistenceConfig
from ne.pop.population import Population

# Create population
nets = BatchedRecurrent([obs_size, 64, action_size], num_nets=50)
population = Population(nets, action_mode="argmax")

# Configure continual learning
state_config = StatePersistenceConfig(
    env_transfer=True,      # Save/restore environment across generations
    mem_transfer=True,      # Keep agent memory between episodes
    fit_transfer=True,      # Accumulate fitness across all generations
    persist_across_generations=True,  # Save hidden states between generations
)

# Evaluate (called by optimizer, curr_gen updated automatically)
fitness = evaluate_env_episodes(
    population, env, num_episodes=10, max_steps_per_episode=1000,
    metric="return", state_config=state_config, curr_gen=population.curr_gen
)
```

## Transfer Mode Combinations

**Standard RL** (env_transfer=False, mem_transfer=False, fit_transfer=False):
- Environment resets each generation
- Agent resets when episode ends
- Optimize per-generation score

**Continual RL** (env_transfer=True, mem_transfer=True, fit_transfer=True):
- Environment state persists across generations
- Agent memory persists through episode resets
- Optimize accumulated fitness across all generations
- Episodes can span multiple generations

**Lifelong Learning** (env_transfer=True, mem_transfer=False, fit_transfer=True):
- Environment persists but agent resets when episode ends
- Optimize accumulated fitness
- Tests ability to quickly adapt to new episodes
