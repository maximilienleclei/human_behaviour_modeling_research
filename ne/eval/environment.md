# Environment-Based Evaluation Orchestration

High-level training interface for TorchRL environment-based tasks using neuroevolution.

## Purpose

Orchestrates episode rollouts and optimization for reinforcement learning / environment interaction tasks. Maintains same clean separation as supervised learning: optimizer never sees environment or episode data.

## Contents

### create_env_fitness_evaluator()
Creates fitness evaluation function (closure) for environment rollouts. Captures environment and rollout parameters internally, returns callable that evaluates population on episodes.

**Metrics**:
- `"return"`: Episode cumulative reward (for RL tasks)
- `"cross_entropy"`: Action prediction accuracy (for imitation learning)

### train_environment()
High-level training function for environment-based tasks:

1. Create fitness evaluators for train and test environments
2. Select optimizer (GA/ES/CMA-ES)
3. Run optimization loop with episode rollouts
4. Return training results

**New**: Accepts `state_config` parameter for continual learning modes (env_transfer, mem_transfer, fit_transfer).

**Design**: Same architecture as supervised learning but uses environment rollouts instead of static datasets.

## Usage

```python
from ne.net.recurrent import BatchedRecurrent
from ne.pop.population import Population
from ne.eval.environment import train_environment
from torchrl.envs import CartPoleEnv

# Create network and wrap in population
nets = BatchedRecurrent([4, 32, 2], num_nets=100, model_type="reservoir")
population = Population(nets, action_mode="argmax")

# Create TorchRL environments
train_env = CartPoleEnv()
test_env = CartPoleEnv()

# Configure continual learning (optional)
from ne.optim.base import StatePersistenceConfig

state_config = StatePersistenceConfig(
    env_transfer=True,  # Save environment state across generations
    fit_transfer=True,  # Accumulate fitness across all generations
    persist_across_generations=True,  # Save hidden states
)

# High-level training with episode rollouts
result = train_environment(
    population=population,
    train_env=train_env,
    test_env=test_env,
    num_episodes=10,
    max_steps_per_episode=200,
    metric="return",
    optimizer="ga",
    max_time=3600,
    state_config=state_config,  # Enable continual learning modes
)

print(f"Best episode return: {-result['fitness_history'][-1]}")
```

## Design Philosophy

Same clean separation as supervised learning:
- eval/ layer handles environments and rollouts
- optim/ layer handles selection/mutation
- pop/ layer bridges them
- Optimizer only sees fitness, never environment or episode data
