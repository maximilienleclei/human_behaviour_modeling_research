# Supervised Learning Orchestration

High-level training interface for supervised learning tasks using neuroevolution.

## Purpose

Orchestrates evaluation and optimization while maintaining clean separation of concerns. The key innovation: optimizer never sees observations/actions, only fitness values.

## Contents

### create_fitness_evaluator()
Creates a fitness evaluation function (closure) that captures observations and actions internally. Returns a callable that takes no arguments and returns fitness tensor.

**Key design**: The closure hides data from optimizer - optimizer just calls `fitness_fn()` and gets back fitness values, never seeing the underlying obs/actions.

### train_supervised()
High-level training function that orchestrates the entire supervised learning workflow:

1. Create fitness evaluators (closures) for train and test data
2. Select optimizer (GA/ES/CMA-ES) based on user choice
3. Run optimization loop
4. Return training results

**Clean separation**: eval layer handles data, optim layer handles selection/mutation, pop layer bridges them.

## Design Philosophy

Traditional design (bad):
```python
# Optimizer sees data directly
optimizer.train(nets, obs, actions)  # Tight coupling
```

New design (good):
```python
# Optimizer only sees fitness
fitness_fn = create_fitness_evaluator(pop, obs, actions)  # Closure
optimizer.train(pop, fitness_fn)  # Clean separation
```

Benefits:
- Optimizer is data-agnostic (works with any fitness function)
- Easy to add new evaluation modes (env-based, adversarial, etc.)
- Clear boundaries between layers

## Usage

```python
from ne.net.feedforward import BatchedFeedforward
from ne.pop.population import Population
from ne.eval.supervised import train_supervised

# Create network and wrap in population
nets = BatchedFeedforward([10, 20, 5], num_nets=100)
population = Population(nets, action_mode="softmax")

# High-level training - handles everything
result = train_supervised(
    population=population,
    train_data=(train_obs, train_actions),
    test_data=(test_obs, test_actions),
    optimizer="ga",  # or "es", "cmaes"
    max_time=3600,
)

print(f"Final fitness: {result['fitness_history'][-1]}")
```
