# Evaluation Layer

ALL evaluation logic - both low-level utilities and high-level orchestration.

## Purpose

Handles all evaluation concerns (data setup, environment configuration, fitness computation) while maintaining clean separation from optimization.

**Key architectural principle**: Optimizer never sees observations/actions - only fitness values.

## Files

### Low-Level Utilities
- **evaluate.py** - Cross-entropy evaluation functions (feedforward, recurrent, episodes, adversarial)
- **env.py** - TorchRL environment evaluation (episode rollouts, batch evaluation, continual learning support)
- **imitation.py** - Imitation learning and adversarial training (GAN-style generator-discriminator)

### High-Level Orchestration
- **supervised.py** - Supervised learning training orchestration
- **environment.py** - TorchRL environment-based training orchestration (with continual learning modes)
- **imitation.py** - Imitation learning training orchestration (train_imitation function)

## Architecture Role

eval/ is the **top layer** in the 4-layer architecture:

```
ne/eval/      ← High-level orchestration (YOU ARE HERE)
ne/pop/       ← Bridge (output conversion, population interface)
ne/optim/     ← Optimization (selection algorithms)
ne/net/       ← Networks (forward pass, mutation)
```

## How It Works

### Fitness Function Closures
eval/ creates fitness functions as **closures** that capture data internally:

```python
# In ne/eval/supervised.py
def create_fitness_evaluator(population, observations, actions):
    def fitness_fn():  # Closure
        return eval_fn(nets, observations, actions)  # Data captured here
    return fitness_fn  # Optimizer only sees this

# Optimizer calls it
fitness = fitness_fn()  # No data passed - optimizer doesn't see obs/actions!
```

### High-Level Training
User-facing training functions that handle everything:

```python
from ne.eval.supervised import train_supervised

result = train_supervised(
    population=population,
    train_data=(obs, actions),
    test_data=(test_obs, test_actions),
    optimizer="ga",  # or "es"
    max_time=3600,
)
```

## Design Benefits

1. **Clean separation**: Optimizer is data-agnostic
2. **Easy extensibility**: Add new evaluation modes without touching optimizers
3. **User-friendly**: Single function call handles entire training workflow
4. **Testable**: Each layer can be tested independently

## Usage Patterns

### Supervised Learning
```python
from ne.eval.supervised import train_supervised
# Handles: data → fitness functions → optimizer → results
```

### Environment-Based
```python
from ne.eval.environment import train_environment
# Handles: env → episode rollouts → fitness functions → optimizer → results
```

Both follow the same pattern: hide data from optimizer, provide clean high-level API.
