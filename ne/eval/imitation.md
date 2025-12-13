# Imitation Learning and Adversarial Training

GAN-style training for learning to imitate target behavior using generator-discriminator populations.

## Purpose

Implements imitation learning where a generator network learns to imitate a target agent's behavior, while a discriminator network learns to distinguish between the generator's actions and the target's actions.

Inspired by old_imitate.py architecture with two-population adversarial training.

## Contents

### hide_elements()
Utility function to hide specific elements from observations for imitation tasks. The discriminator should learn to distinguish behavior without seeing certain privileged information (e.g., game score, internal state, remaining lives) that would make the task trivial.

**Flexible hiding**:
- `hide_fn`: Custom callable(obs) -> obs for arbitrary transformations
- `hide_indices`: List of indices to zero out
- Default: No hiding (returns obs unchanged)

### evaluate_imitation_episode()
Core evaluation function that implements two-phase adversarial training:

**Phase 1**: Generator takes actions → Discriminator scores them
- Generator controls the environment
- Discriminator observes (with score hidden) and outputs probability that behavior is from target

**Phase 2**: Target takes actions → Discriminator scores them
- Target agent controls the environment
- Discriminator observes and scores

**Fitness Computation**:
- Generator fitness = p_target (phase 1) - wants discriminator to think it's target
- Discriminator fitness = -p_target (phase 1) + p_target (phase 2) - wants to distinguish

Supports:
- State persistence for recurrent networks
- Continual fitness accumulation (fit_transfer mode)
- Merge mode (scales generator fitness as fitness * 2 - 1)
- Flexible element hiding for discriminator (via hide_fn or hide_indices)

### create_imitation_fitness_evaluators()
Creates fitness evaluation closures for both populations. Returns tuple of (gen_fitness_fn, disc_fitness_fn) that can be passed to optimizers.

### train_imitation()
High-level training interface for imitation learning:

1. Creates fitness evaluators for both populations
2. Selects optimizer (GA/ES/CMA-ES)
3. Trains generator and discriminator (currently sequential, could be interleaved)
4. Returns results for both populations

**Design Note**: Currently trains populations sequentially (generator first, then discriminator). For true adversarial training, populations should be trained in alternating steps, but this requires modifications to the optimizer architecture.

## Usage

```python
from ne.net.recurrent import BatchedRecurrent
from ne.pop.population import Population
from ne.eval.imitation import train_imitation
from ne.optim.base import StatePersistenceConfig

# Create generator and discriminator networks
gen_nets = BatchedRecurrent([obs_size, 64, action_size], num_nets=50)
disc_nets = BatchedRecurrent([obs_size, 64, 1], num_nets=50)

gen_pop = Population(gen_nets, action_mode="argmax")
disc_pop = Population(disc_nets, action_mode="raw")

# Define target agent to imitate
def target_agent(obs):
    # Your expert agent logic here
    return optimal_action

# Train with imitation learning
state_config = StatePersistenceConfig(
    persist_across_generations=True,
    fit_transfer=True,  # Accumulate fitness across all generations
)

results = train_imitation(
    generator_pop=gen_pop,
    discriminator_pop=disc_pop,
    target_agent=target_agent,
    train_env=train_env,
    test_env=test_env,
    max_steps=1000,
    # Hide score and lives from discriminator
    hide_indices=[3, 7],  # Or use hide_fn=lambda obs: custom_hiding(obs)
    optimizer="ga",
    max_time=3600,
    state_config=state_config,
    merge_mode=False,
)

print(f"Generator fitness: {results['generator']['fitness_history'][-1]}")
print(f"Discriminator fitness: {results['discriminator']['fitness_history'][-1]}")
```

## Design Philosophy

Follows same clean separation as other eval/ modules:
- Optimizer only sees fitness values, never environment or episode data
- Closures capture evaluation details (target agent, environment, etc.)
- High-level train_imitation() provides user-friendly API

## Future Improvements

- Interleaved training: Alternate generator/discriminator updates within single optimization loop
- Multi-emulator support: Use different environments for different phases (like old_imitate.py)
- Vectorized environments: Evaluate multiple networks in parallel instead of sequentially
