# Population Wrapper

Bridge/adapter layer between network objects and eval/optim layers in the neuroevolution architecture. Provides tracking attributes for continual learning and environment transfer modes.

## Purpose

Provides algorithm-agnostic wrapper that **adapts** network representations for different optimizers:
- **GA**: Operates on network objects → Population provides `select_networks(indices)`
- **ES**: Operates on parameter vectors → Population provides `get/set_parameters_flat()`

Additionally provides **tracking attributes** for continual learning:
- Episode/evaluation metrics (scores, step counts)
- Global metrics (total steps, continual fitness)
- Environment state storage for env_transfer mode

**Key principle**: Population is the adapter - optimizers don't need to know network structure.

## Contents

### Population Class
Single class that wraps any network type (BatchedFeedforward, BatchedRecurrent, DynamicNetPopulation).

**New Tracking Attributes** (for continual learning):
- `curr_episode_score` [num_nets]: Score in current episode (for env_transfer)
- `curr_episode_num_steps` [num_nets]: Steps in current episode
- `curr_eval_score` [num_nets]: Score in current evaluation/generation
- `curr_eval_num_steps` [num_nets]: Steps in current evaluation
- `total_num_steps` [num_nets]: Total steps across all evaluations
- `continual_fitness` [num_nets]: Accumulated fitness across all generations
- `saved_env`: Saved environment state (for env_transfer mode)
- `saved_env_out`: Saved environment observation (for env_transfer mode)
- `logged_score`: Score to log (differs based on transfer mode)
- `curr_gen`: Current generation number (updated by optimizer)

**Adapter methods (for optimizers) - Protocol-based dispatch:**
- `select_networks(indices)` - Select networks by indices (for GA)
  - StructuralNetwork (DynamicNetPopulation): calls `select_and_duplicate()`
  - ParameterizableNetwork (Feedforward/Recurrent): calls `clone_network()`
- `get_parameters_flat()` - Flatten all parameters to vector (for ES/CMA-ES)
  - Only works with ParameterizableNetwork
  - Raises TypeError for StructuralNetwork
- `set_parameters_flat(flat_params)` - Unflatten and set parameters (for ES/CMA-ES)
  - Only works with ParameterizableNetwork
  - Raises TypeError for StructuralNetwork

**Service methods:**
- `get_actions(logits)` - Convert network outputs to actions (softmax/argmax/raw)
- `mutate()` - Delegate mutation to wrapped network
- `get_state_dict()` / `load_state_dict()` - Checkpointing support
  - Includes tracking attributes and network state
  - Environment state now serialized with pickle (no longer deepcopy)
  - Handles both tensor and non-tensor saved_env_out
- `reset_episode_tracking()` - Reset per-episode counters
- `reset_eval_tracking()` - Reset per-evaluation counters
- `reset_all_tracking()` - Reset all tracking attributes

**Properties:**
- `nets` - Access to underlying network object
- `num_nets` - Number of networks in population
- `action_mode` - How outputs are converted ("softmax", "argmax", "raw")

## Adapter Pattern

Population translates between what networks provide and what optimizers need:

```
GA needs: "select network objects"
    ↓
Population.select_networks(indices)  ← Adapter method
    ↓
Networks: batched tensors indexed by indices

ES needs: "average parameter vectors"
    ↓
Population.get_parameters_flat()     ← Adapter method
    ↓
Networks: batched tensors flattened to vectors
```

This means:
- **GA** never touches weights/biases - just picks indices
- **ES/CMA-ES** never know network structure - just average vectors
- **Networks** don't need selection methods - population handles translation

**Protocol-Based Design (NEW):** Population now uses protocol-based dispatch via isinstance() checks on NetworkProtocol, ParameterizableNetwork, and StructuralNetwork. This eliminates type-specific conditionals and enables clean extensibility. Each network type implements the appropriate protocol, and Population delegates to protocol methods.

## Usage

### For Training
```python
from ne.net.feedforward import BatchedFeedforward
from ne.pop.population import Population
from ne.eval.supervised import train_supervised

# Create network and wrap
nets = BatchedFeedforward([10, 20, 5], num_nets=100)
population = Population(nets, action_mode="softmax")

# High-level training
result = train_supervised(
    population=population,
    train_data=(obs, actions),
    test_data=(test_obs, test_actions),
    optimizer="ga",  # or "es"
    max_time=3600,
)
```

### Direct Adapter Usage (if implementing custom optimizer)
```python
# GA-style: operate on network objects
indices = torch.tensor([0, 1, 0, 1])  # Select networks 0,1 and duplicate
population.select_networks(indices)

# ES-style: operate on parameter vectors
params = population.get_parameters_flat()  # [num_nets, num_params]
avg_params = params.mean(dim=0)
population.set_parameters_flat(avg_params.unsqueeze(0).expand(num_nets, -1))
```
