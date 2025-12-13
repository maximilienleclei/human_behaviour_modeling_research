# recurrent.py

## Purpose
Batched stacked recurrent MLPs for GPU-parallel neuroevolution. All layers (including output) are recurrent. Supports frozen reservoir or trainable rank-1 recurrent weights.

## Contents

### `BatchedRecurrent`
Stacked RNN population with all layers recurrent, configured via `dimensions` list parameter.

**Modes:**
- `reservoir`: Frozen W_hh [num_nets, layer_size, layer_size] for each layer, echo state style
- `trainable`: Rank-1 u⊗v^T recurrent weights (100 params vs 2,500 for full matrix) for each layer

**Parameters:**
- `dimensions`: List defining network architecture [input_size, layer1, layer2, ..., output_size]
- All layers have W_ih (input-to-hidden) and recurrent connections (W_hh or u/v)
- Optional adaptive sigma tensors mirror parameter structure if sigma_noise provided

**Methods:**
- `forward_batch_step(x, h_states)`: Single timestep parallel forward, returns (output, h_new_states)
- `forward_batch_sequence(x, h_0)`: Full sequence, processes timestep-by-timestep
- `mutate()`: Adaptive (if sigma_noise set) or fixed sigma mutation (skips frozen W_hh for reservoir)

**Protocol Methods (ParameterizableNetwork):**
- `get_parameters_flat()`: Flatten W_ih weights/biases + u/v (trainable mode) → [num_nets, num_params]
- `set_parameters_flat(flat_params)`: Set parameters from flat tensor (used by ES/CMA-ES)
- `clone_network(indices)`: Clone networks including W_hh/u/v and hidden states for GA selection

**Architecture:**
- N recurrent layers based on dimensions list (N = len(dimensions) - 1)
- Each layer: x → W_ih → [+ W_hh @ h_prev] → tanh → h_new
- Output is last layer's activation (also recurrent)
- Example: `dimensions=[10, 64, 32, 4]` creates 10→64→32→4, all layers recurrent

**Initialization:** Xavier for W_ih, scaled random for W_hh/u/v per layer. Device handling fixed for consistent GPU/CPU behavior.

**State Persistence:**
- `save_hidden_states()`: Save current hidden states for all layers
- `restore_hidden_states(states)`: Restore hidden states from previous evaluation
- `reset_hidden_states()`: Reset all hidden states to zero
- `get_state_dict()`: Save full network state including hidden states
- `load_state_dict(state)`: Restore network from checkpoint
- Enables continual learning across generations/episodes

**Protocol Compliance:** Implements ParameterizableNetwork protocol from ne/net/protocol.py, enabling clean integration with Population adapter and all optimizers (GA, ES, CMA-ES).

Maintains list of hidden states (one per layer). Network-specific logic only.
