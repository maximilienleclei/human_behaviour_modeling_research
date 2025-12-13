# feedforward.py

## Purpose
Batched multi-layer feedforward MLPs for GPU-parallel neuroevolution across num_nets networks.

## Contents

### `BatchedFeedforward`
Multi-layer MLP population with arbitrary depth, stored as batched tensors [num_nets, ...].

**Parameters:**
- `dimensions`: List defining network architecture [input_size, hidden1, hidden2, ..., output_size]
- Weights/biases stored as lists, one tensor per layer
- Optional adaptive sigma tensors mirror parameter structure if sigma_noise provided

**Methods:**
- `forward_batch(x)`: Parallel forward pass via torch.bmm() through all layers → [num_nets, N, output_size]
- `mutate()`: Adaptive (if sigma_noise set) or fixed sigma mutation across all layers

**Protocol Methods (ParameterizableNetwork):**
- `get_parameters_flat()`: Flatten all weights/biases → [num_nets, num_params] for ES/CMA-ES
- `set_parameters_flat(flat_params)`: Set parameters from flat tensor (used by ES/CMA-ES)
- `clone_network(indices)`: Clone networks at specified indices for GA selection

**Architecture:**
- N layers created from dimensions list (N = len(dimensions) - 1)
- Tanh activation between hidden layers, linear output layer
- Example: `dimensions=[10, 64, 32, 4]` creates 10→64→32→4 network

**Initialization:** Xavier initialization for all layer weights and biases.

**State Management:**
- `get_state_dict()`: Save full network state for checkpointing
- `load_state_dict(state)`: Restore network from checkpoint
- Includes dimensions, weights, biases, and adaptive sigmas

**Protocol Compliance:** Implements ParameterizableNetwork protocol from ne/net/protocol.py, enabling clean integration with Population adapter and all optimizers (GA, ES, CMA-ES).

Network-specific batched computation only. Evaluation/selection in ne/pop/ and ne/optim/.
