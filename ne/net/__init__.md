# Network Architectures for Neuroevolution

Batched network implementations for efficient GPU-parallel evolutionary optimization.

## Files

### Protocol Definitions
- **protocol.py** - Protocol interfaces for network types using structural subtyping
  - NetworkProtocol: Base protocol for all networks
  - ParameterizableNetwork: For ES/CMA-ES compatible networks (feedforward, recurrent)
  - StructuralNetwork: For topology-evolving networks (dynamic)

### Feedforward Networks
- **feedforward.py** - BatchedFeedforward: Multi-layer MLP population with batched parameters and adaptive sigma mutation

### Recurrent Networks
- **recurrent.py** - BatchedRecurrent: Recurrent MLP population supporting both frozen reservoir and trainable rank-1 recurrent weights

### Dynamic Topology Networks
- **dynamic/** - Evolving graph-based networks with topology mutations (grow/prune nodes and connections)

## Design

All classes store num_nets networks as batched tensors for parallel forward passes and mutations. Network-specific computation only - evaluation, selection, and optimization loops belong in ne/pop/ and ne/optim/. Networks implement protocol interfaces for clean integration with Population adapter and optimizers.
