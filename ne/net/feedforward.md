# Batched Feedforward Networks

Batched multi-layer MLPs for GPU-parallel neuroevolution.

Contains BatchedFeedforward with arbitrary depth via dimensions list [input_size, hidden1, ..., output_size]. Methods: forward_batch(x) via torch.bmm() → [num_nets, N, output_size], mutate() with adaptive/fixed sigma. Protocol methods (ParameterizableNetwork): get_parameters_flat(), set_parameters_flat(), clone_network(). Tanh activation between hidden layers, linear output. Xavier initialization. State management: get_state_dict(), load_state_dict() for checkpointing. Implements ParameterizableNetwork protocol from protocol.py for clean integration with Population and all optimizers (GA, ES, CMA-ES). Optional adaptive sigma tensors mirror parameter structure if sigma_noise provided. Network-specific batched computation only, evaluation/selection in ne/pop/ and ne/optim/.
