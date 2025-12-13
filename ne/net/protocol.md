# Network Protocols

Protocol interfaces for network types using structural subtyping.

NetworkProtocol (base: num_nets, device, forward_batch(), mutate(), get_state_dict(), load_state_dict()), ParameterizableNetwork (adds get_parameters_flat(), set_parameters_flat(), clone_network() for ES/CMA-ES), StructuralNetwork (adds select_and_duplicate() for topology evolution). Python Protocol for duck typing with type checking. Population checks conformance to dispatch methods.
