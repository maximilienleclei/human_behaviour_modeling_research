# Dynamic Network Models

Graph-based neural networks with evolving topology for evolutionary optimization (GA-exclusive, not differentiable).

Contains DynamicNetPopulation wrapping ne/net/dynamic/evolution.py's Net class with batched GPU computation interface. Methods: forward_batch() (parallel population evaluation), get_probs_batch(), set_population(), get_population(). Networks start minimal (input/output only) then apply initial_mutations (default 5). Variable topology per network requires padding/masking for batching. Networks can have cycles (graph-based recurrence) and multiple forward passes per input. Supports Welford running standardization for internal activations. Uses batched computation from ne/net/dynamic/compute_test.py. Works only with evolutionary optimizers (ne/optim/ga.py for recurrent, ne/optim/es.py as fallback).
