# Recurrent Neural Network Models

Recurrent MLPs with frozen reservoir (RecurrentMLPReservoir: echo state network with fixed W_hh) and trainable rank-1 factorization (RecurrentMLPTrainable: W_hh = u ⊗ v^T).

Contains forward_step() for single timestep, forward() for sequences, and get_probs(). Input → recurrent hidden (50 units, tanh) → output. RecurrentMLPReservoir: 2500 fixed + 452 trainable params; RecurrentMLPTrainable: 552 trainable params (CartPole+CL). Supports variable-length sequences with hidden state maintenance. Used with EpisodeDataset for proper episode boundaries. Compatible with SGD and evolutionary optimizers.
