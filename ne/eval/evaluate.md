# Generic Cross-Entropy Evaluation

Network-agnostic cross-entropy evaluation for batched populations.

Contains evaluate_feedforward() for batched forward pass, evaluate_recurrent() for sequence forward pass (resets hidden state h_0=None per sequence), evaluate_episodes() for multi-episode averaging, and evaluate_adversarial() for split-output networks ([:action_size] for actions, [action_size:] for real/fake discrimination). All functions return fitness tensors [num_nets]. Works with any network implementing forward_batch() or forward_batch_sequence(). Used by supervised.py and imitation.py for population fitness computation.
