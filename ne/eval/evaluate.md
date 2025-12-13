# evaluate.py

## Purpose
Generic cross-entropy evaluation for batched network populations - network-agnostic.

## Contents

### `evaluate_feedforward(nets, observations, actions)`
Evaluates feedforward networks via batched forward pass + cross-entropy.
- Input: obs [N, input_size], actions [N]
- Returns: fitness [num_nets] (mean CE per network)

### `evaluate_recurrent(nets, observations, actions)`
Evaluates recurrent networks via sequence forward pass + cross-entropy.
- Input: obs [seq_len, input_size], actions [seq_len]
- Resets hidden state (h_0=None) for each sequence
- Returns: fitness [num_nets]

### `evaluate_episodes(nets, episodes, eval_fn)`
Evaluates on multiple episodes and averages fitness.
- episodes: list of dicts with 'observations' and 'actions'
- eval_fn: evaluate_feedforward or evaluate_recurrent
- Returns: mean fitness across episodes [num_nets]

### `evaluate_adversarial(nets, generator_data, discriminator_data, action_size)`
Evaluates networks with split outputs for adversarial/imitation learning.
- Networks output combined logits: [:action_size] for actions, [action_size:] for discrimination
- generator_data: (observations, actions) for action prediction task
- discriminator_data: (observations, labels) for real/fake classification (0=fake, 1=real)
- Returns: (gen_fitness, disc_fitness) both [num_nets]
- Use case: Single network learns both action generation and behavior discrimination

**Generic:** Works with any network implementing forward_batch() or forward_batch_sequence().
