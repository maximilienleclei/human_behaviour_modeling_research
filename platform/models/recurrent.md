# Recurrent Neural Network Models

Recurrent MLP architectures with two variants: frozen reservoir (echo state network) and trainable recurrent connections (rank-1 factorization).

## What it's for

Provides recurrent models that can maintain memory across timesteps, enabling learning of temporal dependencies in human behavior. Both variants work with SGD and evolutionary optimizers.

## What it contains

### Models
- `RecurrentMLPReservoir` - Recurrent MLP with frozen random reservoir (2500 fixed params + 452 trainable params for CartPole+CL)
- `RecurrentMLPTrainable` - Recurrent MLP with trainable rank-1 recurrent matrix (552 trainable params for CartPole+CL)

### Methods
Both models provide:
- `forward_step()` - Single timestep forward pass with hidden state update
- `forward()` - Sequence forward pass over multiple timesteps
- `get_probs()` - Returns softmax probability distribution over actions

## Key Details

RecurrentMLPReservoir uses echo state network approach with frozen recurrent weights (W_hh), making it parameter-efficient while still capturing temporal dynamics. RecurrentMLPTrainable uses rank-1 factorization (W_hh = u ⊗ v^T) to make recurrent connections trainable while keeping parameter count manageable (~22% more params than reservoir). Both models support variable-length sequences and maintain hidden state across timesteps. Architecture: input → recurrent hidden layer (50 units, tanh) → output. Used with EpisodeDataset (platform/data/preprocessing.py) for proper hidden state reset between episodes.
