# Feedforward Neural Network Models

Simple two-layer MLP architecture for behavior modeling that works with both gradient-based (SGD) and gradient-free (evolutionary) optimization methods.

## What it's for

Provides a basic feedforward neural network baseline that can predict human actions from observations. Used for quick prototyping and as a simpler alternative to recurrent or dynamic models.

## What it contains

### Models
- `MLP` - Two-layer feedforward network with tanh activation (input → hidden → output)

### Methods
- `forward()` - Returns raw logits for action prediction
- `get_probs()` - Returns softmax probability distribution over actions

## Key Details

The architecture is intentionally simple (2 layers, tanh activation) to serve as a baseline. Unlike recurrent models, it has no memory of previous timesteps, making it suitable for tasks where current observation alone is sufficient for action prediction. Compatible with both backpropagation (platform/optimizers/sgd.py) and evolutionary methods (platform/optimizers/evolution.py).
