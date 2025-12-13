# DL Module

Deep learning module for supervised imitation learning from human behavioral data.

## Purpose
Contains deep learning code for gradient-based training of neural networks on behavioral data.

## Structure
- `models/` - Neural network architectures (feedforward, recurrent, dynamic)
- `optim/` - Gradient-based optimizers (SGD with backpropagation)

## Usage
This module uses shared utilities:
- `config/` - Device management, paths, environment configurations
- `metrics/` - Evaluation metrics and behavioral comparison
- `data/process/` - Data loading and preprocessing
