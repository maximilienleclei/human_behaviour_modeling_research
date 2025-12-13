# Config Module

Shared configuration utilities used across both deep learning (dl/) and neuroevolution (ne/) modules.

## Purpose
Centralizes all experiment configuration to avoid code duplication between dl/ and ne/.

## Contents
- `device.py` - Global DEVICE variable and device management
- `paths.py` - Project-wide directory paths
- `state.py` - State persistence configuration for continual learning
- `experiments.py` - Experiment configuration dataclasses (model, optimizer, training, data configs)
