# Platform Models Directory

Neural network architectures for behavior modeling, supporting both gradient-based and gradient-free optimization.

## Overview

This directory contains the model implementations used across the platform. Models are categorized as either "shared" (work with both SGD and evolutionary methods) or "GA-exclusive" (only evolutionary methods).

## Files

### Feedforward Models
- **feedforward.py** - Simple two-layer MLP baseline with tanh activation
  - `MLP` class for basic feedforward networks
  - Works with both SGD and evolutionary optimizers

### Recurrent Models
- **recurrent.py** - RNN architectures with memory across timesteps
  - `RecurrentMLPReservoir` - Fixed random recurrent weights (echo state network)
  - `RecurrentMLPTrainable` - Trainable rank-1 recurrent weights
  - Both work with SGD and evolutionary optimizers

### Dynamic Networks
- **dynamic.py** - Evolving graph-based neural networks (GA-exclusive)
  - `DynamicNetPopulation` - Population wrapper with batched GPU computation
  - Topology evolves through mutation (grow/prune nodes and connections)
  - Only works with evolutionary optimizers due to non-differentiable topology changes

### Module Interface
- **__init__.py** - Model factory and exports
  - `create_model()` - Factory function for dynamic model instantiation
  - Exports all model classes for convenient importing

## Usage

Models are instantiated via the factory pattern in platform/runner.py. All models implement common interface: `forward()` for logits and `get_probs()` for action probabilities.
