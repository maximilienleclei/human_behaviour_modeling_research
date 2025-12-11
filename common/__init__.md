# Common Directory

Shared utilities and specialized implementations used across experiments.

## Overview

This directory contains code that is reusable across multiple experiments but doesn't fit into the main platform layer. Currently focused on dynamic network implementation.

## Directory Structure

### dynamic_net/
Graph-based neural networks with evolving topology:
- **evolution.py** - Network and node classes with mutation operations
- **computation.py** - Batched population-wide forward computation

The dynamic network implementation is kept separate from platform/ because:
1. It has specialized requirements (per-network mutation logic, Welford standardization)
2. It predates the unified platform architecture
3. It serves as a research prototype for topology evolution

## Integration

Dynamic networks are integrated into the platform via platform/models/dynamic.py, which wraps the common/dynamic_net implementation to provide a platform-compatible interface.

## Future Extensions

The common/ directory can house other shared utilities as the project grows:
- Custom loss functions used across experiments
- Data augmentation techniques
- Specialized network layers
- Analysis utilities

## Usage

Most users interact with dynamic networks through platform/models/dynamic.py rather than directly using common/dynamic_net.
