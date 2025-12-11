# Dynamic Network Computation

Population-wide batched forward computation for dynamic networks with online standardization.

## What it's for

Provides efficient GPU-parallel computation across populations of dynamic networks with varying topologies. Handles batched forward passes with Welford running standardization to stabilize training.

## What it contains

### Classes
- `WelfordRunningStandardizer` - Online mean/variance computation for z-score standardization of node activations

### Functions
- `barebone_run()` - Example demonstrating batched population forward pass

## Key Details

Uses custom acronyms: TNMN (Total Number of Mutable Nodes across all networks), NON (Number of Output Nodes), TNN (Total Number of Nodes). The Welford algorithm computes running mean and variance incrementally without storing all samples, enabling online standardization during network execution. Standardization is applied selectively: only updates statistics for new raw values (x), not for previously computed z-scores, using masking to track which values are new vs recycled. Computation handles variable-topology networks by padding to maximum network size and using masks. Networks can have cycles (graph-based recurrence) requiring multiple forward passes per input. This module provides the computational primitives used by platform/models/dynamic.py's DynamicNetPopulation wrapper.
