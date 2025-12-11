# Common Dynamic Network Directory

Core implementation of evolving graph-based neural networks with variable topology.

## Overview

This directory contains the fundamental building blocks for dynamic networks: node/network classes that support topology mutations, and population-wide batched computation primitives.

## Files

### Network Evolution
- **evolution.py** - Per-network topology evolution
  - `Node` - Network node (input/hidden/output roles)
  - `NodeList` - Ordered collection with mutable/immutable tracking
  - `Net` - Complete network with mutation operations
  - Mutations: grow node, grow connection, prune node, prune connection
  - Per-network logic (not batched) due to branching complexity

### Network Computation
- **computation.py** - Batched population-wide forward passes
  - `WelfordRunningStandardizer` - Online z-score standardization
  - `barebone_run()` - Example demonstrating batched evaluation
  - Handles variable-topology networks via padding and masking
  - Population-wide batching for efficient GPU utilization

## Architecture Details

**Nodes**: Three roles (input/hidden/output) with different properties. Hidden/output nodes have up to 3 incoming connections with frozen random weights, no biases, and apply standardization.

**Topology Evolution**: Networks start minimal (input/output only), grow through mutation. Cycles permitted (graph-based recurrence). Dual UID tracking: mutable (topology-dependent) and immutable (lifetime tracking).

**Computation**: Networks have variable structure requiring careful batching. Multiple forward passes per input possible due to cycles. Welford algorithm enables online standardization without storing all activations.

## Integration

Used by platform/models/dynamic.py which wraps this implementation to provide platform-compatible interface with population management and optimizer integration.
