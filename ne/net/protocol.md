# Network Protocols

Defines protocol interfaces for all network types in the neuroevolution framework, establishing contracts without requiring inheritance.

## Purpose

Enables clean polymorphism in the Population adapter by defining what methods all networks must implement. Uses Python's `Protocol` for structural subtyping (duck typing with type checking) instead of abstract base classes.

## Contains

### NetworkProtocol
Base protocol implemented by all networks (BatchedFeedforward, BatchedRecurrent, DynamicNetPopulation):
- **Attributes**: `num_nets`, `device`
- **Methods**: `forward_batch()`, `mutate()`, `get_state_dict()`, `load_state_dict()`

### ParameterizableNetwork
Protocol for networks with fixed architecture (BatchedFeedforward, BatchedRecurrent only):
- **Inherits from**: NetworkProtocol
- **Additional methods**: `get_parameters_flat()`, `set_parameters_flat()`, `clone_network()`
- **Purpose**: Enables ES/CMA-ES optimizers that require parameter averaging
- **NOT implemented by**: DynamicNetPopulation (variable topology can't be averaged)

### StructuralNetwork
Protocol for networks with evolving topology (DynamicNetPopulation only):
- **Inherits from**: NetworkProtocol
- **Additional methods**: `select_and_duplicate()`
- **Purpose**: Enables GA-style selection with topology evolution
- **Why separate**: Topology evolution requires special handling during network duplication

## Design Pattern

**Structural subtyping** via `@runtime_checkable` Protocol:
- No inheritance required (each network class is independent)
- Static type checking support (mypy, pyright)
- Runtime `isinstance()` checks still work
- Population adapter uses protocols to dispatch to correct methods

## Usage

Population adapter checks protocol conformance:
```python
if isinstance(self._nets, ParameterizableNetwork):
    # Can use ES/CMA-ES (parameter averaging)
    params = self._nets.get_parameters_flat()
elif isinstance(self._nets, StructuralNetwork):
    # Can only use GA (topology evolution)
    self._nets.select_and_duplicate(fitness)
```
