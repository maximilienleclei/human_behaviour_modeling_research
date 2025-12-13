# Population Layer

Adapter layer between networks and optimizers in the neuroevolution architecture.

## Files

- **population.py** - Population wrapper class (adapter between nets and optimizers)
- **population.md** - Documentation for Population class

## Architecture

The population layer serves as an **adapter** that translates between network representations and what optimizers need:

```
ne/eval/          # Evaluation (creates fitness functions, never exposes data)
    ↓
ne/pop/           # Adapter layer (THIS)
    ↓
ne/optim/         # Optimization (GA uses indices, ES uses vectors)
    ↓
ne/net/           # Network implementations (forward pass, mutation)
```

### Population Class - The Adapter

**Single algorithm-agnostic wrapper** that adapts network representations for different optimizers:

**Adapter Methods (for optimizers):**
- `select_networks(indices)` - For GA which operates on network objects
- `get_parameters_flat()` / `set_parameters_flat()` - For ES which operates on parameter vectors

**Service Methods:**
- `get_actions(logits)` - Output → action conversion (softmax, argmax, raw)
- `mutate()` - Delegate mutation to networks
- `get_state_dict()` / `load_state_dict()` - Checkpointing

**Key Insight:**
- **GA** operates on network objects (pick indices) → Population.select_networks()
- **ES** operates on parameter vectors (average) → Population.get/set_parameters_flat()
- **Networks** stay simple (forward + mutation) → Population handles translation

**Does NOT handle:**
- Selection logic (that's in `ne/optim/`)
- Evaluation logic (that's in `ne/eval/`)
- Network architecture (that's in `ne/net/`)

## Design Philosophy

Population is an **adapter pattern** implementation - it translates between what networks provide (batched tensors) and what optimizers need (indices for GA, flat vectors for ES). This keeps optimizers simple and network-agnostic.
