# Evolution Strategy Optimizer

Evolution Strategy with soft rank-based selection.

optimize_es(), select_simple_es() for rank-based soft selection (all networks contribute weighted by rank). Algorithm: 1) Evaluate fitness, 2) Soft selection, 3) Mutate, repeat. Difference from GA: soft vs hard truncation. Generic via eval_fn callback.
