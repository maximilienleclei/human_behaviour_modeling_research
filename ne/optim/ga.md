# Genetic Algorithm Optimizer

Genetic Algorithm with hard truncation (top 50% survive).

optimize_ga(), select_simple_ga() for hard selection. Algorithm: 1) Evaluate fitness, 2) Select top 50%, duplicate, 3) Mutate, repeat. Checkpoint every 5min, test evaluation, logging. Generic via eval_fn callback.
