# Supervised Learning Orchestration

High-level training interface maintaining clean separation: optimizer never sees data, only fitness values.

Contains create_fitness_evaluator() creating fitness closure capturing observations/actions internally (returns callable taking no args, returning fitness tensor), and train_supervised() orchestrating workflow (creates train/test fitness evaluators, selects optimizer GA/ES/CMA-ES, runs optimization loop, returns results). Key design: closure hides data from optimizer for clean separation. Easy to add new evaluation modes. Used by ne/optim/*.py for supervised learning tasks.
