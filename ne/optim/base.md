# Optimizer Shared Training Loop

Shared optimization loop for all evolutionary algorithms (GA, ES, CMA-ES) eliminating code duplication.

Contains optimize() handling checkpoint save/load with resumption, time-based training (max_time limit), periodic test evaluation (eval_interval), fitness tracking/logging, and optional recurrent state persistence across generations. Signature: optimize(population, fitness_fn, test_fitness_fn, selection_fn, algorithm_name, ...). Key architectural change: optimizer receives fitness functions (closures) never seeing observations/actions directly for clean separation. Supports optional config/state.py StatePersistenceConfig for continual learning (persist_across_generations, reset_on_selection). Used by ga.py, es.py, cmaes.py as common training loop. Returns training histories.
