# TorchRL Environment Evaluation

TorchRL-based environment evaluation with continual learning transfer modes.

Contains evaluate_env_episodes() accepting population for environment rollouts. Three transfer modes from config/state.py: env_transfer (save/restore env state across generations), mem_transfer (keep agent memory between episodes, don't reset hidden when done=True), fit_transfer (accumulate fitness across all generations via population.continual_fitness). Updates population tracking attributes: curr_eval_score, curr_eval_num_steps, total_num_steps, curr_episode_score/num_steps (if env_transfer), continual_fitness (if fit_transfer). Reset behavior: env_transfer=False resets env each generation, mem_transfer=False resets hidden on done=True, fit_transfer=True returns continual_fitness instead of curr_eval_score. Used by ne/eval/environment.py's create_env_fitness_evaluator(). Based on old gen_transfer.py architecture.
