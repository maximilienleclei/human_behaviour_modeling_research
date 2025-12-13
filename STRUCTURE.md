# Repository Structure

Complete tree of all classes, methods, and functions (like `tree` command).

```
human_behaviour_modeling_research/
├── config/
│   ├── device.py
│   │   └── ⚙️  set_device(gpu_index) → Set the global DEVICE variable
│   ├── experiments.py
│   │   ├── 🏛️  DataConfig → Data configuration
│   │   ├── 🏛️  ExperimentConfig → Complete experiment configuration
│   │   ├── 🏛️  ModelConfig → Model architecture configuration
│   │   ├── 🏛️  OptimizerConfig → Optimizer configuration
│   │   └── 🏛️  TrainingConfig → Training configuration
│   ├── paths.py
│   └── state.py
│       └── 🏛️  StatePersistenceConfig → Configuration for state persistence and transfer modes in continual learning
├── data/hf_control_tasks/
│   └── loaders.py
│       ├── ⚙️  load_cartpole_data() → Load CartPole-v1 dataset from HuggingFace
│       └── ⚙️  load_lunarlander_data() → Load LunarLander-v2 dataset from HuggingFace
├── data/simexp_control_tasks/source/
│   ├── collect.py
│   │   ├── 🏛️  PausedSeededWrapper
│   │   │   └── reset()
│   │   └── ⚙️  save_data_callback(obs_t, obs_tp1, action, rew, terminated, truncated, info)
│   ├── data_analysis.py
│   │   └── ⚙️  parse_analyze_and_plot(filenames)
│   └── replay.py
├── data/simexp_control_tasks/
│   ├── environments.py
│   │   └── ⚙️  get_data_file(env_name, subject) → Get data filename for environment and subject
│   ├── loaders.py
│   │   └── ⚙️  load_human_data(env_name, use_cl_info, subject, holdout_pct) → Load human behavior data from JSON files with random run holdout
│   └── preprocessing.py
│       ├── 🏛️  EpisodeDataset → Dataset that returns complete episodes instead of individual steps
│       ├── ⚙️  compute_session_run_ids(timestamps) → Compute session and run IDs from episode timestamps
│       ├── ⚙️  episode_collate_fn(batch) → Collate episodes with padding to handle variable lengths
│       └── ⚙️  normalize_session_run_features(session_ids, run_ids) → Normalize session and run IDs to [-1, 1] range
├── dl/models/
│   ├── dynamic.py
│   │   └── 🏛️  DynamicNetPopulation → Wrapper for common/dynamic_net population with BatchedPopulation interface
│   │       ├── forward_batch(observations) → Batched forward pass for all networks
│   │       ├── evaluate(observations, actions) → Evaluate fitness (cross-entropy) of all networks
│   │       ├── mutate() → Apply mutations to all networks in population
│   │       ├── select_simple_ga(fitness) → Simple GA selection: top 50% survive and duplicate
│   │       ├── get_state_dict() → Get state dict for checkpointing
│   │       └── load_state_dict(state) → Load state dict from checkpoint
│   ├── feedforward.py
│   │   └── 🏛️  MLP → Two-layer MLP with tanh activations
│   │       ├── forward(x) → Forward pass returning logits
│   │       └── get_probs(x) → Get probability distribution over actions
│   └── recurrent.py
│       ├── 🏛️  RecurrentMLPReservoir → Recurrent MLP with frozen reservoir (echo state network style)
│       │   ├── forward_step(x, h) → Single timestep forward pass
│       │   ├── forward(x, h_0) → Sequence forward pass
│       │   └── get_probs(x, h) → Get probability distribution over actions for a single step
│       └── 🏛️  RecurrentMLPTrainable → Recurrent MLP with trainable recurrent weights (rank-1 factorization)
│           ├── forward_step(x, h) → Single timestep forward pass
│           ├── forward(x, h_0) → Sequence forward pass
│           └── get_probs(x, h) → Get probability distribution over actions for a single step
├── dl/optim/
│   ├── base.py
│   │   ├── ⚙️  create_episode_list(observations, actions, episode_boundaries) → Convert flat data with episode boundaries into list of episode dicts
│   │   ├── ⚙️  load_checkpoint(checkpoint_path) → Load checkpoint from disk
│   │   └── ⚙️  save_checkpoint(checkpoint_path, checkpoint_data) → Save checkpoint to disk
│   └── sgd.py
│       └── ⚙️  optimize_sgd(model, optim_obs, optim_act, test_obs, test_act, output_size, metadata, checkpoint_path, max_optim_time, batch_size, learning_rate, loss_eval_interval_seconds, logger) → Optimize model using SGD with backpropagation
├── metrics/
│   ├── comparison.py
│   │   └── ⚙️  evaluate_progression_recurrent(model, episode_details, env, max_steps, use_cl_features) → Quick evaluation to track progression during optimization (for recurrent models)
│   └── metrics.py
│       ├── ⚙️  compute_cross_entropy(model, observations, actions) → Compute cross-entropy loss
│       └── ⚙️  compute_macro_f1(model, observations, actions, num_samples, num_classes) → Compute macro F1 score with multiple sampling trials
├── ne/net/dynamic/
│   ├── compute_test.py
│   │   ├── 🏛️  WelfordRunningStandardizer
│   │   └── ⚙️  barebone_run(verbose) → Simple working example to demonstrate how to run computation for the
│   ├── evolution.py
│   │   ├── 🏛️  Net → Network that expands/contracts through architectural mutations
│   │   │   ├── initialize_architecture()
│   │   │   ├── grow_node(in_node_1, role) → Method first called during initialization to grow the irremovable
│   │   │   ├── grow_connection(in_node, out_node)
│   │   │   ├── prune_node(node_being_pruned) → Removes an existing hidden node
│   │   │   ├── prune_connection(in_node, out_node, node_being_pruned) → Called by `prune_node` to remove the `node_being_pruned`'s
│   │   │   ├── mutate()
│   │   │   ├── clone() → Create a deep copy of this network
│   │   │   ├── get_state_dict() → Serialize complete network state for checkpointing
│   │   │   └── load_state_dict(state) → Restore network from serialized state
│   │   ├── 🏛️  Node
│   │   │   ├── sample_nearby_node(nodes_considered, local_connectivity_probability)
│   │   │   ├── connect_to(node)
│   │   │   └── disconnect_from(node)
│   │   └── 🏛️  NodeList → Holds `Node` instances for ease of manipulation
│   └── main.py
│       └── 🏛️  DynamicNetPopulation → Wrapper for dynamic network population with batched population interface
│           ├── forward_batch(observations) → Batched forward pass for all networks
│           ├── evaluate(observations, actions) → Evaluate fitness (cross-entropy) of all networks
│           ├── mutate() → Apply mutations to all networks in population
│           ├── select_and_duplicate(fitness) → Select top performers and duplicate to fill population
│           ├── get_state_dict() → Get state dict for checkpointing
│           └── load_state_dict(state) → Load state dict from checkpoint
├── ne/net/
│   ├── feedforward.py
│   │   └── 🏛️  BatchedFeedforward → Batched population of feedforward MLPs for efficient GPU-parallel computation
│   │       ├── forward_batch(x) → Batched forward pass for all networks in parallel
│   │       ├── mutate() → Apply mutations to all networks in parallel using adaptive or fixed sigma
│   │       ├── get_state_dict() → Get network state for checkpointing
│   │       ├── load_state_dict(state) → Restore network state from checkpoint
│   │       ├── get_parameters_flat() → Get flattened parameter vectors for all networks
│   │       ├── set_parameters_flat(flat_params) → Set network parameters from flat vectors
│   │       └── clone_network(indices) → Clone networks at specified indices to fill population
│   ├── protocol.py
│   │   ├── 🏛️  NetworkProtocol → Base protocol that all network types must implement
│   │   │   ├── forward_batch(x) → Batched forward pass for all networks in parallel
│   │   │   ├── mutate() → Apply mutations to all networks in the population
│   │   │   ├── get_state_dict() → Get state dictionary for checkpointing
│   │   │   └── load_state_dict(state) → Restore network from checkpoint state
│   │   ├── 🏛️  ParameterizableNetwork → Networks with flat parameter operations (for ES/CMA-ES optimizers)
│   │   │   ├── get_parameters_flat() → Get flattened parameter vectors for all networks
│   │   │   ├── set_parameters_flat(params) → Set network parameters from flat vectors
│   │   │   └── clone_network(indices) → Clone networks at specified indices to fill population
│   │   └── 🏛️  StructuralNetwork → Networks with evolving topology (for GA-only optimization)
│   │       └── select_and_duplicate(fitness) → Select top performers and duplicate to fill population
│   └── recurrent.py
│       └── 🏛️  BatchedRecurrent → Batched population of stacked recurrent MLPs for efficient GPU-parallel computation
│           ├── forward_batch_step(x, h_states) → Single timestep forward pass for all networks in parallel
│           ├── forward_batch_sequence(x, h_0) → Batched forward pass for sequence across all networks
│           ├── mutate() → Apply mutations to all networks in parallel using adaptive or fixed sigma
│           ├── save_hidden_states() → Save current hidden states for persistence across generations/episodes
│           ├── restore_hidden_states(states) → Restore hidden states from previous evaluation
│           ├── reset_hidden_states() → Reset all hidden states to zero
│           ├── get_state_dict() → Get full network state for checkpointing
│           ├── load_state_dict(state) → Restore network state from checkpoint
│           ├── get_parameters_flat() → Get flattened parameter vectors for all networks
│           ├── set_parameters_flat(flat_params) → Set network parameters from flat vectors
│           └── clone_network(indices) → Clone networks at specified indices to fill population
├── ne/eval/
│   ├── env.py
│   │   ├── ⚙️  evaluate_env_batch(nets, env, observations, actions) → Evaluate networks on batch of states from pre-recorded episodes
│   │   └── ⚙️  evaluate_env_episodes(population, env, num_episodes, max_steps_per_episode, metric, state_config, curr_gen) → Evaluate population on environment episodes with continual learning support
│   ├── environment.py
│   │   ├── ⚙️  create_env_fitness_evaluator(population, env, num_episodes, max_steps_per_episode, metric, state_config) → Create fitness evaluator for environment rollouts
│   │   ├── ⚙️  fitness_fn() → Evaluate current population on environment episodes
│   │   └── ⚙️  train_environment(population, train_env, test_env, num_episodes, max_steps_per_episode, metric, optimizer, max_time, eval_interval, checkpoint_path, logger, state_config) → High-level environment-based training
│   ├── evaluate.py
│   │   ├── ⚙️  evaluate_adversarial(nets, generator_data, discriminator_data, action_size) → Evaluate adversarial networks with split outputs
│   │   ├── ⚙️  evaluate_episodes(nets, episodes, eval_fn) → Evaluate networks on multiple episodes
│   │   ├── ⚙️  evaluate_feedforward(nets, observations, actions) → Evaluate feedforward networks on data using cross-entropy loss
│   │   └── ⚙️  evaluate_recurrent(nets, observations, actions) → Evaluate recurrent networks on sequence using cross-entropy loss
│   ├── imitation.py
│   │   ├── ⚙️  create_imitation_fitness_evaluators(generator_pop, discriminator_pop, target_agent, env, max_steps, hide_fn, indices, state_config, merge_mode) → Create fitness evaluators for imitation learning
│   │   ├── ⚙️  disc_fitness_fn() → Evaluate discriminator fitness
│   │   ├── ⚙️  evaluate_imitation_episode(generator_pop, discriminator_pop, target_agent, env, max_steps, hide_fn, indices, state_config, curr_gen, merge_mode) → Evaluate generator and discriminator populations on imitation task
│   │   ├── ⚙️  gen_fitness_fn() → Evaluate generator fitness
│   │   ├── ⚙️  hide_elements(obs, hide_fn, indices) → Hide specific elements from observation
│   │   └── ⚙️  train_imitation(generator_pop, discriminator_pop, target_agent, train_env, test_env, max_steps, hide_fn, indices, optimizer, max_time, eval_interval, checkpoint_path_gen, checkpoint_path_disc, logger, state_config, merge_mode) → High-level imitation learning training
│   └── supervised.py
│       ├── ⚙️  create_fitness_evaluator(population, observations, actions) → Create fitness evaluator for supervised learning
│       ├── ⚙️  fitness_fn() → Evaluate current population on data
│       └── ⚙️  train_supervised(population, train_data, test_data, optimizer, max_time, eval_interval, checkpoint_path, logger) → High-level supervised learning training
├── ne/pop/
│   └── population.py
│       └── 🏛️  Population → Bridge between networks and eval/optim layers
│           ├── nets() → Access underlying network object
│           ├── num_nets() → Number of networks in population
│           ├── reset_episode_tracking() → Reset per-episode tracking attributes
│           ├── reset_eval_tracking() → Reset per-evaluation tracking attributes
│           ├── reset_all_tracking() → Reset all tracking attributes (including global counters)
│           ├── get_actions(logits) → Convert network outputs to actions
│           ├── select_networks(indices) → Select networks by indices and duplicate to fill population
│           ├── get_parameters_flat() → Get flattened parameters for all networks
│           ├── set_parameters_flat(flat_params) → Set parameters from flattened tensor
│           ├── mutate() → Apply mutations to all networks
│           ├── get_state_dict() → Get state for checkpointing
│           └── load_state_dict(state) → Restore from checkpoint
├── ne/optim/
│   ├── base.py
│   │   ├── ⚙️  optimize(population, fitness_fn, test_fitness_fn, selection_fn, algorithm_name, max_time, eval_interval, checkpoint_path, logger, state_config) → Shared optimization loop for all evolutionary algorithms
│   │   └── ⚙️  save_checkpoint(path, gen, fit_hist, test_hist, time, population, algorithm, hidden_states) → Save optimization checkpoint
│   ├── cmaes.py
│   │   ├── 🏛️  CMAESState → CMA-ES algorithm state (mean, covariance, evolution paths)
│   │   ├── ⚙️  optimize_cmaes(population, fitness_fn, test_fitness_fn, max_time, eval_interval, checkpoint_path, logger, state_config) → Optimize networks using CMA-ES
│   │   └── ⚙️  select_cmaes(population, fitness) → CMA-ES selection: adapt search distribution based on fitness
│   ├── es.py
│   │   ├── ⚙️  optimize_es(population, fitness_fn, test_fitness_fn, max_time, eval_interval, checkpoint_path, logger, state_config) → Optimize networks using Evolution Strategy
│   │   └── ⚙️  select_es(population, fitness) → ES selection: rank-weighted parameter averaging
│   └── ga.py
│       ├── ⚙️  optimize_ga(population, fitness_fn, test_fitness_fn, max_time, eval_interval, checkpoint_path, logger, state_config) → Optimize networks using Simple Genetic Algorithm
│       └── ⚙️  select_ga(population, fitness) → GA selection: top 50% survive and duplicate
└── root/
    └── generate_structure.py
        ├── ⚙️  get_docstring(node) → Extract first line of docstring
        └── ⚙️  parse_file(file_path) → Extract classes, methods, and functions with descriptions
```

---

**Legend:**
- 🏛️ Class
- ⚙️ Function
- Methods are listed under their parent class (no icon)

**Statistics:**
- 13 directories
- 35 files
- 24 classes
- 71 methods
- 46 functions