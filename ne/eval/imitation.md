# Imitation Learning and Adversarial Training

GAN-style training with generator-discriminator populations for behavior imitation.

Contains hide_elements() for hiding privileged info from discriminator (via hide_fn callable or hide_indices list), evaluate_imitation_episode() implementing two-phase adversarial training (Phase 1: generator acts → discriminator scores; Phase 2: target acts → discriminator scores; Generator fitness = p_target (phase 1), Discriminator fitness = -p_target (phase 1) + p_target (phase 2)), create_imitation_fitness_evaluators() creating fitness closures for both populations, and train_imitation() orchestrating adversarial optimization. Supports state persistence for recurrent networks, continual fitness accumulation (fit_transfer), and merge mode (scales generator fitness as fitness * 2 - 1). Inspired by old_imitate.py. Used for learning target agent behavior via adversarial training.
