"""Training methods and objectives.

Methods work with both SGD and GA optimizers:
- supervised: Standard behavioral cloning
- gail: Adversarial imitation learning (future)
- imitation: Generator/discriminator co-evolution (future, GA-focused)
- transfer: Transfer learning across generations (future, GA-exclusive)

Extensible for: IRL, AIRL, other RL/imitation methods
"""
