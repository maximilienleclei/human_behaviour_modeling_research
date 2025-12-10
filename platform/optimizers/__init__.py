"""Optimization algorithms for training models.

This module contains both gradient-based (SGD) and evolutionary (GA) optimizers
for training neural networks on behavior modeling tasks.

Functions:
    optimize_sgd: Stochastic gradient descent with backpropagation
    optimize_ga: Genetic algorithm with mutation-based evolution
"""

__all__ = [
    "optimize_sgd",
    "optimize_ga",
]
