"""Deep learning optimizers module.

This module contains gradient-based optimization for training neural networks
on behavior modeling tasks.

Functions:
    optimize_sgd: Stochastic gradient descent with backpropagation
"""

from dl.optim.sgd import optimize_sgd

__all__ = ["optimize_sgd"]
