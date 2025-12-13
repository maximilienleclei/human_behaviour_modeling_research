"""Optimization algorithms for training models.

This module contains gradient-based (SGD) and evolutionary (GA, ES, CMA-ES) optimizers
for training neural networks on behavior modeling tasks.

Functions:
    optimize_sgd: Stochastic gradient descent with backpropagation
    optimize_ga: Genetic algorithm (works with all model types via dispatch)
    optimize_ga_feedforward: Genetic algorithm for feedforward networks (hard selection)
    optimize_es_feedforward: Evolution strategies for feedforward networks (soft selection)
    optimize_es_recurrent: Evolution strategies for recurrent networks (soft selection)
    optimize_cmaes_feedforward: CMA-ES for feedforward networks (diagonal covariance adaptation)
    optimize_cmaes_recurrent: CMA-ES for recurrent networks (diagonal covariance adaptation)
"""

from platform.optimizers.sgd import optimize_sgd
from platform.optimizers.genetic import optimize_ga, optimize_es_recurrent, optimize_cmaes_recurrent
from platform.optimizers.evolution import optimize_ga_feedforward, optimize_es_feedforward, optimize_cmaes_feedforward

__all__ = [
    "optimize_sgd",
    "optimize_ga",
    "optimize_ga_feedforward",
    "optimize_es_feedforward",
    "optimize_es_recurrent",
    "optimize_cmaes_feedforward",
    "optimize_cmaes_recurrent",
]
