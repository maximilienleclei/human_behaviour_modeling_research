"""Unified experimentation platform for human behavior modeling research.

This package provides a reusable framework for training and evaluating neural networks
(both feedforward and recurrent) using gradient-based (SGD) and evolutionary (GA) methods.

Modules:
    models: Neural network architectures (feedforward, recurrent, dynamic)
    optimizers: Training algorithms (SGD, genetic algorithms)
    data: Dataset loading and preprocessing
    evaluation: Metrics and behavioral comparison
    config: Configuration schemas and constants
    runner: Main experiment execution engine
"""

from beartype import BeartypeConf
from beartype.claw import beartype_this_package

# Enable beartype runtime type checking for all modules in this package
beartype_this_package(conf=BeartypeConf(is_pep484_tower=True))

__version__ = "1.0.0"
