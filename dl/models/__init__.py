"""Neural network models for behavior modeling.

This module contains model architectures that work with both SGD and GA optimizers,
as well as GA-exclusive models like dynamic networks.

Classes:
    MLP: Feedforward multilayer perceptron (shared: SGD + GA)
    RecurrentMLPReservoir: Recurrent MLP with frozen reservoir (shared: SGD + GA)
    RecurrentMLPTrainable: Recurrent MLP with trainable weights (shared: SGD + GA)
    DynamicNetPopulation: Dynamic network population wrapper (GA-exclusive)
"""

from dl.models.dynamic import DynamicNetPopulation
from dl.models.feedforward import MLP
from dl.models.recurrent import RecurrentMLPReservoir, RecurrentMLPTrainable

__all__ = [
    "MLP",
    "RecurrentMLPReservoir",
    "RecurrentMLPTrainable",
    "DynamicNetPopulation",
    "create_model",
]


def create_model(model_type: str, **kwargs):
    """Factory function for creating models.

    Args:
        model_type: Model type name ('MLP', 'RecurrentMLPReservoir', 'RecurrentMLPTrainable', 'DynamicNetPopulation')
        **kwargs: Model-specific arguments

    Returns:
        Model instance

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type == "MLP":
        from .feedforward import MLP
        return MLP(**kwargs)
    elif model_type == "RecurrentMLPReservoir":
        from .recurrent import RecurrentMLPReservoir
        return RecurrentMLPReservoir(**kwargs)
    elif model_type == "RecurrentMLPTrainable":
        from .recurrent import RecurrentMLPTrainable
        return RecurrentMLPTrainable(**kwargs)
    elif model_type == "DynamicNetPopulation":
        from .dynamic import DynamicNetPopulation
        return DynamicNetPopulation(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
