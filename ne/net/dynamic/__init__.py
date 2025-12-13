"""Dynamic topology neural networks for neuroevolution.

This package implements networks with evolving graph structures where nodes and
connections are added/removed through mutation. Networks can have cyclic
connections (graph-based recurrence) and use frozen random weights with online
standardization.
"""

from .evolution import Net, Node, NodeList
from .main import DynamicNetPopulation

__all__ = [
    "Net",
    "Node",
    "NodeList",
    "DynamicNetPopulation",
]
