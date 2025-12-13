"""HuggingFace control tasks dataset loaders.

Loads pre-trained agent trajectories from HuggingFace for CartPole and LunarLander.

Functions:
    load_cartpole_data: Load CartPole-v1 from HuggingFace
    load_lunarlander_data: Load LunarLander-v2 from HuggingFace
"""

from data.hf_control_tasks.loaders import load_cartpole_data, load_lunarlander_data

__all__ = [
    "load_cartpole_data",
    "load_lunarlander_data",
]
