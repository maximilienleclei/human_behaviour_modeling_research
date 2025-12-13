"""State persistence configuration for continual learning.

Defines configuration for hidden state persistence and transfer modes
in evolutionary optimization of recurrent networks.
"""

from dataclasses import dataclass


@dataclass
class StatePersistenceConfig:
    """Configuration for state persistence and transfer modes in continual learning.

    Combines recurrent network hidden state persistence with environment/fitness transfer modes.

    Hidden State Persistence (for recurrent networks):
        persist_across_generations: Save/restore hidden states between optimization steps
        persist_across_episodes: Maintain hidden state across episode boundaries during eval
        reset_on_selection: Reset hidden states after selection (before mutation)

    Transfer Modes (inspired by old gen_transfer.py):
        env_transfer: Save/restore environment state across generations
            - When True: Each generation continues from where previous generation left off
            - Environment state and observation are saved/restored
            - Episodes can span multiple generations
        mem_transfer: Keep agent memory (hidden states) between episodes
            - When True: Don't reset agent hidden states when episode ends
            - Only applies within a single evaluation
        fit_transfer: Accumulate fitness across all generations (continual fitness)
            - When True: Optimize continual_fitness instead of per-generation fitness
            - Fitness accumulates across entire evolutionary run

    Reset Behavior:
        The three transfer modes combine to control when agent/environment reset:
        - env_transfer=False: Reset environment each generation
        - mem_transfer=False: Reset agent when episode ends (done=True)
        - fit_transfer=True: Use continual_fitness as optimization target

    Args:
        enabled: Master switch for state persistence (ignored if transfer modes are set)
        persist_across_generations: Save/restore hidden states between optimization steps
        persist_across_episodes: Maintain hidden state across episode boundaries
        reset_on_selection: Reset hidden states after selection
        env_transfer: Save/restore environment state across generations
        mem_transfer: Keep agent memory between episodes
        fit_transfer: Accumulate fitness across all generations
    """

    enabled: bool = False
    persist_across_generations: bool = False
    persist_across_episodes: bool = False
    reset_on_selection: bool = True

    # Transfer modes from old gen_transfer.py
    env_transfer: bool = False
    mem_transfer: bool = False
    fit_transfer: bool = False
