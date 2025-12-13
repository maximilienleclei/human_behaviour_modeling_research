"""Environment configurations for control tasks.

Defines environment-specific settings like observation/action dimensions,
data files per subject, and environment names.
"""

# Environment configurations
ENV_CONFIGS: dict[str, dict] = {
    "cartpole": {
        "data_files": {
            "sub01": "sub01_data_cartpole.json",
            "sub02": "sub02_data_cartpole.json",
        },
        "obs_dim": 4,
        "action_dim": 2,
        "name": "CartPole",
    },
    "mountaincar": {
        "data_files": {
            "sub01": "sub01_data_mountaincar.json",
            "sub02": "sub02_data_mountaincar.json",
        },
        "obs_dim": 2,
        "action_dim": 3,
        "name": "MountainCar",
    },
    "acrobot": {
        "data_files": {
            "sub01": "sub01_data_acrobot.json",
            "sub02": "sub02_data_acrobot.json",
        },
        "obs_dim": 6,
        "action_dim": 3,
        "name": "Acrobot",
    },
    "lunarlander": {
        "data_files": {
            "sub01": "sub01_data_lunarlander.json",
            "sub02": "sub02_data_lunarlander.json",
        },
        "obs_dim": 8,
        "action_dim": 4,
        "name": "LunarLander",
    },
}


def get_data_file(env_name: str, subject: str) -> str:
    """Get data filename for environment and subject.

    Args:
        env_name: Environment name
        subject: Subject identifier (sub01, sub02)

    Returns:
        Data filename

    Raises:
        ValueError: If environment or subject not found
    """
    if env_name not in ENV_CONFIGS:
        raise ValueError(f"Unknown environment: {env_name}")

    env_config: dict = ENV_CONFIGS[env_name]

    if subject not in env_config["data_files"]:
        available_subjects: list[str] = list(env_config["data_files"].keys())
        raise ValueError(
            f"No data for subject '{subject}' in environment '{env_name}'. "
            f"Available: {available_subjects}"
        )

    return env_config["data_files"][subject]
