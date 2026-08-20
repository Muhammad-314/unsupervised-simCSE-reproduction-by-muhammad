from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load a YAML experiment configuration.

    Args:
        config_path:
            Path to a YAML configuration file.

    Returns:
        Configuration as a Python dictionary.
    """

    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration must contain a YAML mapping: {path}"
        )

    return config


def print_config(config: dict[str, Any]) -> None:
    """
    Print configuration in a readable format.
    """

    print("=" * 60)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 60)

    for key, value in config.items():
        print(f"{key}: {value}")