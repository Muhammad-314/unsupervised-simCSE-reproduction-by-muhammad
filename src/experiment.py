from datetime import datetime
from pathlib import Path
import shutil
import yaml


def create_experiment_directory(
    experiment_name: str,
    config_path: str,
    root: str = "experiments",
) -> Path:
    """
    Create a timestamped experiment directory.

    Example:

        experiments/
        └── unsupervised_bert_base/
            └── 20260816_223500/
    """

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    experiment_dir = (
        Path(root)
        / experiment_name
        / timestamp
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    # Preserve the exact configuration used.
    shutil.copy2(
        config_path,
        experiment_dir / "config.yaml",
    )

    return experiment_dir


def save_config(
    config: dict,
    output_path: Path,
) -> None:
    """
    Save a Python configuration dictionary as YAML.
    """

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )