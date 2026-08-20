from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
) -> None:
    """
    Save a complete SimCSE training checkpoint.

    The checkpoint contains:

        epoch
        global_step
        model_state_dict
        optimizer_state_dict
        scheduler_state_dict
        config
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": (
            model.state_dict()
        ),
        "optimizer_state_dict": (
            optimizer.state_dict()
            if optimizer is not None
            else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict()
            if scheduler is not None
            else None
        ),
        "config": config,
    }

    torch.save(
        state,
        path,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """
    Load a SimCSE training checkpoint.

    The model state is always restored.

    If supplied, optimizer and scheduler state are restored.

    Returns:

        {
            "epoch": ...,
            "global_step": ...,
            "config": ...,
        }
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Checkpoint must contain a dictionary."
        )

    required_keys = {
        "epoch",
        "global_step",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "config",
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise KeyError(
            "Checkpoint is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer_state = (
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    if (
        optimizer is not None
        and optimizer_state is not None
    ):
        optimizer.load_state_dict(
            optimizer_state
        )

    scheduler_state = (
        checkpoint[
            "scheduler_state_dict"
        ]
    )

    if (
        scheduler is not None
        and scheduler_state is not None
    ):
        scheduler.load_state_dict(
            scheduler_state
        )

    return {
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint[
            "global_step"
        ],
        "config": checkpoint["config"],
    }