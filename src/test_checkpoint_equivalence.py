import sys
from pathlib import Path

import numpy as np
import torch


CONTINUOUS_CHECKPOINT = Path(
    "experiments/rng_continuous/20260817_013758/"
    "checkpoint/epoch_2/training_state.pt"
)

RESUMED_CHECKPOINT = Path(
    "experiments/rng_resumed/20260817_014600/"
    "checkpoint/epoch_2/training_state.pt"
)


def assert_tensor_equal(
    name: str,
    a: torch.Tensor,
    b: torch.Tensor,
) -> None:
    if not torch.equal(a, b):
        difference = (
            a.detach().cpu().float()
            - b.detach().cpu().float()
        ).abs().max().item()

        raise AssertionError(
            f"{name} differs. "
            f"Maximum absolute difference: {difference}"
        )


def compare_nested(
    name: str,
    a,
    b,
) -> int:
    """
    Recursively compare checkpoint state objects.

    Returns:
        Number of differences found.
    """

    differences = 0

    if isinstance(a, torch.Tensor):
        if not isinstance(b, torch.Tensor):
            print(
                f"DIFFERENCE: {name}: "
                f"tensor vs {type(b)}"
            )
            return 1

        if not torch.equal(a, b):
            max_difference = (
                a.detach().cpu().float()
                - b.detach().cpu().float()
            ).abs().max().item()

            print(
                f"DIFFERENCE: {name} "
                f"(max abs diff: {max_difference})"
            )

            differences += 1

        return differences

    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray):
            print(
                f"DIFFERENCE: {name}: "
                f"numpy array vs {type(b)}"
            )
            return 1

        if not np.array_equal(a, b):
            print(
                f"DIFFERENCE: {name} "
                "(numpy arrays differ)"
            )
            differences += 1

        return differences

    if isinstance(a, dict):
        if not isinstance(b, dict):
            print(
                f"DIFFERENCE: {name}: "
                f"dict vs {type(b)}"
            )
            return 1

        keys_a = set(a.keys())
        keys_b = set(b.keys())

        for key in sorted(
            keys_a - keys_b,
            key=str,
        ):
            print(
                f"DIFFERENCE: {name}.{key} "
                "(missing from resumed checkpoint)"
            )
            differences += 1

        for key in sorted(
            keys_b - keys_a,
            key=str,
        ):
            print(
                f"DIFFERENCE: {name}.{key} "
                "(unexpected in resumed checkpoint)"
            )
            differences += 1

        for key in sorted(
            keys_a & keys_b,
            key=str,
        ):
            differences += compare_nested(
                f"{name}.{key}",
                a[key],
                b[key],
            )

        return differences

    if isinstance(a, (list, tuple)):
        if not isinstance(b, type(a)):
            print(
                f"DIFFERENCE: {name}: "
                f"{type(a)} vs {type(b)}"
            )
            return 1

        if len(a) != len(b):
            print(
                f"DIFFERENCE: {name}: "
                f"length {len(a)} vs {len(b)}"
            )
            differences += 1
            return differences

        for index, (item_a, item_b) in enumerate(
            zip(a, b)
        ):
            differences += compare_nested(
                f"{name}[{index}]",
                item_a,
                item_b,
            )

        return differences

    if isinstance(a, float):
        if a != b:
            print(
                f"DIFFERENCE: {name}: "
                f"{a} vs {b}"
            )
            differences += 1

        return differences

    if a != b:
        print(
            f"DIFFERENCE: {name}: "
            f"{a!r} vs {b!r}"
        )
        differences += 1

    return differences


def main() -> None:
    print("=" * 70)
    print("SIMCSE FINAL CHECKPOINT EQUIVALENCE TEST")
    print("=" * 70)

    print()
    print("Continuous checkpoint:")
    print(CONTINUOUS_CHECKPOINT)

    print()
    print("Resumed checkpoint:")
    print(RESUMED_CHECKPOINT)

    if not CONTINUOUS_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Continuous checkpoint not found:\n"
            f"{CONTINUOUS_CHECKPOINT}"
        )

    if not RESUMED_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Resumed checkpoint not found:\n"
            f"{RESUMED_CHECKPOINT}"
        )

    print()
    print("Loading checkpoints...")

    continuous = torch.load(
        CONTINUOUS_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    resumed = torch.load(
        RESUMED_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print()
    print("Checking metadata...")

    assert (
        continuous["epoch"]
        == resumed["epoch"]
    )

    assert (
        continuous["global_step"]
        == resumed["global_step"]
    )

    print(
        "Epoch:",
        continuous["epoch"],
    )

    print(
        "Global step:",
        continuous["global_step"],
    )

    print()
    print("Comparing model state...")

    model_differences = compare_nested(
        "model_state_dict",
        continuous["model_state_dict"],
        resumed["model_state_dict"],
    )

    if model_differences == 0:
        print(
            "Model state: IDENTICAL"
        )
    else:
        print(
            f"Model state differences: "
            f"{model_differences}"
        )

    print()
    print("Comparing optimizer state...")

    optimizer_differences = compare_nested(
        "optimizer_state_dict",
        continuous["optimizer_state_dict"],
        resumed["optimizer_state_dict"],
    )

    if optimizer_differences == 0:
        print(
            "Optimizer state: IDENTICAL"
        )
    else:
        print(
            f"Optimizer state differences: "
            f"{optimizer_differences}"
        )

    print()
    print("Comparing scheduler state...")

    scheduler_differences = compare_nested(
        "scheduler_state_dict",
        continuous["scheduler_state_dict"],
        resumed["scheduler_state_dict"],
    )

    if scheduler_differences == 0:
        print(
            "Scheduler state: IDENTICAL"
        )
    else:
        print(
            f"Scheduler state differences: "
            f"{scheduler_differences}"
        )

    print()
    print("Comparing RNG state...")

    rng_differences = compare_nested(
        "rng_state",
        continuous["rng_state"],
        resumed["rng_state"],
    )

    if rng_differences == 0:
        print(
            "RNG state: IDENTICAL"
        )
    else:
        print(
            f"RNG state differences: "
            f"{rng_differences}"
        )

    print()
    print("Comparing configuration...")

    config_differences = compare_nested(
        "config",
        continuous["config"],
        resumed["config"],
    )

    if config_differences == 0:
        print(
            "Configuration: IDENTICAL"
        )
    else:
        print(
            f"Configuration differences: "
            f"{config_differences}"
        )

    total_differences = (
        model_differences
        + optimizer_differences
        + scheduler_differences
        + rng_differences
        + config_differences
    )

    print()
    print("-" * 70)

    if total_differences == 0:
        print(
            "ALL FINAL CHECKPOINT STATES ARE IDENTICAL."
        )
        print(
            "Deterministic resume test: PASSED"
        )
    else:
        print(
            "CHECKPOINTS ARE NOT IDENTICAL."
        )
        print(
            f"Total differences: "
            f"{total_differences}"
        )
        print(
            "Deterministic resume test: FAILED"
        )

        sys.exit(1)

    print("-" * 70)


if __name__ == "__main__":
    main()