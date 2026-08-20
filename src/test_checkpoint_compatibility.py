from pathlib import Path

import torch

from model import SimCSEModel


CHECKPOINT = Path(
    "experiments/tiny_runner_test/"
    "20260816_231924/"
    "checkpoint/epoch_1/"
    "training_state.pt"
)

MODEL_NAME = "bert-base-uncased"


def main():

    print("=" * 70)
    print("SIMCSE CHECKPOINT COMPATIBILITY TEST")
    print("=" * 70)

    print(
        "\nCheckpoint:"
    )
    print(CHECKPOINT)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Load checkpoint metadata
    # --------------------------------------------------------

    print(
        "\nLoading checkpoint..."
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint global step:",
        checkpoint["global_step"],
    )

    print(
        "Checkpoint config:"
    )

    print(
        checkpoint["config"]
    )

    checkpoint_state = checkpoint[
        "model_state_dict"
    ]

    # --------------------------------------------------------
    # Current model
    # --------------------------------------------------------

    print(
        "\nLoading current SimCSE model..."
    )

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    current_state = model.state_dict()

    # --------------------------------------------------------
    # Compare key sets
    # --------------------------------------------------------

    checkpoint_keys = set(
        checkpoint_state.keys()
    )

    current_keys = set(
        current_state.keys()
    )

    missing_from_checkpoint = (
        current_keys
        - checkpoint_keys
    )

    unexpected_in_checkpoint = (
        checkpoint_keys
        - current_keys
    )

    print(
        "\nCurrent model parameter count:",
        len(current_keys),
    )

    print(
        "Checkpoint parameter count:",
        len(checkpoint_keys),
    )

    print(
        "\nMissing from checkpoint:",
        len(missing_from_checkpoint),
    )

    for key in sorted(
        missing_from_checkpoint
    ):
        print(
            "  MISSING:",
            key,
        )

    print(
        "\nUnexpected checkpoint keys:",
        len(unexpected_in_checkpoint),
    )

    for key in sorted(
        unexpected_in_checkpoint
    ):
        print(
            "  UNEXPECTED:",
            key,
        )

    # --------------------------------------------------------
    # Shape comparison
    # --------------------------------------------------------

    shape_mismatches = []

    for key in sorted(
        checkpoint_keys
        & current_keys
    ):

        checkpoint_shape = tuple(
            checkpoint_state[key].shape
        )

        current_shape = tuple(
            current_state[key].shape
        )

        if (
            checkpoint_shape
            != current_shape
        ):
            shape_mismatches.append(
                (
                    key,
                    checkpoint_shape,
                    current_shape,
                )
            )

    print(
        "\nShape mismatches:",
        len(shape_mismatches),
    )

    for (
        key,
        checkpoint_shape,
        current_shape,
    ) in shape_mismatches:

        print(
            "  MISMATCH:",
            key,
            "checkpoint=",
            checkpoint_shape,
            "current=",
            current_shape,
        )

    # --------------------------------------------------------
    # Strict compatibility decision
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    if (
        not missing_from_checkpoint
        and not unexpected_in_checkpoint
        and not shape_mismatches
    ):

        print(
            "CHECKPOINT IS FULLY COMPATIBLE."
        )

        # Actually load it strictly.
        model.load_state_dict(
            checkpoint_state,
            strict=True,
        )

        print(
            "Strict model loading: PASSED"
        )

    else:

        print(
            "CHECKPOINT IS NOT FULLY COMPATIBLE."
        )

        print(
            "\nThis is expected if the checkpoint was "
            "created before the current model architecture "
            "was finalized."
        )

        print(
            "We will NOT use this checkpoint for the "
            "final reproduction."
        )

    print(
        "-" * 70
    )

    print(
        "\nCheckpoint compatibility test complete."
    )


if __name__ == "__main__":
    main()