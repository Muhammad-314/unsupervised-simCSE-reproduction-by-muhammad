import tempfile
from pathlib import Path

import torch
from transformers import (
    get_linear_schedule_with_warmup,
)

from checkpoint import (
    load_checkpoint,
    save_checkpoint,
)


def main():

    print("=" * 70)
    print("SIMCSE CHECKPOINT TEST")
    print("=" * 70)

    model = torch.nn.Linear(
        4,
        4,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=10,
    )

    config = {
        "model_name": "bert-base-uncased",
        "learning_rate": 3e-5,
        "temperature": 0.05,
    }

    # --------------------------------------------------------
    # Create optimizer + scheduler state.
    # --------------------------------------------------------

    print(
        "\nCreating optimizer/scheduler state..."
    )

    x = torch.randn(
        2,
        4,
    )

    loss = model(x).pow(2).mean()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    scheduler.step()

    original_lr = (
        scheduler.get_last_lr()[0]
    )

    original_scheduler_step = (
        scheduler.last_epoch
    )

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    with tempfile.TemporaryDirectory() as temp_dir:

        checkpoint_path = (
            Path(temp_dir)
            / "training_state.pt"
        )

        print(
            "\nSaving checkpoint..."
        )

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=3,
            global_step=42,
            config=config,
        )

        assert checkpoint_path.exists()

        print(
            "Checkpoint exists:",
            checkpoint_path.exists(),
        )

        # ----------------------------------------------------
        # Create fresh objects.
        # ----------------------------------------------------

        new_model = torch.nn.Linear(
            4,
            4,
        )

        new_optimizer = torch.optim.AdamW(
            new_model.parameters(),
            lr=3e-5,
        )

        new_scheduler = (
            get_linear_schedule_with_warmup(
                optimizer=new_optimizer,
                num_warmup_steps=0,
                num_training_steps=10,
            )
        )

        # ----------------------------------------------------
        # Load.
        # ----------------------------------------------------

        print(
            "\nLoading checkpoint..."
        )

        metadata = load_checkpoint(
            path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
            scheduler=new_scheduler,
            device="cpu",
        )

        print(
            "Loaded epoch:",
            metadata["epoch"],
        )

        print(
            "Loaded global step:",
            metadata["global_step"],
        )

        print(
            "Loaded config:",
            metadata["config"],
        )

        # ----------------------------------------------------
        # Model.
        # ----------------------------------------------------

        for original, restored in zip(
            model.parameters(),
            new_model.parameters(),
        ):
            assert torch.equal(
                original,
                restored,
            )

        print(
            "\nModel state: PASSED"
        )

        # ----------------------------------------------------
        # Optimizer.
        # ----------------------------------------------------

        assert (
            len(optimizer.state)
            == len(new_optimizer.state)
        )

        print(
            "Optimizer state: PASSED"
        )

        # ----------------------------------------------------
        # Scheduler.
        # ----------------------------------------------------

        restored_lr = (
            new_scheduler.get_last_lr()[0]
        )

        restored_scheduler_step = (
            new_scheduler.last_epoch
        )

        print(
            "Original scheduler LR:",
            original_lr,
        )

        print(
            "Restored scheduler LR:",
            restored_lr,
        )

        print(
            "Original scheduler step:",
            original_scheduler_step,
        )

        print(
            "Restored scheduler step:",
            restored_scheduler_step,
        )

        assert (
            original_lr
            == restored_lr
        )

        assert (
            original_scheduler_step
            == restored_scheduler_step
        )

        print(
            "Scheduler state: PASSED"
        )

        # ----------------------------------------------------
        # Metadata.
        # ----------------------------------------------------

        assert metadata["epoch"] == 3

        assert (
            metadata["global_step"]
            == 42
        )

        assert (
            metadata["config"]
            == config
        )

        print(
            "Metadata: PASSED"
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL CHECKPOINT TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()