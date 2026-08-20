import copy
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


def make_model():
    return torch.nn.Linear(4, 2)


def make_optimizer(model):
    return torch.optim.AdamW(
        model.parameters(),
        lr=3e-5,
    )


def make_scheduler(optimizer):
    return get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=4,
    )


def train_one_step(
    model,
    optimizer,
    scheduler,
    x,
    y,
):
    optimizer.zero_grad()

    prediction = model(x)

    loss = torch.nn.functional.mse_loss(
        prediction,
        y,
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()
    scheduler.step()

    return loss.item()


def main():

    print("=" * 70)
    print("SIMCSE TRAINING RESUME TEST")
    print("=" * 70)

    torch.manual_seed(42)

    x = torch.randn(
        8,
        4,
    )

    y = torch.randn(
        8,
        2,
    )

    config = {
        "learning_rate": 3e-5,
        "total_steps": 4,
    }

    # ========================================================
    # RUN A: CONTINUOUS TRAINING
    # ========================================================

    print(
        "\nRUN A: CONTINUOUS TRAINING"
    )

    torch.manual_seed(123)

    model_a = make_model()
    optimizer_a = make_optimizer(model_a)
    scheduler_a = make_scheduler(
        optimizer_a
    )

    for step in range(4):

        loss = train_one_step(
            model_a,
            optimizer_a,
            scheduler_a,
            x,
            y,
        )

        print(
            f"Step {step + 1}/4 "
            f"| Loss {loss:.8f} "
            f"| LR "
            f"{scheduler_a.get_last_lr()[0]:.10f}"
        )

    final_model_a = copy.deepcopy(
        model_a.state_dict()
    )

    final_optimizer_a = copy.deepcopy(
        optimizer_a.state_dict()
    )

    final_scheduler_a = copy.deepcopy(
        scheduler_a.state_dict()
    )

    # ========================================================
    # RUN B: INTERRUPTED + RESUMED
    # ========================================================

    print(
        "\nRUN B: INTERRUPTED + RESUMED"
    )

    torch.manual_seed(123)

    model_b = make_model()
    optimizer_b = make_optimizer(model_b)
    scheduler_b = make_scheduler(
        optimizer_b
    )

    # --------------------------------------------------------
    # First half.
    # --------------------------------------------------------

    for step in range(2):

        loss = train_one_step(
            model_b,
            optimizer_b,
            scheduler_b,
            x,
            y,
        )

        print(
            f"Before checkpoint "
            f"step {step + 1}/4 "
            f"| Loss {loss:.8f} "
            f"| LR "
            f"{scheduler_b.get_last_lr()[0]:.10f}"
        )

    checkpoint_path = (
        Path(
            tempfile.mkdtemp()
        )
        / "resume_test.pt"
    )

    save_checkpoint(
        path=checkpoint_path,
        model=model_b,
        optimizer=optimizer_b,
        scheduler=scheduler_b,
        epoch=1,
        global_step=2,
        config=config,
    )

    print(
        "\nCheckpoint saved after step 2."
    )

    # --------------------------------------------------------
    # Simulate a completely new process.
    # --------------------------------------------------------

    model_c = make_model()
    optimizer_c = make_optimizer(model_c)
    scheduler_c = make_scheduler(
        optimizer_c
    )

    metadata = load_checkpoint(
        path=checkpoint_path,
        model=model_c,
        optimizer=optimizer_c,
        scheduler=scheduler_c,
        device="cpu",
    )

    print(
        "Checkpoint loaded."
    )

    print(
        "Restored global step:",
        metadata["global_step"],
    )

    print(
        "Restored scheduler step:",
        scheduler_c.last_epoch,
    )

    print(
        "Restored LR:",
        scheduler_c.get_last_lr()[0],
    )

    # --------------------------------------------------------
    # Continue remaining two steps.
    # --------------------------------------------------------

    for step in range(2, 4):

        loss = train_one_step(
            model_c,
            optimizer_c,
            scheduler_c,
            x,
            y,
        )

        print(
            f"After resume "
            f"step {step + 1}/4 "
            f"| Loss {loss:.8f} "
            f"| LR "
            f"{scheduler_c.get_last_lr()[0]:.10f}"
        )

    # ========================================================
    # COMPARE FINAL STATES
    # ========================================================

    print(
        "\nComparing final states..."
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    for key in final_model_a:

        assert torch.equal(
            final_model_a[key],
            model_c.state_dict()[key],
        ), f"Model mismatch: {key}"

    print(
        "Model final state: PASSED"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer_a_state = (
        final_optimizer_a["state"]
    )

    optimizer_c_state = (
        optimizer_c.state_dict()["state"]
    )

    assert (
        optimizer_a_state.keys()
        == optimizer_c_state.keys()
    )

    for parameter_id in optimizer_a_state:

        state_a = optimizer_a_state[
            parameter_id
        ]

        state_c = optimizer_c_state[
            parameter_id
        ]

        assert (
            state_a.keys()
            == state_c.keys()
        )

        for state_name in state_a:

            value_a = state_a[
                state_name
            ]

            value_c = state_c[
                state_name
            ]

            if torch.is_tensor(value_a):

                assert torch.equal(
                    value_a,
                    value_c,
                ), (
                    "Optimizer tensor state "
                    f"mismatch: {parameter_id} "
                    f"/ {state_name}"
                )

            else:

                assert (
                    value_a == value_c
                ), (
                    "Optimizer scalar state "
                    f"mismatch: {parameter_id} "
                    f"/ {state_name}"
                )

    print(
        "Optimizer final state: PASSED"
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler_a_state = (
        final_scheduler_a
    )

    scheduler_c_state = (
        scheduler_c.state_dict()
    )

    assert (
        scheduler_a_state.keys()
        == scheduler_c_state.keys()
    )

    for key in scheduler_a_state:

        value_a = scheduler_a_state[
            key
        ]

        value_c = scheduler_c_state[
            key
        ]

        if torch.is_tensor(value_a):

            assert torch.equal(
                value_a,
                value_c,
            ), (
                "Scheduler tensor state "
                f"mismatch: {key}"
            )

        else:

            assert (
                value_a == value_c
            ), (
                "Scheduler state "
                f"mismatch: {key}"
            )

    print(
        "Scheduler final state: PASSED"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TRAINING RESUME TEST: PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()