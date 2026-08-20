import argparse
import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from config import load_config
from data import WikipediaSentenceDataset, SimCSECollator
from experiment import create_experiment_directory
from loss import SimCSELoss
from model_ablations import SimCSEModel


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ============================================================
# RNG state
# ============================================================

def get_rng_state(
    loader_generator: torch.Generator,
) -> dict:
    """
    Capture all RNG states needed for reproducible resume.
    """

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "loader_generator": (
            loader_generator.get_state()
        ),
    }

    if torch.cuda.is_available():
        state["cuda"] = (
            torch.cuda.get_rng_state_all()
        )

    return state


def restore_rng_state(
    state: dict,
    loader_generator: torch.Generator,
) -> None:
    """
    Restore all RNG states needed for reproducible resume.
    """

    random.setstate(
        state["python"]
    )

    np.random.set_state(
        state["numpy"]
    )

    torch.set_rng_state(
        state["torch"]
    )

    loader_generator.set_state(
        state["loader_generator"]
    )

    if (
        torch.cuda.is_available()
        and "cuda" in state
    ):
        torch.cuda.set_rng_state_all(
            state["cuda"]
        )

# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    """
    Select CUDA when available, otherwise CPU.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Metrics
# ============================================================

def calculate_similarity_metrics(
    similarity_matrix: torch.Tensor,
) -> tuple[float, float]:
    """
    Calculate mean positive and negative cosine similarity.

    Positive pairs are on the diagonal.

    Negative pairs are all off-diagonal entries.
    """

    batch_size = similarity_matrix.shape[0]

    positive_similarity = torch.diagonal(
        similarity_matrix
    ).mean()

    negative_mask = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=similarity_matrix.device,
    )

    negative_similarity = similarity_matrix[
        negative_mask
    ].mean()

    return (
        positive_similarity.item(),
        negative_similarity.item(),
    )


# ============================================================
# Environment recording
# ============================================================

def save_environment(
    output_path: Path,
) -> None:
    """
    Save the Python and PyTorch environment used for the run.
    """

    lines = []

    lines.append(
        f"Timestamp: {datetime.now().isoformat()}"
    )

    lines.append(
        f"Python: {sys.version}"
    )

    lines.append(
        f"PyTorch: {torch.__version__}"
    )

    import transformers

    lines.append(
        f"Transformers: {transformers.__version__}"
    )

    lines.append(
        f"Platform: {sys.platform}"
    )

    lines.append(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    lines.append(
        f"CUDA version: {torch.version.cuda}"
    )

    if torch.cuda.is_available():

        lines.append(
            f"GPU count: {torch.cuda.device_count()}"
        )

        for index in range(
            torch.cuda.device_count()
        ):
            lines.append(
                f"GPU {index}: "
                f"{torch.cuda.get_device_name(index)}"
            )

    else:
        lines.append(
            "GPU: No CUDA GPU available"
        )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# Argument parsing
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train unsupervised SimCSE."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=(
            "configs/"
            "unsupervised_bert_base.yaml"
        ),
        help="Path to YAML configuration.",
    )

    parser.add_argument(
        "--max-sentences",
        type=int,
        default=None,
        help=(
            "Optional safety override limiting "
            "the number of training sentences."
        ),
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Optional experiment name override."
        ),
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help=(
            "Path to a training_state.pt checkpoint "
            "from which to resume training."
        ),
    )

    return parser.parse_args()


# ============================================================
# Training
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------

    config = load_config(
        args.config
    )

    experiment_name = (
        args.experiment_name
        or config["experiment_name"]
    )

    # --------------------------------------------------------
    # Safety information
    # --------------------------------------------------------

    configured_dataset = config[
        "dataset_path"
    ]

    configured_batch_size = config[
        "batch_size"
    ]

    configured_epochs = config[
        "epochs"
    ]

    # --------------------------------------------------------
    # Create experiment directory
    # --------------------------------------------------------

    experiment_dir = (
        create_experiment_directory(
            experiment_name=experiment_name,
            config_path=args.config,
        )
    )

    # --------------------------------------------------------
    # Save environment
    # --------------------------------------------------------

    save_environment(
        experiment_dir
        / "environment.txt"
    )

    # --------------------------------------------------------
    # Create training log
    # --------------------------------------------------------

    log_path = (
        experiment_dir
        / "train.log"
    )

    # Simple file logger.
    log_file = log_path.open(
        "w",
        encoding="utf-8",
    )

    def log(message: str = ""):
        print(message)
        log_file.write(
            message + "\n"
        )
        log_file.flush()

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    seed = config["seed"]

    set_seed(seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    log("=" * 70)
    log("SIMCSE TRAINING")
    log("=" * 70)

    log(
        f"Experiment directory: "
        f"{experiment_dir}"
    )

    log(
        f"Device: {device}"
    )

    if torch.cuda.is_available():
        log(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    log("\nConfiguration:")

    for key, value in config.items():
        log(
            f"{key}: {value}"
        )

    if args.max_sentences is not None:
        log(
            "\nCommand-line sentence limit:"
        )
        log(
            f"{args.max_sentences}"
        )

    # --------------------------------------------------------
    # Load tokenizer
    # --------------------------------------------------------

    log("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"]
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    log("\nLoading dataset...")

    dataset = WikipediaSentenceDataset(
        file_path=config["dataset_path"],
        max_sentences=args.max_sentences,
    )

    log(
        f"Number of sentences: "
        f"{len(dataset)}"
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=config[
            "max_seq_length"
        ],
    )

    loader_generator = torch.Generator()

    loader_generator.manual_seed(
        seed
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=config["shuffle"],
        drop_last=config["drop_last"],
        num_workers=config["num_workers"],
        collate_fn=collator,
        generator=loader_generator,
    )

    log(
        f"Number of batches per epoch: "
        f"{len(loader)}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    log("\nLoading model...")

    model = SimCSEModel(
        model_name=config["model_name"],
        mlp_only_train=config[
            "mlp_only_train"
        ],
        dropout=config["dropout"],
        fixed_dropout_mask=config.get(
            "fixed_dropout_mask",
            False,
        ),
    )

    model.to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    loss_fn = SimCSELoss(
        temperature=config[
            "temperature"
        ],
    )

    # --------------------------------------------------------
    # Optimizer + learning-rate scheduler
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=config["learning_rate"],
    )

    total_training_steps = (
        len(loader)
        * config["epochs"]
    )

    num_warmup_steps = 0

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # --------------------------------------------------------
    # Resume checkpoint
    # --------------------------------------------------------

    start_epoch = 0
    global_step = 0

    if args.resume is not None:

        resume_path = Path(
            args.resume
        )

        if not resume_path.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: "
                f"{resume_path}"
            )

        log(
            "\nLoading resume checkpoint:"
        )

        log(
            str(resume_path)
        )

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "epoch",
            "global_step",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "config",
            "rng_state",
        }

        missing_keys = (
            required_keys
            - checkpoint.keys()
        )

        if missing_keys:
            raise KeyError(
                "Resume checkpoint is missing "
                "required keys: "
                f"{sorted(missing_keys)}"
            )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        scheduler.load_state_dict(
            checkpoint[
                "scheduler_state_dict"
            ]
        )

        if "rng_state" not in checkpoint:
            raise KeyError(
                "Resume checkpoint is missing "
                "required key: rng_state"
            )

        restore_rng_state(
            checkpoint["rng_state"],
            loader_generator,
        )

        start_epoch = (
            checkpoint["epoch"]
        )

        global_step = (
            checkpoint["global_step"]
        )

        log(
            f"Resumed from epoch: "
            f"{start_epoch}"
        )

        log(
            f"Resumed global step: "
            f"{global_step}"
        )

        log(
            f"Resumed scheduler step: "
            f"{scheduler.last_epoch}"
        )

        log(
            f"Resumed learning rate: "
            f"{scheduler.get_last_lr()[0]}"
        )

    log(
        f"\nTotal training steps: "
        f"{total_training_steps}"
    )

    log(
        f"Warmup steps: "
        f"{num_warmup_steps}"
    )

    log(
        f"Initial learning rate: "
        f"{config['learning_rate']}"
    )

    # --------------------------------------------------------
    # Metrics file
    # --------------------------------------------------------

    metrics_path = (
        experiment_dir
        / "metrics.csv"
    )

    metrics_file = metrics_path.open(
        "w",
        newline="",
        encoding="utf-8",
    )

    metrics_writer = csv.writer(
        metrics_file
    )

    metrics_writer.writerow(
        [
            "epoch",
            "step",
            "global_step",
            "loss",
            "positive_cosine",
            "negative_cosine",
            "learning_rate",
        ]
    )

    metrics_file.flush()

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    log("\n" + "=" * 70)
    log("STARTING TRAINING")
    log("=" * 70)

    for epoch in range(
        start_epoch,
        config["epochs"],
    ):

        model.train()

        epoch_loss = 0.0
        epoch_positive = 0.0
        epoch_negative = 0.0

        for step, batch in enumerate(
            loader,
            start=1,
        ):

            global_step += 1

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # SimCSE forward pass
            #
            # input_ids shape:
            #
            #     [batch, 2, sequence_length]
            #
            # The model flattens the two views internally,
            # runs BERT once, and returns:
            #
            #     z1: [batch, hidden_size]
            #     z2: [batch, hidden_size]
            #
            # The two identical token sequences receive
            # independent dropout masks.
            # ------------------------------------------------

            z1, z2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            loss = loss_fn(
                z1,
                z2,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config["max_grad_norm"],
            )

            optimizer.step()
            scheduler.step()

            current_lr = scheduler.get_last_lr()[0]

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            with torch.no_grad():
                diagnostics = loss_fn.diagnostics(
                    z1,
                    z2,
                )

                positive_similarity = (
                    diagnostics["positive_similarity"].item()
                )

                negative_similarity = (
                    diagnostics["negative_similarity"].item()
                )

            epoch_loss += loss.item()
            epoch_positive += (
                positive_similarity
            )
            epoch_negative += (
                negative_similarity
            )

            metrics_writer.writerow(
                [
                    epoch + 1,
                    step,
                    global_step,
                    loss.item(),
                    positive_similarity,
                    negative_similarity,
                    current_lr,
                ]
            )

            metrics_file.flush()

            log(
                f"Epoch "
                f"{epoch + 1}/"
                f"{config['epochs']} "
                f"| Step "
                f"{step:04d}/"
                f"{len(loader):04d} "
                f"| Loss "
                f"{loss.item():.6f} "
                f"| PosSim "
                f"{positive_similarity:.6f} "
                f"| NegSim "
                f"{negative_similarity:.6f} "
                f"| LR "
                f"{current_lr:.8f}"
            )

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        num_steps = len(loader)

        average_loss = (
            epoch_loss
            / num_steps
        )

        average_positive = (
            epoch_positive
            / num_steps
        )

        average_negative = (
            epoch_negative
            / num_steps
        )

        log("\n" + "-" * 70)
        log(
            f"Epoch {epoch + 1} summary"
        )
        log("-" * 70)

        log(
            f"Average loss: "
            f"{average_loss:.6f}"
        )

        log(
            f"Average positive cosine: "
            f"{average_positive:.6f}"
        )

        log(
            f"Average negative cosine: "
            f"{average_negative:.6f}"
        )

        # ----------------------------------------------------
        # Save checkpoint
        # ----------------------------------------------------

        checkpoint_dir = (
            experiment_dir
            / "checkpoint"
            / f"epoch_{epoch + 1}"
        )

        checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "rng_state": get_rng_state(
                    loader_generator
                ),
                "config": config,
            },
            checkpoint_dir
            / "training_state.pt",
        )

        log(
            f"\nCheckpoint saved: "
            f"{checkpoint_dir}"
        )

    # --------------------------------------------------------
    # Close files
    # --------------------------------------------------------

    metrics_file.close()
    log_file.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print(
        f"\nExperiment saved to:"
    )
    print(experiment_dir)


if __name__ == "__main__":
    main()