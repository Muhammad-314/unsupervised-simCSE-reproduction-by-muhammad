"""Run Phase A.3 geometry analysis for the final 50K experiment set.

This driver is intentionally compatible with the current alignment_uniformity.py.

Alignment follows the SimCSE paper:
    - positive STS-B pairs are those with gold score > 4
    - STS-B Spearman uses all validation pairs
    - uniformity uses all unique sentences in the validation split

No training or checkpoint modification is performed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from alignment_uniformity import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    alignment_loss,
    batched_encode,
    evaluate_stsb_correlations,
    load_model_and_tokenizer,
    load_stsb,
    uniformity_loss,
)

# The SimCSE paper uses STS-B pairs with score > 4 as p_pos.
DEFAULT_POSITIVE_THRESHOLD = 4.0

EXPERIMENTS = [
    ("raw_BERT", None),
    ("SimCSE_50K", "reduced_50k"),
    ("no_dropout", "dropout_no_dropout_50k"),
    ("fixed_mask_0.10", "dropout_fixed_mask_50k"),
    ("dropout_0.20", "dropout_020_50k"),
    ("temperature_0.01", "temperature_001_50k"),
    ("temperature_1.00", "temperature_100_50k"),
]


def find_checkpoint(experiments_root: Path, experiment_name: str) -> Path:
    """Find the epoch-1 checkpoint for a named final experiment."""
    matches = sorted(
        experiments_root.glob(
            f"{experiment_name}/*/checkpoint/epoch_1/training_state.pt"
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"No epoch_1 checkpoint found for {experiment_name} "
            f"under {experiments_root}"
        )

    if len(matches) > 1:
        print(
            f"WARNING: multiple checkpoints found for {experiment_name}; "
            f"using {matches[-1]}"
        )

    return matches[-1]


def analyze_phase_a3(
    *,
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_length: int,
    batch_size: int,
    positive_threshold: float,
    uniformity_samples: int | None,
    uniformity_chunk_size: int,
) -> dict[str, float | int]:
    """Compute STS-B, paper-style alignment, and uniformity.

    Important distinction:
      * STS-B Spearman uses every STS-B validation pair.
      * Alignment uses only pairs with label > positive_threshold.
      * Uniformity uses unique sentences from the full validation set.
    """
    sentence1 = list(dataset["sentence1"])
    sentence2 = list(dataset["sentence2"])
    gold_scores = [float(x) for x in dataset["label"]]

    # Encode both sides once. These embeddings are reused for STS-B and alignment.
    embeddings1 = batched_encode(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence1,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        use_mlp=False,
    )

    embeddings2 = batched_encode(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence2,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        use_mlp=False,
    )

    # STS-B evaluation: ALL validation pairs.
    sts_b_spearman = evaluate_stsb_correlations(
        embeddings1,
        embeddings2,
        gold_scores,
    )

    # Alignment: only STS-B positive pairs with score > 4.
    positive_mask = torch.tensor(
        [score > positive_threshold for score in gold_scores],
        dtype=torch.bool,
    )

    num_positive_pairs = int(positive_mask.sum().item())
    if num_positive_pairs == 0:
        raise ValueError(
            f"No STS-B pairs have score > {positive_threshold}."
        )

    positive_embeddings1 = embeddings1[positive_mask]
    positive_embeddings2 = embeddings2[positive_mask]

    alignment = alignment_loss(
        positive_embeddings1,
        positive_embeddings2,
    )

    # Uniformity: full STS-B sentence population, de-duplicated.
    unique_sentences = list(dict.fromkeys(sentence1 + sentence2))

    if uniformity_samples is not None:
        unique_sentences = unique_sentences[:uniformity_samples]

    unique_embeddings = batched_encode(
        model=model,
        tokenizer=tokenizer,
        sentences=unique_sentences,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        use_mlp=False,
    )

    uniformity = uniformity_loss(
        unique_embeddings,
        chunk_size=uniformity_chunk_size,
    )

    return {
        "num_stsb_pairs": len(sentence1),
        "num_positive_pairs": num_positive_pairs,
        "positive_threshold": positive_threshold,
        "num_unique_sentences": len(unique_sentences),
        "sts_b_spearman": sts_b_spearman,
        "alignment": alignment,
        "uniformity": uniformity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase A.3: analyze all final 50K SimCSE models"
    )

    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=Path("experiments"),
    )
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=DEFAULT_POSITIVE_THRESHOLD,
        help="STS-B score must be strictly greater than this for alignment.",
    )
    parser.add_argument(
        "--uniformity-samples",
        type=int,
        default=None,
        help="Optional cap on unique sentences used for uniformity.",
    )
    parser.add_argument(
        "--uniformity-chunk-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("analysis/phase_a3/geometry_results.csv"),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.max_length <= 0:
        parser.error("--max-length must be positive.")
    if args.positive_threshold < 0 or args.positive_threshold > 5:
        parser.error("--positive-threshold must be between 0 and 5.")
    if args.uniformity_samples is not None and args.uniformity_samples < 2:
        parser.error("--uniformity-samples must be at least 2.")
    if args.uniformity_chunk_size <= 0:
        parser.error("--uniformity-chunk-size must be positive.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")
    print(f"Experiments root: {args.experiments_root}")
    print(f"STS-B split: {args.split}")
    print(
        "Alignment positives: "
        f"STS-B score > {args.positive_threshold}"
    )

    dataset = load_stsb(args.split)
    results: list[dict[str, object]] = []

    for i, (label, directory) in enumerate(EXPERIMENTS, start=1):
        checkpoint = (
            None
            if directory is None
            else find_checkpoint(args.experiments_root, directory)
        )

        print("\n" + "=" * 72)
        print(f"[{i}/{len(EXPERIMENTS)}] {label}")

        if checkpoint is not None:
            print(f"Checkpoint: {checkpoint}")

        model, tokenizer, metadata = load_model_and_tokenizer(
            model_name="bert-base-uncased" if checkpoint is None else None,
            checkpoint_path=None if checkpoint is None else str(checkpoint),
            device=device,
        )

        result = analyze_phase_a3(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            device=device,
            max_length=args.max_length,
            batch_size=args.batch_size,
            positive_threshold=args.positive_threshold,
            uniformity_samples=args.uniformity_samples,
            uniformity_chunk_size=args.uniformity_chunk_size,
        )

        result = {
            "experiment": label,
            "model_name": metadata["model_name"],
            "checkpoint": (
                "bert-base-uncased"
                if checkpoint is None
                else str(checkpoint)
            ),
            **result,
        }

        results.append(result)

        print(
            f"STS-B={result['sts_b_spearman']:.6f} | "
            f"alignment={result['alignment']:.6f} | "
            f"uniformity={result['uniformity']:.6f} | "
            f"positive_pairs={result['num_positive_pairs']}"
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not results:
        raise RuntimeError("No experiments were analyzed.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fields = list(results[0].keys())

    with args.output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 72)
    print(f"Saved combined results: {args.output_csv}")


if __name__ == "__main__":
    main()
