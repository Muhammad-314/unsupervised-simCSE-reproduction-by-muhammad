"""Alignment and uniformity analysis for SimCSE sentence embeddings.

This module implements the Wang--Isola representation-geometry metrics used
in the SimCSE analysis:

    alignment(x, x+) = mean ||norm(f(x)) - norm(f(x+))||_2^2

    uniformity = log E_{x,y}[exp(-2 ||norm(f(x)) - norm(f(y))||_2^2)]

Both metrics are lower-is-better. Alignment is computed from the paired
sentences in STS-B; uniformity is computed over the sentence embeddings from
the same STS-B split (with the diagonal excluded).

The script supports:
    * raw Hugging Face BERT (--model)
    * local SimCSE training checkpoints (--checkpoint)

For SimCSE checkpoints the saved dropout configuration is restored so that the
model architecture matches the training run. Evaluation is always performed in
model.eval() mode and uses raw CLS by default, matching the project's existing
unsupervised SimCSE STS-B evaluation path.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F



DEFAULT_MAX_LENGTH = 32
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_SAMPLES = None
DEFAULT_UNIFORMITY_SAMPLES = None
DEFAULT_UNIFORMITY_CHUNK_SIZE = 512


def normalize_embeddings(embeddings: torch.Tensor) -> torch.Tensor:
    """L2-normalize embeddings row-wise."""

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [n, hidden_size], "
            f"got {tuple(embeddings.shape)}"
        )

    return F.normalize(embeddings, p=2, dim=1)


def alignment_loss(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
) -> float:
    """Compute alignment on paired positive examples.

    The metric is the mean squared Euclidean distance between normalized
    representations of positive pairs.
    """

    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(
            "Positive-pair embedding tensors must have identical shapes: "
            f"{tuple(embeddings_a.shape)} vs {tuple(embeddings_b.shape)}"
        )

    a = normalize_embeddings(embeddings_a)
    b = normalize_embeddings(embeddings_b)

    distances_sq = (a - b).pow(2).sum(dim=1)
    return float(distances_sq.mean().item())


def uniformity_loss(
    embeddings: torch.Tensor,
    chunk_size: int = DEFAULT_UNIFORMITY_CHUNK_SIZE,
) -> float:
    """Compute the hyperspherical uniformity metric.

    This evaluates all unordered pairs and excludes self-pairs. Chunking keeps
    memory bounded for larger STS collections.
    """

    if embeddings.ndim != 2:
        raise ValueError(
            "embeddings must have shape [n, hidden_size], "
            f"got {tuple(embeddings.shape)}"
        )

    if embeddings.shape[0] < 2:
        raise ValueError(
            "Uniformity requires at least two embeddings."
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    z = normalize_embeddings(embeddings)

    # Accumulate the sum of exp(-2 * squared distance) over i < j.
    # Compute in float64 for stable aggregation even though the model outputs
    # are normally float32.
    z = z.to(dtype=torch.float64)
    n = z.shape[0]
    pair_sum = torch.zeros((), dtype=torch.float64, device=z.device)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = z[start:end]

        # Squared Euclidean distance on normalized vectors:
        # ||u-v||^2 = 2 - 2 u^T v.
        distances_sq = 2.0 - 2.0 * (block @ z.T)
        values = torch.exp(-2.0 * distances_sq)

        # Keep only pairs with global i < j. This removes the diagonal and
        # avoids counting each pair twice.
        local_rows = torch.arange(start, end, device=z.device)
        global_cols = torch.arange(n, device=z.device)
        upper_mask = global_cols.unsqueeze(0) > local_rows.unsqueeze(1)

        if upper_mask.any():
            pair_sum = pair_sum + values[upper_mask].sum()

    num_pairs = n * (n - 1) // 2
    mean_value = pair_sum / float(num_pairs)

    # mean_value is strictly positive in floating point.
    return float(torch.log(mean_value).item())


def load_stsb(
    split: str = "validation",
    max_samples: int | None = None,
):
    """Load STS-B and optionally restrict the number of rows."""

    from datasets import load_dataset

    dataset = load_dataset("glue", "stsb", split=split)

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    return dataset


def batched_encode(
    model,
    tokenizer,
    sentences: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
    use_mlp: bool = False,
) -> torch.Tensor:
    """Encode sentences using the project's evaluation representation."""

    if not sentences:
        raise ValueError("Cannot encode an empty sentence collection.")

    model.eval()
    chunks: list[torch.Tensor] = []

    with torch.no_grad():
        for start in range(0, len(sentences), batch_size):
            batch_sentences = sentences[start : start + batch_size]

            encoded = tokenizer(
                batch_sentences,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
            }

            from model_ablations import SimCSEModel

            if isinstance(model, SimCSEModel):
                embeddings = model.sentence_embedding(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    use_mlp=use_mlp,
                )
            else:
                outputs = model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    return_dict=True,
                )
                embeddings = outputs.last_hidden_state[:, 0]

            chunks.append(embeddings.detach().cpu())

    return torch.cat(chunks, dim=0)


def evaluate_stsb_correlations(
    embeddings_a: torch.Tensor,
    embeddings_b: torch.Tensor,
    gold_scores: Sequence[float],
) -> float:
    """Compute STS-B Spearman correlation from paired embeddings."""

    a = normalize_embeddings(embeddings_a)
    b = normalize_embeddings(embeddings_b)
    predicted = (a * b).sum(dim=1).numpy()
    from scipy.stats import spearmanr

    result = spearmanr(np.asarray(gold_scores, dtype=np.float64), predicted)
    return float(result.statistic)


def load_model_and_tokenizer(
    *,
    model_name: str | None,
    checkpoint_path: str | None,
    device: torch.device,
):
    """Load either raw BERT or a local SimCSE checkpoint."""

    if (model_name is None) == (checkpoint_path is None):
        raise ValueError("Provide exactly one of model_name or checkpoint_path.")

    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )
        config = checkpoint["config"]

        model_name = config["model_name"]
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        from model_ablations import SimCSEModel

        model = SimCSEModel(
            model_name=model_name,
            mlp_only_train=config.get("mlp_only_train", True),
            dropout=float(config.get("dropout", 0.1)),
            fixed_dropout_mask=bool(config.get("fixed_dropout_mask", False)),
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device)
        model.eval()

        return model, tokenizer, {
            "type": "checkpoint",
            "model_name": model_name,
            "checkpoint": str(path),
            "config": config,
            "epoch": checkpoint.get("epoch"),
            "global_step": checkpoint.get("global_step"),
        }

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    return model, tokenizer, {
        "type": "model",
        "model_name": model_name,
    }


def analyze(
    *,
    model,
    tokenizer,
    dataset,
    device: torch.device,
    max_length: int,
    batch_size: int,
    uniformity_samples: int | None,
    uniformity_chunk_size: int,
    use_mlp: bool = False,
) -> dict[str, float | int]:
    """Run the complete STS-B geometry analysis."""

    sentence1 = list(dataset["sentence1"])
    sentence2 = list(dataset["sentence2"])
    gold_scores = list(dataset["label"])

    embeddings1 = batched_encode(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence1,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        use_mlp=use_mlp,
    )

    embeddings2 = batched_encode(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence2,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
        use_mlp=use_mlp,
    )

    alignment = alignment_loss(embeddings1, embeddings2)
    spearman = evaluate_stsb_correlations(
        embeddings1,
        embeddings2,
        gold_scores,
    )

    # Uniformity uses unique sentences rather than duplicated STS pairs.
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
        use_mlp=use_mlp,
    )

    uniformity = uniformity_loss(
        unique_embeddings,
        chunk_size=uniformity_chunk_size,
    )

    return {
        "num_stsb_pairs": len(sentence1),
        "num_unique_sentences": len(unique_sentences),
        "sts_b_spearman": spearman,
        "alignment": alignment,
        "uniformity": uniformity,
    }


def write_result_csv(path: str | Path, result: dict) -> None:
    """Write one analysis result as a one-row CSV."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SimCSE alignment/uniformity analysis on STS-B."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--model",
        type=str,
        help="Hugging Face model name/path, e.g. bert-base-uncased.",
    )
    group.add_argument(
        "--checkpoint",
        type=str,
        help="Local SimCSE training_state.pt checkpoint.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES)
    parser.add_argument(
        "--uniformity-samples",
        type=int,
        default=DEFAULT_UNIFORMITY_SAMPLES,
        help="Optional cap on unique sentences used for uniformity.",
    )
    parser.add_argument("--uniformity-chunk-size", type=int, default=DEFAULT_UNIFORMITY_CHUNK_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--pooler",
        choices=["cls_before_pooler", "cls"],
        default="cls_before_pooler",
        help="SimCSE checkpoint representation. Default is raw CLS.",
    )
    parser.add_argument("--output-csv", type=str, default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, tokenizer, metadata = load_model_and_tokenizer(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    dataset = load_stsb(
        split=args.split,
        max_samples=args.max_samples,
    )

    use_mlp = args.pooler == "cls"

    result = analyze(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
        uniformity_samples=args.uniformity_samples,
        uniformity_chunk_size=args.uniformity_chunk_size,
        use_mlp=use_mlp,
    )

    result = {
        "model_type": metadata["type"],
        "model_name": metadata["model_name"],
        "checkpoint": metadata.get("checkpoint", ""),
        "split": args.split,
        "pooler": args.pooler,
        "device": str(device),
        **result,
    }

    print("=" * 70)
    print("SIMCSE ALIGNMENT / UNIFORMITY ANALYSIS")
    print("=" * 70)
    for key, value in result.items():
        print(f"{key}: {value}")

    if args.output_csv:
        write_result_csv(args.output_csv, result)
        print("\nSaved:", args.output_csv)


if __name__ == "__main__":
    main()
