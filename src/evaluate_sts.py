import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from model import SimCSEModel


DEFAULT_MODEL = "bert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SimCSE on STS-B."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=(
            "Base Hugging Face model name. "
            "Used when --checkpoint is not provided."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a local SimCSE training_state.pt "
            "checkpoint."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "train",
            "validation",
            "test",
        ],
        default="validation",
        help="STS-B split.",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=32,
        help="Maximum tokenizer sequence length.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Evaluation batch size.",
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Optional number of examples to evaluate. "
            "Useful for debugging."
        ),
    )

    return parser.parse_args()


def load_stsb(
    split: str,
    max_samples: int | None = None,
):
    """
    Load the GLUE STS-B dataset.

    Each example contains:
        sentence1
        sentence2
        label
    """

    dataset = load_dataset(
        "glue",
        "stsb",
        split=split,
    )

    if max_samples is not None:
        max_samples = min(
            max_samples,
            len(dataset),
        )

        dataset = dataset.select(
            range(max_samples)
        )

    return dataset


def encode_sentences(
    model: SimCSEModel,
    tokenizer,
    sentences,
    device: torch.device,
    max_length: int,
    batch_size: int,
):
    """
    Encode sentences using the unsupervised SimCSE
    evaluation representation.

    For mlp_only_train=True:

        training:
            BERT -> CLS -> MLP

        evaluation:
            BERT -> raw CLS

    Therefore evaluation explicitly uses:

        use_mlp=False
    """

    model.eval()

    embeddings = []

    with torch.no_grad():

        for start in range(
            0,
            len(sentences),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(sentences),
            )

            batch_sentences = sentences[
                start:end
            ]

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

            batch_embeddings = (
                model.sentence_embedding(
                    input_ids=encoded[
                        "input_ids"
                    ],
                    attention_mask=encoded[
                        "attention_mask"
                    ],
                    use_mlp=False,
                )
            )

            embeddings.append(
                batch_embeddings.cpu()
            )

    return torch.cat(
        embeddings,
        dim=0,
    )


def cosine_similarity(
    embeddings1: torch.Tensor,
    embeddings2: torch.Tensor,
) -> np.ndarray:
    """
    Compute paired cosine similarities.

    embeddings1[i] is compared only with
    embeddings2[i].
    """

    embeddings1 = torch.nn.functional.normalize(
        embeddings1,
        p=2,
        dim=1,
    )

    embeddings2 = torch.nn.functional.normalize(
        embeddings2,
        p=2,
        dim=1,
    )

    similarities = (
        embeddings1 * embeddings2
    ).sum(dim=1)

    return similarities.numpy()


def evaluate_stsb(
    model: SimCSEModel,
    tokenizer,
    dataset,
    device: torch.device,
    max_length: int,
    batch_size: int,
):
    """
    Evaluate STS-B using:

        cosine(sentence1, sentence2)

    and compare predictions against the human
    similarity scores using Spearman correlation.

    No regression model is trained.
    """

    sentence1 = dataset["sentence1"]
    sentence2 = dataset["sentence2"]

    gold_scores = np.asarray(
        dataset["label"],
        dtype=np.float64,
    )

    embeddings1 = encode_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence1,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    embeddings2 = encode_sentences(
        model=model,
        tokenizer=tokenizer,
        sentences=sentence2,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )

    predicted_scores = cosine_similarity(
        embeddings1,
        embeddings2,
    )

    result = spearmanr(
        gold_scores,
        predicted_scores,
    )

    return {
        "spearman": float(
            result.statistic
        ),
        "gold_scores": gold_scores,
        "predicted_scores": predicted_scores,
    }


def load_model_and_tokenizer(
    args,
    device: torch.device,
):
    """
    Load either:

    1. A normal Hugging Face model using --model

    or

    2. A locally trained SimCSE checkpoint using
       --checkpoint.

    For a local checkpoint, the model name and
    mlp_only_train setting are taken from the
    saved training configuration.
    """

    checkpoint = None

    # --------------------------------------------------------
    # Local checkpoint mode
    # --------------------------------------------------------

    if args.checkpoint is not None:

        checkpoint_path = Path(
            args.checkpoint
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found:\n"
                f"{checkpoint_path}"
            )

        print(
            "\nLoading checkpoint..."
        )

        print(
            "Checkpoint:",
            checkpoint_path,
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "epoch",
            "global_step",
            "model_state_dict",
            "config",
        }

        missing_keys = (
            required_keys
            - checkpoint.keys()
        )

        if missing_keys:
            raise KeyError(
                "Checkpoint is missing "
                "required keys: "
                f"{sorted(missing_keys)}"
            )

        checkpoint_config = (
            checkpoint["config"]
        )

        model_name = checkpoint_config[
            "model_name"
        ]

        mlp_only_train = checkpoint_config[
            "mlp_only_train"
        ]

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"],
        )

        print(
            "Checkpoint global step:",
            checkpoint["global_step"],
        )

        print(
            "Model:",
            model_name,
        )

        print(
            "MLP only during training:",
            mlp_only_train,
        )

    # --------------------------------------------------------
    # Normal Hugging Face model mode
    # --------------------------------------------------------

    else:

        model_name = args.model

        mlp_only_train = True

        print(
            "\nNo local checkpoint provided."
        )

        print(
            "Loading Hugging Face model:",
            model_name,
        )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading SimCSE model..."
    )

    model = SimCSEModel(
        model_name=model_name,
        mlp_only_train=mlp_only_train,
    )

    # --------------------------------------------------------
    # Load local checkpoint weights
    # --------------------------------------------------------

    if checkpoint is not None:

        print(
            "Loading checkpoint weights..."
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ],
            strict=True,
        )

        print(
            "Checkpoint weights loaded successfully."
        )

    model.to(device)
    model.eval()

    return model, tokenizer


def main():

    args = parse_args()

    print("=" * 70)
    print("SIMCSE STS-B EVALUATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    if args.checkpoint is not None:

        print(
            "\nEvaluation mode:",
            "LOCAL CHECKPOINT",
        )

        print(
            "Checkpoint:",
            args.checkpoint,
        )

    else:

        print(
            "\nEvaluation mode:",
            "HUGGING FACE MODEL",
        )

        print(
            "Model:",
            args.model,
        )

    print(
        "Split:",
        args.split,
    )

    print(
        "Max length:",
        args.max_length,
    )

    print(
        "Batch size:",
        args.batch_size,
    )

    if args.max_samples is not None:

        print(
            "Maximum samples:",
            args.max_samples,
        )

    else:

        print(
            "Maximum samples:",
            "None (full split)",
        )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\nDevice:",
        device,
    )

    # --------------------------------------------------------
    # Model + tokenizer
    # --------------------------------------------------------

    model, tokenizer = (
        load_model_and_tokenizer(
            args=args,
            device=device,
        )
    )

    # --------------------------------------------------------
    # STS-B
    # --------------------------------------------------------

    print(
        "\nLoading STS-B dataset..."
    )

    dataset = load_stsb(
        split=args.split,
        max_samples=args.max_samples,
    )

    print(
        "Number of examples:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Show one example
    # --------------------------------------------------------

    print(
        "\nFirst example:"
    )

    print(
        "Sentence 1:",
        dataset[0]["sentence1"],
    )

    print(
        "Sentence 2:",
        dataset[0]["sentence2"],
    )

    print(
        "Gold score:",
        dataset[0]["label"],
    )

    # --------------------------------------------------------
    # Evaluation representation
    # --------------------------------------------------------

    print(
        "\nEvaluation representation:"
    )

    print(
        "BERT -> raw CLS"
    )

    print(
        "MLP used during evaluation:",
        False,
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    print(
        "\nRunning evaluation..."
    )

    result = evaluate_stsb(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        max_length=args.max_length,
        batch_size=args.batch_size,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print(
        "STS-B Spearman:",
        f"{result['spearman']:.6f}",
    )

    print(
        "STS-B Spearman (%):",
        f"{result['spearman'] * 100:.2f}",
    )

    print(
        "-" * 70
    )

    print(
        "\nEvaluation complete."
    )


if __name__ == "__main__":
    main()