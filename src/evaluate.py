import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from checkpoint import load_simcse_checkpoint


DEFAULT_MAX_LENGTH = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate sentence embeddings."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--model",
        type=str,
        help=(
            "Hugging Face model name or local Hugging Face model path. "
            "Use this for a pretrained Hugging Face model."
        ),
    )

    group.add_argument(
        "--checkpoint",
        type=str,
        help=(
            "Path to a SimCSE training_state.pt checkpoint. "
            "Use this for a checkpoint produced by src/train.py."
        ),
    )

    parser.add_argument(
        "--pooler",
        type=str,
        choices=[
            "cls_before_pooler",
            "cls",
        ],
        default="cls_before_pooler",
        help=(
            "Representation to use for a SimCSE checkpoint. "
            "'cls_before_pooler' = raw CLS; "
            "'cls' = CLS + training MLP."
        ),
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_LENGTH,
        help="Maximum token sequence length. Default: 32.",
    )

    return parser.parse_args()


def encode_sentences(
    model,
    tokenizer,
    sentences,
    device,
    max_length,
):
    """
    Encode sentences with a generic Hugging Face model.

    This returns the raw last-hidden-state [CLS] representation.
    """

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(
            **inputs,
            return_dict=True,
        )

    return outputs.last_hidden_state[:, 0]


def encode_simcse_model(
    model,
    tokenizer,
    sentences,
    device,
    max_length,
    use_mlp=False,
):
    """
    Encode sentences using our SimCSEModel.

    For unsupervised SimCSE evaluation, the recommended representation
    is raw CLS, so use_mlp=False.

    If --pooler cls is requested, use_mlp=True and obtain CLS + MLP.
    """

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    model.eval()

    with torch.no_grad():
        embeddings = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_mlp=use_mlp,
        )

    return embeddings


def cosine_similarity(
    embeddings_a,
    embeddings_b,
):
    """
    Calculate row-wise cosine similarity.
    """

    embeddings_a = F.normalize(
        embeddings_a,
        p=2,
        dim=1,
    )

    embeddings_b = F.normalize(
        embeddings_b,
        p=2,
        dim=1,
    )

    return (
        embeddings_a * embeddings_b
    ).sum(dim=1)


def main():
    args = parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("SIMCSE SENTENCE EMBEDDING EVALUATOR")
    print("=" * 60)

    print("\nDevice:")
    print(device)

    print("\nMax length:")
    print(args.max_length)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    if args.checkpoint:

        print("\nCheckpoint:")
        print(args.checkpoint)

        checkpoint = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=False,
        )

        config = checkpoint["config"]

        model_name = config["model_name"]
        mlp_only_train = config["mlp_only_train"]

        print("\nModel from checkpoint:")
        print(model_name)

        tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        model, checkpoint_data = load_simcse_checkpoint(
            checkpoint_path=args.checkpoint,
            model_name=model_name,
            mlp_only_train=mlp_only_train,
            device=device,
        )

        print("\nLoaded checkpoint epoch:")
        print(checkpoint_data["epoch"])

        print("\nLoaded checkpoint global step:")
        print(checkpoint_data["global_step"])

        print("\nEvaluation pooler:")
        print(args.pooler)

    else:

        print("\nModel:")
        print(args.model)

        print("\nEvaluation representation:")
        print("raw CLS")

        tokenizer = AutoTokenizer.from_pretrained(
            args.model
        )

        model = AutoModel.from_pretrained(
            args.model
        )

        model.to(device)
        model.eval()

    # --------------------------------------------------------
    # Test sentences
    # --------------------------------------------------------

    sentences_a = [
        "A woman is reading.",
        "A man is playing a guitar.",
    ]

    sentences_b = [
        "A woman is reading a book.",
        "A person is playing music.",
    ]

    if args.checkpoint:

        use_mlp = args.pooler == "cls"

        embeddings_a = encode_simcse_model(
            model=model,
            tokenizer=tokenizer,
            sentences=sentences_a,
            device=device,
            max_length=args.max_length,
            use_mlp=use_mlp,
        )

        embeddings_b = encode_simcse_model(
            model=model,
            tokenizer=tokenizer,
            sentences=sentences_b,
            device=device,
            max_length=args.max_length,
            use_mlp=use_mlp,
        )

    else:

        embeddings_a = encode_sentences(
            model=model,
            tokenizer=tokenizer,
            sentences=sentences_a,
            device=device,
            max_length=args.max_length,
        )

        embeddings_b = encode_sentences(
            model=model,
            tokenizer=tokenizer,
            sentences=sentences_b,
            device=device,
            max_length=args.max_length,
        )

    similarities = cosine_similarity(
        embeddings_a,
        embeddings_b,
    )

    print("\nTest sentence similarities:")

    for i in range(len(sentences_a)):

        print(f"\nA: {sentences_a[i]}")
        print(f"B: {sentences_b[i]}")

        print(
            f"Cosine similarity: "
            f"{similarities[i].item():.4f}"
        )


if __name__ == "__main__":
    main()
