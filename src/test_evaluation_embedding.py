import torch
from transformers import AutoTokenizer

from model import SimCSEModel


MODEL_NAME = "bert-base-uncased"


def main():

    print("=" * 70)
    print("SIMCSE EVALUATION EMBEDDING TEST")
    print("=" * 70)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    sentences = [
        "A woman is reading a book.",
        "A man is playing a guitar.",
    ]

    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

    print(
        "\nInput shape:",
        batch["input_ids"].shape,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.to(device)

    # ========================================================
    # TEST 1
    # Explicit raw CLS vs explicit no-MLP
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST 1: EXPLICIT EVALUATION REPRESENTATION")
    print("=" * 70)

    model.eval()

    with torch.no_grad():

        explicit_false = (
            model.sentence_embedding(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_mlp=False,
            )
        )

        automatic_eval = (
            model.sentence_embedding(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_mlp=None,
            )
        )

    maximum_difference = torch.max(
        torch.abs(
            explicit_false
            - automatic_eval
        )
    ).item()

    print(
        "Explicit use_mlp=False shape:",
        explicit_false.shape,
    )

    print(
        "Automatic evaluation shape:",
        automatic_eval.shape,
    )

    print(
        "Maximum difference:",
        maximum_difference,
    )

    assert torch.allclose(
        explicit_false,
        automatic_eval,
        atol=1e-6,
    )

    print(
        "\nExplicit evaluation representation: PASSED"
    )

    # ========================================================
    # TEST 2
    # Evaluation determinism
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST 2: EVALUATION DETERMINISM")
    print("=" * 70)

    with torch.no_grad():

        embedding_1 = (
            model.sentence_embedding(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_mlp=False,
            )
        )

        embedding_2 = (
            model.sentence_embedding(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_mlp=False,
            )
        )

    maximum_difference = torch.max(
        torch.abs(
            embedding_1
            - embedding_2
        )
    ).item()

    print(
        "Maximum difference:",
        maximum_difference,
    )

    assert torch.allclose(
        embedding_1,
        embedding_2,
        atol=1e-6,
    )

    print(
        "\nEvaluation determinism: PASSED"
    )

    # ========================================================
    # TEST 3
    # Training representation differs from raw CLS
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST 3: TRAINING VS EVALUATION REPRESENTATION")
    print("=" * 70)

    model.train()

    # use_mlp=True explicitly disables ambiguity caused by
    # training/evaluation mode.
    training_embedding = (
        model.sentence_embedding(
            input_ids=batch["input_ids"],
            attention_mask=batch[
                "attention_mask"
            ],
            use_mlp=True,
        )
    )

    model.eval()

    with torch.no_grad():

        evaluation_embedding = (
            model.sentence_embedding(
                input_ids=batch["input_ids"],
                attention_mask=batch[
                    "attention_mask"
                ],
                use_mlp=False,
            )
        )

    maximum_difference = torch.max(
        torch.abs(
            training_embedding
            - evaluation_embedding
        )
    ).item()

    print(
        "Training embedding shape:",
        training_embedding.shape,
    )

    print(
        "Evaluation embedding shape:",
        evaluation_embedding.shape,
    )

    print(
        "Maximum difference:",
        maximum_difference,
    )

    assert maximum_difference > 1e-4

    print(
        "\nTraining/evaluation distinction: PASSED"
    )

    # ========================================================
    # TEST 4
    # Cosine similarity interface
    # ========================================================

    print("\n" + "=" * 70)
    print("TEST 4: SENTENCE EMBEDDING COSINE")
    print("=" * 70)

    cosine = torch.nn.functional.cosine_similarity(
        evaluation_embedding[0].unsqueeze(0),
        evaluation_embedding[1].unsqueeze(0),
    ).item()

    print(
        "Evaluation cosine similarity:",
        cosine,
    )

    assert -1.0 <= cosine <= 1.0

    print(
        "\nCosine similarity test: PASSED"
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print("\n" + "=" * 70)
    print(
        "ALL EVALUATION EMBEDDING TESTS PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()