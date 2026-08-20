import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model import SimCSEModel


MODEL_NAME = "bert-base-uncased"


def build_inputs():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    sentences = [
        "A dog is running.",
        "A cat is sleeping.",
    ]

    batch = tokenizer(
        [
            sentence
            for sentence in sentences
            for _ in range(2)
        ],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    batch_size = len(sentences)

    input_ids = batch["input_ids"].view(
        batch_size,
        2,
        -1,
    )

    attention_mask = (
        batch["attention_mask"].view(
            batch_size,
            2,
            -1,
        )
    )

    return (
        tokenizer,
        input_ids,
        attention_mask,
    )


def test_shapes():
    print("=" * 60)
    print("TEST 1: TWO-VIEW INPUT / OUTPUT SHAPES")
    print("=" * 60)

    (
        tokenizer,
        input_ids,
        attention_mask,
    ) = build_inputs()

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.train()

    z1, z2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    print(
        "Input shape:",
        input_ids.shape,
    )

    print(
        "z1 shape:",
        z1.shape,
    )

    print(
        "z2 shape:",
        z2.shape,
    )

    assert input_ids.ndim == 3

    assert input_ids.shape[1] == 2

    assert z1.shape == (
        input_ids.shape[0],
        model.bert.config.hidden_size,
    )

    assert z2.shape == z1.shape

    print(
        "\nTwo-view shape test: PASSED"
    )


def test_training_dropout():
    print("\n" + "=" * 60)
    print("TEST 2: TRAINING DROPOUT")
    print("=" * 60)

    (
        tokenizer,
        input_ids,
        attention_mask,
    ) = build_inputs()

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.train()

    z1_a, z2_a = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    z1_b, z2_b = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    difference_z1 = torch.max(
        torch.abs(z1_a - z1_b)
    ).item()

    difference_z2 = torch.max(
        torch.abs(z2_a - z2_b)
    ).item()

    similarity_z1 = F.cosine_similarity(
        z1_a,
        z1_b,
        dim=1,
    ).mean().item()

    similarity_z2 = F.cosine_similarity(
        z2_a,
        z2_b,
        dim=1,
    ).mean().item()

    print(
        "z1 cosine similarity:",
        similarity_z1,
    )

    print(
        "z2 cosine similarity:",
        similarity_z2,
    )

    print(
        "Maximum z1 difference:",
        difference_z1,
    )

    print(
        "Maximum z2 difference:",
        difference_z2,
    )

    assert difference_z1 > 0.0

    assert difference_z2 > 0.0

    print(
        "\nTraining dropout test: PASSED"
    )


def test_evaluation_determinism():
    print("\n" + "=" * 60)
    print("TEST 3: EVALUATION DETERMINISM")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    sentences = [
        "A dog is running.",
    ]

    batch = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.eval()

    z1 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    z2 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    difference = torch.max(
        torch.abs(z1 - z2)
    ).item()

    similarity = F.cosine_similarity(
        z1,
        z2,
        dim=1,
    ).item()

    print(
        "Cosine similarity:",
        similarity,
    )

    print(
        "Maximum difference:",
        difference,
    )

    assert difference == 0.0

    print(
        "\nEvaluation determinism test: PASSED"
    )


def test_mlp_only_train():
    print("\n" + "=" * 60)
    print("TEST 4: MLP-ONLY-DURING-TRAINING")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    batch = tokenizer(
        ["A dog is running."],
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    # --------------------------------------------------------
    # Evaluation representation
    # --------------------------------------------------------

    model.eval()

    eval_embedding = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    raw_cls = model.get_cls_embedding(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    eval_difference = torch.max(
        torch.abs(
            eval_embedding - raw_cls
        )
    ).item()

    print(
        "Evaluation embedding shape:",
        eval_embedding.shape,
    )

    print(
        "Raw CLS shape:",
        raw_cls.shape,
    )

    print(
        "Evaluation vs raw CLS maximum difference:",
        eval_difference,
    )

    assert eval_difference == 0.0

    # --------------------------------------------------------
    # Training representation
    # --------------------------------------------------------

    model.train()

    train_embedding = model.sentence_embedding(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )

    train_difference = torch.max(
        torch.abs(
            train_embedding - raw_cls
        )
    ).item()

    print(
        "Training embedding shape:",
        train_embedding.shape,
    )

    print(
        "Training vs raw CLS maximum difference:",
        train_difference,
    )

    assert train_difference > 0.0

    print(
        "\nMLP-only-train test: PASSED"
    )


def test_mlp_initialization():
    print("\n" + "=" * 60)
    print("TEST 5: MLP INITIALIZATION")
    print("=" * 60)

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    weight = model.mlp[0].weight

    bias = model.mlp[0].bias

    weight_mean = weight.mean().item()

    weight_std = weight.std().item()

    max_bias = torch.max(
        torch.abs(bias)
    ).item()

    initializer_range = (
        model.bert.config.initializer_range
    )

    print(
        "Expected initializer range:",
        initializer_range,
    )

    print(
        "MLP weight mean:",
        weight_mean,
    )

    print(
        "MLP weight std:",
        weight_std,
    )

    print(
        "Maximum absolute bias:",
        max_bias,
    )

    assert abs(
        weight_std - initializer_range
    ) < 0.002

    assert max_bias == 0.0

    print(
        "\nMLP initialization test: PASSED"
    )


def main():
    print("=" * 60)
    print("SIMCSE MODEL TESTS")
    print("=" * 60)

    test_shapes()

    test_training_dropout()

    test_evaluation_determinism()

    test_mlp_only_train()

    test_mlp_initialization()

    print("\n" + "=" * 60)
    print("ALL MODEL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()