import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_ablations import SimCSEModel


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

    return input_ids, attention_mask


def check_condition(
    name,
    dropout,
    fixed_dropout_mask,
    expected_identical,
):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    input_ids, attention_mask = build_inputs()

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
        dropout=dropout,
        fixed_dropout_mask=fixed_dropout_mask,
    )

    model.train()

    z1, z2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    difference = torch.max(
        torch.abs(z1 - z2)
    ).item()

    similarity = F.cosine_similarity(
        z1,
        z2,
        dim=1,
    ).mean().item()

    print("Dropout:", dropout)
    print(
        "Fixed dropout mask:",
        fixed_dropout_mask,
    )
    print(
        "Mean positive cosine:",
        similarity,
    )
    print(
        "Maximum absolute difference:",
        difference,
    )

    if expected_identical:
        assert difference == 0.0
    else:
        assert difference > 0.0

    print("PASSED")


def main():

    check_condition(
        name="NORMAL DROPOUT",
        dropout=0.1,
        fixed_dropout_mask=False,
        expected_identical=False,
    )

    check_condition(
        name="NO DROPOUT",
        dropout=0.0,
        fixed_dropout_mask=False,
        expected_identical=True,
    )

    check_condition(
        name="FIXED DROPOUT MASK",
        dropout=0.1,
        fixed_dropout_mask=True,
        expected_identical=True,
    )

    print("\n" + "=" * 70)
    print("ALL DROPOUT ABLATION TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()