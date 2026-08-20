import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from data import (
    SimCSECollator,
    WikipediaSentenceDataset,
)


MODEL_NAME = "bert-base-uncased"

TINY_DATA_PATH = (
    "data/tiny_wiki.txt"
)

FULL_DATA_PATH = (
    "data/raw/wiki1m_for_simcse.txt"
)


def test_tiny_dataset():
    print("=" * 60)
    print("TEST 1: TINY DATASET")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = WikipediaSentenceDataset(
        file_path=TINY_DATA_PATH,
        max_sentences=10,
    )

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=32,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print(
        "Number of sentences:",
        len(dataset),
    )

    print(
        "input_ids shape:",
        batch["input_ids"].shape,
    )

    print(
        "attention_mask shape:",
        batch["attention_mask"].shape,
    )

    print("\nFirst sentence:")
    print(repr(dataset[0]))

    print("\nFirst view:")
    print(batch["input_ids"][0, 0])

    print("\nSecond view:")
    print(batch["input_ids"][0, 1])

    # --------------------------------------------------------
    # Shape tests
    # --------------------------------------------------------

    assert batch["input_ids"].ndim == 3

    assert batch["attention_mask"].ndim == 3

    assert batch["input_ids"].shape[0] == 4

    assert batch["input_ids"].shape[1] == 2

    assert batch["attention_mask"].shape[0] == 4

    assert batch["attention_mask"].shape[1] == 2

    # --------------------------------------------------------
    # Identical-view tests
    # --------------------------------------------------------

    assert torch.equal(
        batch["input_ids"][:, 0],
        batch["input_ids"][:, 1],
    )

    assert torch.equal(
        batch["attention_mask"][:, 0],
        batch["attention_mask"][:, 1],
    )

    print(
        "\nIdentical token views: PASSED"
    )

    print(
        "Tiny dataset test: PASSED"
    )


def test_full_dataset():
    print("\n" + "=" * 60)
    print("TEST 2: FULL WIKIPEDIA DATASET")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    dataset = WikipediaSentenceDataset(
        file_path=FULL_DATA_PATH,
        max_sentences=None,
    )

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=32,
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        drop_last=True,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print(
        "Number of sentences:",
        len(dataset),
    )

    print("\nFirst sentence:")
    print(repr(dataset[0]))

    print("\nSecond sentence:")
    print(repr(dataset[1]))

    print("\nLast sentence:")
    print(repr(dataset[-1]))

    print("\nFull training batch:")

    print(
        "input_ids shape:",
        batch["input_ids"].shape,
    )

    print(
        "attention_mask shape:",
        batch["attention_mask"].shape,
    )

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    assert batch["input_ids"].shape[0] == 64

    assert batch["input_ids"].shape[1] == 2

    assert (
        batch["input_ids"].shape[2]
        <= 32
    )

    assert batch["attention_mask"].shape == (
        64,
        2,
        batch["attention_mask"].shape[2],
    )

    # --------------------------------------------------------
    # Identical views
    # --------------------------------------------------------

    assert torch.equal(
        batch["input_ids"][:, 0],
        batch["input_ids"][:, 1],
    )

    assert torch.equal(
        batch["attention_mask"][:, 0],
        batch["attention_mask"][:, 1],
    )

    print(
        "\nFull dataset batch test: PASSED"
    )


def test_truncation():
    print("\n" + "=" * 60)
    print("TEST 3: TRUNCATION")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=8,
    )

    sentence = (
        "This is a deliberately very long "
        "sentence that should be truncated."
    )

    batch = collator(
        [sentence]
    )

    print(
        "input_ids shape:",
        batch["input_ids"].shape,
    )

    print(
        "attention_mask shape:",
        batch["attention_mask"].shape,
    )

    print(
        "input_ids:",
        batch["input_ids"][0, 0],
    )

    # [batch, 2, seq]
    assert batch["input_ids"].shape == (
        1,
        2,
        8,
    )

    # Both views must still be identical.
    assert torch.equal(
        batch["input_ids"][0, 0],
        batch["input_ids"][0, 1],
    )

    print(
        "\nTruncation test: PASSED"
    )


def main():
    print("=" * 60)
    print("SIMCSE DATA PIPELINE TESTS")
    print("=" * 60)

    test_tiny_dataset()

    test_full_dataset()

    test_truncation()

    print("\n" + "=" * 60)
    print("ALL DATA PIPELINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()