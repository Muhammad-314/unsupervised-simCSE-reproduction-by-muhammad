import torch
from transformers import AutoTokenizer

from data import WikipediaSentenceDataset, SimCSECollator
from model import SimCSEModel


MODEL_NAME = "bert-base-uncased"
DATA_PATH = "data/raw/wiki1m_for_simcse.txt"


def main():

    print("=" * 70)
    print("SIMCSE TRAINING INTERFACE TEST")
    print("=" * 70)

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
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    dataset = WikipediaSentenceDataset(
        file_path=DATA_PATH,
        max_sentences=8,
    )

    print(
        "Number of sentences:",
        len(dataset),
    )

    # --------------------------------------------------------
    # Collator
    # --------------------------------------------------------

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=32,
    )

    sentences = [
        dataset[i]
        for i in range(4)
    ]

    batch = collator(
        sentences
    )

    print(
        "\nBatch input_ids shape:"
    )
    print(
        batch["input_ids"].shape
    )

    print(
        "\nBatch attention_mask shape:"
    )
    print(
        batch["attention_mask"].shape
    )

    # --------------------------------------------------------
    # Verify two-view structure
    # --------------------------------------------------------

    assert batch["input_ids"].ndim == 3

    assert batch["input_ids"].shape[1] == 2

    assert batch[
        "attention_mask"
    ].ndim == 3

    assert batch[
        "attention_mask"
    ].shape[1] == 2

    print(
        "\nTwo-view batch structure: PASSED"
    )

    # --------------------------------------------------------
    # Verify token views are identical
    # --------------------------------------------------------

    identical_input_ids = torch.equal(
        batch["input_ids"][:, 0],
        batch["input_ids"][:, 1],
    )

    identical_attention_masks = torch.equal(
        batch["attention_mask"][:, 0],
        batch["attention_mask"][:, 1],
    )

    print(
        "Identical input views:",
        identical_input_ids,
    )

    print(
        "Identical attention-mask views:",
        identical_attention_masks,
    )

    assert identical_input_ids

    assert identical_attention_masks

    print(
        "\nIdentical token views: PASSED"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nLoading model..."
    )

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.to(device)

    model.train()

    input_ids = batch[
        "input_ids"
    ].to(device)

    attention_mask = batch[
        "attention_mask"
    ].to(device)

    # --------------------------------------------------------
    # ONE model call
    # --------------------------------------------------------

    print(
        "\nRunning ONE model forward call..."
    )

    z1, z2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    print(
        "z1 shape:",
        z1.shape,
    )

    print(
        "z2 shape:",
        z2.shape,
    )

    assert z1.shape == (
        4,
        768,
    )

    assert z2.shape == (
        4,
        768,
    )

    print(
        "\nSingle-call model interface: PASSED"
    )

    # --------------------------------------------------------
    # Verify stochastic dropout
    # --------------------------------------------------------

    print(
        "\nChecking stochastic representations..."
    )

    similarity = torch.nn.functional.cosine_similarity(
        z1,
        z2,
        dim=1,
    )

    print(
        "Mean positive cosine:",
        similarity.mean().item(),
    )

    print(
        "Maximum absolute difference:",
        torch.max(
            torch.abs(z1 - z2)
        ).item(),
    )

    # They should NOT be identical because BERT dropout
    # creates independent stochastic representations.

    assert not torch.equal(
        z1,
        z2,
    )

    print(
        "\nDropout stochasticity: PASSED"
    )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL TRAINING INTERFACE TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()