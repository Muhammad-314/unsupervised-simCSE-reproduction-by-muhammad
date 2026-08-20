import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import (
    SimCSECollator,
    WikipediaSentenceDataset,
)
from loss import SimCSELoss
from model import SimCSEModel


MODEL_NAME = "bert-base-uncased"

DATA_PATH = (
    "data/raw/wiki1m_for_simcse.txt"
)


def main():

    print("=" * 70)
    print("SIMCSE SINGLE TRAINING STEP TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    batch_size = 4
    max_length = 32
    learning_rate = 3e-5
    temperature = 0.05

    print("\nConfiguration:")
    print(
        "Model:",
        MODEL_NAME,
    )

    print(
        "Batch size:",
        batch_size,
    )

    print(
        "Max length:",
        max_length,
    )

    print(
        "Learning rate:",
        learning_rate,
    )

    print(
        "Temperature:",
        temperature,
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
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print(
        "\nLoading dataset..."
    )

    dataset = WikipediaSentenceDataset(
        file_path=DATA_PATH,
        max_sentences=16,
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
        max_length=max_length,
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collator,
    )

    batch = next(iter(loader))

    print(
        "\nBatch shapes:"
    )

    print(
        "input_ids:",
        batch["input_ids"].shape,
    )

    print(
        "attention_mask:",
        batch["attention_mask"].shape,
    )

    # --------------------------------------------------------
    # Verify two views are identical
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
        "Identical token views: PASSED"
    )

    # --------------------------------------------------------
    # Move batch to device
    # --------------------------------------------------------

    batch = {
        key: value.to(device)
        for key, value in batch.items()
    }

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

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    criterion = SimCSELoss(
        temperature=temperature,
    )

    criterion.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    print(
        "\nRunning forward pass..."
    )

    z1, z2 = model(
        input_ids=batch["input_ids"],
        attention_mask=batch[
            "attention_mask"
        ],
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
        batch_size,
        model.bert.config.hidden_size,
    )

    assert z2.shape == z1.shape

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    diagnostics = (
        criterion.diagnostics(
            z1,
            z2,
        )
    )

    loss = diagnostics["loss"]

    print(
        "\nLoss:",
        loss.item(),
    )

    print(
        "Positive cosine:",
        diagnostics[
            "positive_similarity"
        ].item(),
    )

    print(
        "Negative cosine:",
        diagnostics[
            "negative_similarity"
        ].item(),
    )

    assert torch.isfinite(loss)

    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    print(
        "\nRunning backward pass..."
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss.backward()

    # --------------------------------------------------------
    # Gradient verification
    # --------------------------------------------------------

    total_gradient_norm = 0.0

    parameters_with_gradients = 0

    for parameter in model.parameters():

        if parameter.grad is not None:

            parameters_with_gradients += 1

            total_gradient_norm += (
                parameter.grad.detach()
                .norm()
                .item()
                ** 2
            )

    total_gradient_norm = (
        total_gradient_norm ** 0.5
    )

    print(
        "Parameters with gradients:",
        parameters_with_gradients,
    )

    print(
        "Total gradient norm:",
        total_gradient_norm,
    )

    assert (
        parameters_with_gradients > 0
    )

    assert (
        total_gradient_norm > 0
    )

    # --------------------------------------------------------
    # Optimizer step
    # --------------------------------------------------------

    print(
        "\nRunning optimizer step..."
    )

    # Save one parameter before the step.
    parameter_before = (
        next(
            model.parameters()
        )
        .detach()
        .clone()
    )

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    parameter_after = (
        next(
            model.parameters()
        )
        .detach()
        .clone()
    )

    parameter_change = torch.max(
        torch.abs(
            parameter_after
            - parameter_before
        )
    ).item()

    print(
        "Maximum parameter change:",
        parameter_change,
    )

    assert parameter_change > 0

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        "SINGLE TRAINING STEP TEST: PASSED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()