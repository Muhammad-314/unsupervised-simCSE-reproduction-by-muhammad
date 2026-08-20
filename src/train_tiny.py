import random

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import WikipediaSentenceDataset, SimCSECollator
from model import SimCSEModel
from loss import SimCSELoss


# ============================================================
# Experiment configuration
# ============================================================

MODEL_NAME = "bert-base-uncased"

DATA_PATH = "data/raw/wiki1m_for_simcse.txt"

MAX_SENTENCES = 256

BATCH_SIZE = 16

MAX_LENGTH = 32

LEARNING_RATE = 3e-5

TEMPERATURE = 0.05

EPOCHS = 3

MAX_GRAD_NORM = 1.0

SEED = 42


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    """
    Set random seeds for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================

def calculate_similarity_metrics(
    similarity_matrix: torch.Tensor,
):
    """
    Calculate:

        positive similarity:
            mean of diagonal entries

        negative similarity:
            mean of all off-diagonal entries
    """

    batch_size = similarity_matrix.shape[0]

    diagonal = torch.diagonal(
        similarity_matrix,
    )

    positive_similarity = diagonal.mean()

    mask = ~torch.eye(
        batch_size,
        dtype=torch.bool,
        device=similarity_matrix.device,
    )

    negative_similarity = similarity_matrix[
        mask
    ].mean()

    return (
        positive_similarity.item(),
        negative_similarity.item(),
    )


# ============================================================
# Main training experiment
# ============================================================

def main():

    print("=" * 70)
    print("DAY 8: TINY END-TO-END SIMCSE TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. Set seed
    # --------------------------------------------------------

    set_seed(SEED)

    # --------------------------------------------------------
    # 2. Select device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:")
    print(device)

    if torch.cuda.is_available():
        print("GPU:")
        print(torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # 3. Print experiment configuration
    # --------------------------------------------------------

    print("\nExperiment configuration:")
    print(f"Model:           {MODEL_NAME}")
    print(f"Dataset:         {DATA_PATH}")
    print(f"Sentences:       {MAX_SENTENCES}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Max length:      {MAX_LENGTH}")
    print(f"Learning rate:   {LEARNING_RATE}")
    print(f"Temperature:     {TEMPERATURE}")
    print(f"Epochs:          {EPOCHS}")
    print(f"Max grad norm:   {MAX_GRAD_NORM}")
    print(f"Seed:            {SEED}")

    # --------------------------------------------------------
    # 4. Load tokenizer
    # --------------------------------------------------------

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # 5. Load a tiny subset of the real dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = WikipediaSentenceDataset(
        file_path=DATA_PATH,
        max_sentences=MAX_SENTENCES,
    )

    print(
        f"Loaded {len(dataset)} sentences."
    )

    # --------------------------------------------------------
    # 6. Create collator
    # --------------------------------------------------------

    collator = SimCSECollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    # --------------------------------------------------------
    # 7. Create DataLoader
    # --------------------------------------------------------

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
    )

    print(
        f"Number of batches per epoch: {len(loader)}"
    )

    # --------------------------------------------------------
    # 8. Create model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = SimCSEModel(
        model_name=MODEL_NAME,
        mlp_only_train=True,
    )

    model.to(device)

    # --------------------------------------------------------
    # 9. Create loss
    # --------------------------------------------------------

    loss_fn = SimCSELoss(
        temperature=TEMPERATURE,
    )

    # --------------------------------------------------------
    # 10. Create optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # --------------------------------------------------------
    # 11. Training loop
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    for epoch in range(EPOCHS):

        model.train()

        epoch_loss = 0.0
        epoch_positive_similarity = 0.0
        epoch_negative_similarity = 0.0

        for step, batch in enumerate(
            loader,
            start=1,
        ):

            # ------------------------------------------------
            # Move batch to device
            # ------------------------------------------------

            input_ids = batch[
                "input_ids"
            ].to(device)

            attention_mask = batch[
                "attention_mask"
            ].to(device)

            # ------------------------------------------------
            # Clear previous gradients
            # ------------------------------------------------

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # First stochastic forward pass
            # ------------------------------------------------

            z1 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ------------------------------------------------
            # Second stochastic forward pass
            # ------------------------------------------------

            z2 = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            # ------------------------------------------------
            # Calculate loss
            # ------------------------------------------------

            loss = loss_fn(
                z1,
                z2,
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                MAX_GRAD_NORM,
            )

            # ------------------------------------------------
            # Update parameters
            # ------------------------------------------------

            optimizer.step()

            # ------------------------------------------------
            # Calculate similarity metrics
            #
            # We calculate this BEFORE temperature scaling.
            # Therefore values are cosine similarities in
            # approximately [-1, 1].
            # ------------------------------------------------

            with torch.no_grad():

                similarity_matrix = (
                    loss_fn.similarity_matrix(
                        z1,
                        z2,
                    )
                )

                (
                    positive_similarity,
                    negative_similarity,
                ) = calculate_similarity_metrics(
                    similarity_matrix
                )

            # ------------------------------------------------
            # Accumulate epoch statistics
            # ------------------------------------------------

            epoch_loss += loss.item()

            epoch_positive_similarity += (
                positive_similarity
            )

            epoch_negative_similarity += (
                negative_similarity
            )

            # ------------------------------------------------
            # Print every step
            # ------------------------------------------------

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Step {step:02d}/{len(loader)} "
                f"| Loss {loss.item():.4f} "
                f"| PosSim {positive_similarity:.4f} "
                f"| NegSim {negative_similarity:.4f}"
            )

        # ----------------------------------------------------
        # Epoch averages
        # ----------------------------------------------------

        num_steps = len(loader)

        average_loss = (
            epoch_loss / num_steps
        )

        average_positive_similarity = (
            epoch_positive_similarity
            / num_steps
        )

        average_negative_similarity = (
            epoch_negative_similarity
            / num_steps
        )

        print("\n" + "-" * 70)
        print(f"Epoch {epoch + 1} summary")
        print("-" * 70)

        print(
            f"Average loss:             "
            f"{average_loss:.6f}"
        )

        print(
            f"Average positive cosine:  "
            f"{average_positive_similarity:.6f}"
        )

        print(
            f"Average negative cosine:  "
            f"{average_negative_similarity:.6f}"
        )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TINY TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()