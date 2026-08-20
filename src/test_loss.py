import torch
import torch.nn.functional as F

from loss import SimCSELoss


def test_similarity_matrix():
    print("=" * 60)
    print("TEST 1: SIMILARITY MATRIX")
    print("=" * 60)

    torch.manual_seed(42)

    z1 = torch.randn(
        4,
        768,
    )

    z2 = torch.randn(
        4,
        768,
    )

    criterion = SimCSELoss(
        temperature=0.05,
    )

    similarity = criterion.similarity_matrix(
        z1,
        z2,
    )

    print(
        "z1 shape:",
        z1.shape,
    )

    print(
        "z2 shape:",
        z2.shape,
    )

    print(
        "similarity matrix shape:",
        similarity.shape,
    )

    print(
        "similarity matrix:"
    )

    print(similarity)

    assert similarity.shape == (
        4,
        4,
    )

    print(
        "\nSimilarity matrix test: PASSED"
    )


def test_diagonal_labels():
    print("\n" + "=" * 60)
    print("TEST 2: DIAGONAL LABELS")
    print("=" * 60)

    criterion = SimCSELoss(
        temperature=0.05,
    )

    labels = criterion.labels(
        batch_size=4,
        device=torch.device("cpu"),
    )

    print(
        "Labels:",
        labels,
    )

    expected = torch.tensor(
        [0, 1, 2, 3]
    )

    assert torch.equal(
        labels,
        expected,
    )

    print(
        "\nDiagonal label test: PASSED"
    )


def test_loss_calculation():
    print("\n" + "=" * 60)
    print("TEST 3: LOSS CALCULATION")
    print("=" * 60)

    torch.manual_seed(42)

    z1 = torch.randn(
        4,
        768,
        requires_grad=True,
    )

    z2 = torch.randn(
        4,
        768,
        requires_grad=True,
    )

    criterion = SimCSELoss(
        temperature=0.05,
    )

    loss = criterion(
        z1,
        z2,
    )

    print(
        "Loss:",
        loss.item(),
    )

    assert torch.isfinite(loss)

    assert loss.item() > 0

    print(
        "\nLoss calculation test: PASSED"
    )


def test_gradients():
    print("\n" + "=" * 60)
    print("TEST 4: GRADIENTS")
    print("=" * 60)

    torch.manual_seed(42)

    z1 = torch.randn(
        4,
        768,
        requires_grad=True,
    )

    z2 = torch.randn(
        4,
        768,
        requires_grad=True,
    )

    criterion = SimCSELoss(
        temperature=0.05,
    )

    loss = criterion(
        z1,
        z2,
    )

    loss.backward()

    print(
        "z1 gradient exists:",
        z1.grad is not None,
    )

    print(
        "z2 gradient exists:",
        z2.grad is not None,
    )

    print(
        "z1 gradient norm:",
        z1.grad.norm().item(),
    )

    print(
        "z2 gradient norm:",
        z2.grad.norm().item(),
    )

    assert z1.grad is not None

    assert z2.grad is not None

    assert (
        z1.grad.norm().item()
        > 0
    )

    assert (
        z2.grad.norm().item()
        > 0
    )

    print(
        "\nGradient test: PASSED"
    )


def test_positive_pairs():
    print("\n" + "=" * 60)
    print("TEST 5: POSITIVE PAIR STRUCTURE")
    print("=" * 60)

    torch.manual_seed(42)

    # --------------------------------------------------------
    # Strongly aligned positive pairs.
    #
    # z2 is exactly z1.
    # --------------------------------------------------------

    z1 = F.normalize(
        torch.randn(
            4,
            768,
        ),
        dim=1,
    )

    z2_aligned = z1.clone()

    criterion = SimCSELoss(
        temperature=0.05,
    )

    aligned_loss = criterion(
        z1,
        z2_aligned,
    )

    # --------------------------------------------------------
    # Random positive pair assignments.
    # --------------------------------------------------------

    z2_random = z2_aligned[
        torch.tensor(
            [1, 2, 3, 0]
        )
    ]

    random_loss = criterion(
        z1,
        z2_random,
    )

    print(
        "Loss with strongly aligned positive pairs:",
        aligned_loss.item(),
    )

    print(
        "Loss with random pairs:",
        random_loss.item(),
    )

    assert (
        aligned_loss
        < random_loss
    )

    print(
        "\nPositive-pair structure test: PASSED"
    )


def test_diagnostics():
    print("\n" + "=" * 60)
    print("TEST 6: DIAGNOSTICS")
    print("=" * 60)

    torch.manual_seed(42)

    z1 = torch.randn(
        4,
        768,
    )

    z2 = torch.randn(
        4,
        768,
    )

    criterion = SimCSELoss(
        temperature=0.05,
    )

    diagnostics = criterion.diagnostics(
        z1,
        z2,
    )

    print(
        "Loss:",
        diagnostics["loss"].item(),
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

    assert torch.isfinite(
        diagnostics["loss"]
    )

    assert torch.isfinite(
        diagnostics[
            "positive_similarity"
        ]
    )

    assert torch.isfinite(
        diagnostics[
            "negative_similarity"
        ]
    )

    print(
        "\nDiagnostics test: PASSED"
    )


def main():
    print("=" * 60)
    print("SIMCSE LOSS TESTS")
    print("=" * 60)

    test_similarity_matrix()

    test_diagonal_labels()

    test_loss_calculation()

    test_gradients()

    test_positive_pairs()

    test_diagnostics()

    print("\n" + "=" * 60)
    print("ALL LOSS TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()