import torch
import torch.nn as nn

from pooler import SimCSEPooler


def build_fake_inputs():
    torch.manual_seed(42)

    batch_size = 2
    sequence_length = 5
    hidden_size = 8

    last_hidden_state = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
    )

    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    # Simulate:
    #
    # embeddings
    # layer 1
    # layer 2
    # layer 3
    #
    all_hidden_states = (
        torch.randn(
            batch_size,
            sequence_length,
            hidden_size,
        )
        for _ in range(4)
    )

    all_hidden_states = tuple(
        all_hidden_states
    )

    return (
        last_hidden_state,
        attention_mask,
        all_hidden_states,
    )


def test_cls_before_pooler():
    print("=" * 60)
    print("TEST 1: CLS_BEFORE_POOLER")
    print("=" * 60)

    (
        hidden,
        mask,
        all_hidden,
    ) = build_fake_inputs()

    pooler = SimCSEPooler(
        pooler_type="cls_before_pooler"
    )

    output = pooler(
        hidden,
        mask,
        all_hidden,
    )

    expected = hidden[:, 0]

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Maximum difference:",
        torch.max(
            torch.abs(
                output - expected
            )
        ).item(),
    )

    assert torch.equal(
        output,
        expected,
    )

    print(
        "\nCLS-before-pooler test: PASSED"
    )


def test_cls_with_mlp():
    print("\n" + "=" * 60)
    print("TEST 2: CLS + MLP")
    print("=" * 60)

    (
        hidden,
        mask,
        all_hidden,
    ) = build_fake_inputs()

    mlp = nn.Sequential(
        nn.Linear(8, 8),
        nn.Tanh(),
    )

    pooler = SimCSEPooler(
        pooler_type="cls",
    )

    output = pooler(
        hidden,
        mask,
        all_hidden,
        mlp=mlp,
    )

    expected = mlp(
        hidden[:, 0]
    )

    print(
        "Output shape:",
        output.shape,
    )

    assert torch.equal(
        output,
        expected,
    )

    print(
        "\nCLS + MLP test: PASSED"
    )

def test_avg_pooling():
    print("\n" + "=" * 60)
    print("TEST 3: AVG POOLING")
    print("=" * 60)

    (
        hidden,
        mask,
        all_hidden,
    ) = build_fake_inputs()

    pooler = SimCSEPooler(
        pooler_type="avg"
    )

    output = pooler(
        hidden,
        mask,
        all_hidden,
    )

    expected = torch.stack(
        [
            hidden[0, :4].mean(dim=0),
            hidden[1, :3].mean(dim=0),
        ]
    )

    print(
        "Output shape:",
        output.shape,
    )

    print(
        "Maximum difference:",
        torch.max(
            torch.abs(
                output - expected
            )
        ).item(),
    )

    assert torch.allclose(
        output,
        expected,
    )

    print(
        "\nAverage pooling test: PASSED"
    )


def test_avg_top2():
    print("\n" + "=" * 60)
    print("TEST 4: AVG_TOP2")
    print("=" * 60)

    (
        hidden,
        mask,
        all_hidden,
    ) = build_fake_inputs()

    pooler = SimCSEPooler(
        pooler_type="avg_top2"
    )

    output = pooler(
        hidden,
        mask,
        all_hidden,
    )

    combined = (
        all_hidden[-1]
        + all_hidden[-2]
    ) / 2.0

    expected = (
        combined
        * mask.unsqueeze(-1)
    ).sum(dim=1) / mask.sum(
        dim=1,
        keepdim=True,
    )

    print(
        "Output shape:",
        output.shape,
    )

    assert torch.allclose(
        output,
        expected,
    )

    print(
        "\nAvg-top2 test: PASSED"
    )


def test_avg_first_last():
    print("\n" + "=" * 60)
    print("TEST 5: AVG_FIRST_LAST")
    print("=" * 60)

    (
        hidden,
        mask,
        all_hidden,
    ) = build_fake_inputs()

    pooler = SimCSEPooler(
        pooler_type="avg_first_last"
    )

    output = pooler(
        hidden,
        mask,
        all_hidden,
    )

    combined = (
        all_hidden[1]
        + all_hidden[-1]
    ) / 2.0

    expected = (
        combined
        * mask.unsqueeze(-1)
    ).sum(dim=1) / mask.sum(
        dim=1,
        keepdim=True,
    )

    print(
        "Output shape:",
        output.shape,
    )

    assert torch.allclose(
        output,
        expected,
    )

    print(
        "\nAvg-first-last test: PASSED"
    )


def test_invalid_pooler():
    print("\n" + "=" * 60)
    print("TEST 6: INVALID POOLER")
    print("=" * 60)

    try:
        SimCSEPooler(
            pooler_type="invalid"
        )

    except ValueError:
        print(
            "Invalid pooler correctly rejected."
        )

    else:
        raise AssertionError(
            "Invalid pooler was not rejected."
        )

    print(
        "\nInvalid pooler test: PASSED"
    )


def main():
    print("=" * 60)
    print("SIMCSE POOLER TESTS")
    print("=" * 60)

    test_cls_before_pooler()

    test_cls_with_mlp()

    test_avg_pooling()

    test_avg_top2()

    test_avg_first_last()

    test_invalid_pooler()

    print("\n" + "=" * 60)
    print("ALL POOLER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()