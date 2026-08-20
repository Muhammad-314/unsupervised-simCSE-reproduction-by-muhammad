import math

import torch

from alignment_uniformity import alignment_loss, normalize_embeddings, uniformity_loss


def test_normalize_embeddings_unit_norm():
    x = torch.tensor([[3.0, 4.0], [5.0, 12.0]])
    z = normalize_embeddings(x)
    norms = torch.linalg.vector_norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms))


def test_alignment_identical_pairs_is_zero():
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert alignment_loss(x, x) == 0.0


def test_alignment_opposite_unit_vectors_is_four():
    x = torch.tensor([[1.0, 0.0]])
    y = torch.tensor([[-1.0, 0.0]])
    assert math.isclose(alignment_loss(x, y), 4.0, rel_tol=0.0, abs_tol=1e-7)


def test_uniformity_matches_manual_two_point_case():
    # Two orthogonal unit vectors have squared distance 2.
    # Uniformity = log(exp(-2 * 2)) = -4.
    x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert math.isclose(uniformity_loss(x), -4.0, rel_tol=0.0, abs_tol=1e-7)


def test_uniformity_is_invariant_to_positive_scaling():
    x = torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]])
    scaled = x * 7.5
    assert math.isclose(
        uniformity_loss(x),
        uniformity_loss(scaled),
        rel_tol=0.0,
        abs_tol=1e-7,
    )
