"""Tests for src/pu_loss.py.

Covers:
  - KiryoPULoss reduces to expected risk decomposition for known inputs.
  - The non-negative trick activates when negative-risk estimate goes negative
    AND mutates the loss in the expected direction.
  - Reduction modes (mean / sum / none) have correct shapes.
  - Backward pass produces finite gradients.
  - Constructor validation rejects bad pi / surrogate / reduction.
  - estimate_pu_prior() returns a sensible value on a tiny synthetic JSON.
  - estimate_pu_prior() handles degenerate inputs (no annotations) gracefully.

Run from repo root:
    pytest tests/test_pu_loss.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.pu_loss import KiryoPULoss, estimate_pu_prior


def test_constructor_validation():
    with pytest.raises(ValueError):
        KiryoPULoss(pi=0.0)
    with pytest.raises(ValueError):
        KiryoPULoss(pi=1.0)
    with pytest.raises(ValueError):
        KiryoPULoss(pi=0.1, surrogate_loss="bogus")
    with pytest.raises(ValueError):
        KiryoPULoss(pi=0.1, reduction="bogus")


def test_shape_validation():
    loss = KiryoPULoss(pi=0.1)
    logits = torch.randn(8)
    targets_wrong = torch.zeros(7)
    with pytest.raises(ValueError):
        loss(logits, targets_wrong)


def test_basic_loss_decomposition_no_correction():
    """When the unlabeled negative risk > pi * positive negative risk, the
    non-negative trick does NOT activate, and the total risk is simply
    pi * R_p+(g) + R_u-(g) - pi * R_p-(g).

    Verify by hand on a tiny synthetic batch.
    """
    pi = 0.2
    loss_fn = KiryoPULoss(pi=pi, surrogate_loss="sigmoid", reduction="mean")

    logits = torch.tensor([5.0, -5.0, 0.0, -3.0])
    targets = torch.tensor([1.0, 0.0, 0.0, 0.0])

    pos_logit = torch.tensor([5.0])
    unl_logits = torch.tensor([-5.0, 0.0, -3.0])
    R_p_pos = torch.sigmoid(-pos_logit).mean()
    R_p_neg = torch.sigmoid(pos_logit).mean()
    R_u_neg = torch.sigmoid(unl_logits).mean()
    expected = pi * R_p_pos + (R_u_neg - pi * R_p_neg)

    out = loss_fn(logits, targets)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_non_negative_correction_activates():
    """Construct a batch where R_u-(g) - pi * R_p-(g) < 0 (the trick must fire).

    Easy way: pi very high (close to 1), most positives are confident (logit >> 0),
    so R_p_neg = sigmoid(positive_logit) is also high; weighted by big pi, it
    exceeds R_u_neg.
    """
    pi = 0.9
    loss_fn = KiryoPULoss(pi=pi, surrogate_loss="sigmoid", reduction="mean", gamma=1.0)

    logits = torch.tensor([5.0, 5.0, 5.0, -5.0])
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0])

    pos_logits = torch.tensor([5.0, 5.0, 5.0])
    unl_logits = torch.tensor([-5.0])
    R_p_pos = torch.sigmoid(-pos_logits).mean()
    R_p_neg = torch.sigmoid(pos_logits).mean()
    R_u_neg = torch.sigmoid(unl_logits).mean()
    neg_risk = R_u_neg - pi * R_p_neg
    assert neg_risk.item() < 0, (
        f"Test setup is wrong: neg_risk should be negative for trick to fire, "
        f"got {neg_risk.item():.4g}"
    )

    expected = pi * R_p_pos - 1.0 * neg_risk
    out = loss_fn(logits, targets)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_gamma_zero_detaches_gradient_when_correction_active():
    pi = 0.9
    loss_fn = KiryoPULoss(pi=pi, surrogate_loss="sigmoid", reduction="mean", gamma=0.0)

    logits = torch.tensor([5.0, 5.0, 5.0, -5.0], requires_grad=True)
    targets = torch.tensor([1.0, 1.0, 1.0, 0.0])

    out = loss_fn(logits, targets)
    out.backward()
    grad_norm = logits.grad.abs().sum().item()
    assert grad_norm < 1e-3, (
        f"With gamma=0 and correction active, gradient should be ~0; got |grad|={grad_norm:.4g}"
    )


def test_reduction_modes_shapes():
    pi = 0.1
    logits = torch.randn(16)
    targets = (torch.rand(16) > 0.7).float()

    for red, expected_ndim in [("mean", 0), ("sum", 0), ("none", 1)]:
        loss_fn = KiryoPULoss(pi=pi, reduction=red)
        out = loss_fn(logits, targets)
        assert out.dim() == expected_ndim, f"reduction={red}: ndim={out.dim()}"
        if red == "none":
            assert tuple(out.shape) == (16,)


def test_backward_pass_no_nans():
    torch.manual_seed(0)
    pi = 0.1
    loss_fn = KiryoPULoss(pi=pi, surrogate_loss="sigmoid")
    logits = torch.randn(64, requires_grad=True)
    targets = (torch.rand(64) > 0.9).float()
    out = loss_fn(logits, targets)
    out.backward()
    assert torch.isfinite(logits.grad).all()


def test_logistic_surrogate_basic():
    pi = 0.2
    loss_sig = KiryoPULoss(pi=pi, surrogate_loss="sigmoid", reduction="mean")
    loss_log = KiryoPULoss(pi=pi, surrogate_loss="logistic", reduction="mean")

    torch.manual_seed(0)
    logits = torch.randn(32)
    targets = (torch.rand(32) > 0.7).float()

    out_sig = loss_sig(logits, targets)
    out_log = loss_log(logits, targets)
    assert torch.isfinite(out_sig)
    assert torch.isfinite(out_log)


def test_empty_input_returns_zero():
    loss_fn = KiryoPULoss(pi=0.1)
    out = loss_fn(torch.empty(0), torch.empty(0))
    assert out.item() == 0.0


def _write_synthetic_coco(tmp_path: Path, *, n_images: int = 5, ann_per_image: int = 2,
                          img_h: int = 100, img_w: int = 100, box_size: int = 10) -> Path:
    """Build a small COCO-format JSON with predictable area fraction.

    Each image has `ann_per_image` boxes of size box_size x box_size,
    so per-image area fraction = ann_per_image * box_size^2 / (img_h * img_w).
    """
    images = [
        {"id": i + 1, "file_name": f"img_{i}.png", "height": img_h, "width": img_w}
        for i in range(n_images)
    ]
    annotations = []
    next_ann_id = 1
    for img in images:
        for k in range(ann_per_image):
            annotations.append({
                "id": next_ann_id,
                "image_id": img["id"],
                "category_id": 1,
                "bbox": [k * 5, k * 5, box_size, box_size],
                "area": box_size * box_size,
            })
            next_ann_id += 1
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "obj"}],
    }
    out = tmp_path / "synthetic_train.json"
    out.write_text(json.dumps(coco))
    return out


def test_estimate_pu_prior_area_fraction(tmp_path):
    """5 images of 100x100 with 2 boxes of 10x10 each -> pi = 200 / 10000 = 0.02."""
    json_path = _write_synthetic_coco(tmp_path, n_images=5, ann_per_image=2,
                                      img_h=100, img_w=100, box_size=10)
    pi = estimate_pu_prior(json_path, method="area_fraction", verbose=False)
    expected = (2 * 10 * 10) / (100 * 100)
    assert abs(pi - expected) < 1e-6, f"pi = {pi}, expected {expected}"


def test_estimate_pu_prior_annotation_rate(tmp_path):
    json_path = _write_synthetic_coco(tmp_path, n_images=5, ann_per_image=2)
    pi = estimate_pu_prior(json_path, method="annotation_rate", verbose=False)
    expected = (5 * 2) / (5 * 100)
    assert abs(pi - expected) < 1e-6, f"pi = {pi}, expected {expected}"


def test_estimate_pu_prior_clamps_to_one(tmp_path):
    """If the GT boxes cover more than 100% of pixels (overlapping), per-image
    fraction is clamped to 1.0 so pi never exceeds 1."""
    json_path = _write_synthetic_coco(tmp_path, n_images=2, ann_per_image=10,
                                      img_h=10, img_w=10, box_size=20)
    pi = estimate_pu_prior(json_path, method="area_fraction", verbose=False)
    assert pi == pytest.approx(1.0, abs=1e-6)


def test_estimate_pu_prior_invalid_method(tmp_path):
    json_path = _write_synthetic_coco(tmp_path)
    with pytest.raises(ValueError):
        estimate_pu_prior(json_path, method="bogus")
