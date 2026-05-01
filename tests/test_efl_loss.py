"""Tests for src/efl_loss.py.

Covers:
  - At equal class frequencies (g_j = 0.5 for all classes), EFL applies
    a uniform per-class gamma (= gamma + 0.5 * gamma_b). It does NOT
    reduce to standard focal loss unless gamma_b == 0; verify the
    reduction-to-focal-loss claim only for that gamma_b == 0 case.
  - Rare classes (low g_j) get a HIGHER effective gamma and a HIGHER
    per-example loss for misclassified positives.
  - EMA updates: the first forward pass initializes EMA; subsequent
    passes update with momentum.
  - Reduction modes (mean/sum/none) work and have the expected shapes.
  - Backward pass produces non-NaN gradients.

Run from repo root:
    pytest tests/test_efl_loss.py -v
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from src.efl_loss import EqualizedFocalLoss


def _focal_loss_reference(
    logits: torch.Tensor, targets: torch.Tensor, *, gamma: float, alpha: float | None = None
) -> torch.Tensor:
    """Bog-standard focal loss (binary, sigmoid head), no per-class gamma."""
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = targets * p + (1 - targets) * (1 - p)
    focal_w = (1 - p_t).clamp_min(1e-7).pow(gamma)
    if alpha is not None:
        alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
        focal_w = focal_w * alpha_t
    return focal_w * ce


def test_reduces_to_standard_focal_when_gamma_b_zero():
    """If gamma_b == 0, per-class gamma is constant (= gamma) regardless of g_j,
    so EFL must produce IDENTICAL output to standard focal loss."""
    torch.manual_seed(0)
    nc = 5
    n = 64
    logits = torch.randn(n, nc) * 2
    targets = (torch.rand(n, nc) > 0.7).float()

    efl = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=0.0, reduction="none")
    efl.set_grad_ratios(torch.full((nc,), 0.3))  # arbitrary, should not matter
    out_efl = efl(logits, targets, update_ema=False)
    out_ref = _focal_loss_reference(logits, targets, gamma=2.0)

    torch.testing.assert_close(out_efl, out_ref, rtol=1e-5, atol=1e-6)


def test_per_class_gamma_formula():
    """Per-class gamma must equal gamma + (1 - g_j) * gamma_b."""
    nc = 4
    efl = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=4.0)
    g = torch.tensor([0.0, 0.25, 0.5, 1.0])
    efl.set_grad_ratios(g)
    expected = torch.tensor([6.0, 5.0, 4.0, 2.0])
    torch.testing.assert_close(efl.per_class_gamma(), expected)


def test_rare_class_gets_higher_loss_than_frequent_class():
    """For the SAME logit/target pair, the rare class (low g_j -> high gamma)
    should yield a STRICTLY LOWER loss for an easy positive (high p) and
    a STRICTLY HIGHER loss for a hard positive (low p) than a frequent class."""
    nc = 2
    efl = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=4.0, reduction="none")
    efl.set_grad_ratios(torch.tensor([0.0, 1.0]))

    logit_easy_pos = torch.tensor([[5.0, 5.0]])
    target_pos = torch.tensor([[1.0, 1.0]])
    out = efl(logit_easy_pos, target_pos, update_ema=False)
    assert out[0, 0] < out[0, 1], (
        f"Rare-class easy positive should have lower loss (heavier focal "
        f"down-weighting): got rare={out[0,0].item():.4g}, freq={out[0,1].item():.4g}"
    )

    logit_hard_pos = torch.tensor([[-2.0, -2.0]])
    out_hard = efl(logit_hard_pos, target_pos, update_ema=False)
    assert out_hard[0, 0] > out_hard[0, 1], (
        f"Rare-class hard positive should have higher loss (less down-weighting "
        f"for hard examples is the EFL trade-off): got rare={out_hard[0,0].item():.4g}, "
        f"freq={out_hard[0,1].item():.4g}"
    )


def test_ema_initialization_and_update():
    """First forward pass must initialize EMA from raw counts; second
    forward pass must blend with `momentum`."""
    torch.manual_seed(42)
    nc = 3
    efl = EqualizedFocalLoss(num_classes=nc, momentum=0.9)

    assert not bool(efl._initialized.item())

    logits1 = torch.tensor([[2.0, -2.0, 0.0]])
    targets1 = torch.tensor([[1.0, 0.0, 1.0]])
    efl(logits1, targets1)
    assert bool(efl._initialized.item())
    pos1 = efl.pos_grad_ema.clone()
    neg1 = efl.neg_grad_ema.clone()

    logits2 = torch.tensor([[-2.0, 2.0, 0.0]])
    targets2 = torch.tensor([[0.0, 1.0, 0.0]])
    efl(logits2, targets2)

    p2 = torch.sigmoid(logits2)
    grad2 = (p2 - targets2).abs()
    pos_mask2 = targets2 > 0.5
    neg_mask2 = ~pos_mask2
    raw_pos2 = (grad2 * pos_mask2.float()).sum(dim=0)
    raw_neg2 = (grad2 * neg_mask2.float()).sum(dim=0)

    expected_pos = 0.9 * pos1 + 0.1 * raw_pos2
    expected_neg = 0.9 * neg1 + 0.1 * raw_neg2
    torch.testing.assert_close(efl.pos_grad_ema, expected_pos, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(efl.neg_grad_ema, expected_neg, rtol=1e-5, atol=1e-7)


def test_reduction_modes_have_correct_shapes():
    nc = 4
    n = 8
    logits = torch.randn(n, nc)
    targets = (torch.rand(n, nc) > 0.5).float()

    for red, expected_ndim in [("mean", 0), ("sum", 0), ("none", 2)]:
        efl = EqualizedFocalLoss(num_classes=nc, reduction=red)
        out = efl(logits, targets)
        assert out.dim() == expected_ndim, (
            f"reduction={red!r}: expected ndim={expected_ndim}, got {out.dim()}"
        )
        if red == "none":
            assert tuple(out.shape) == (n, nc)


def test_backward_pass_no_nans():
    """End-to-end gradient check: loss should be differentiable w.r.t. logits
    and produce non-NaN, non-Inf gradients."""
    torch.manual_seed(0)
    nc = 6
    n = 16
    logits = torch.randn(n, nc, requires_grad=True)
    targets = (torch.rand(n, nc) > 0.6).float()

    efl = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=4.0)
    loss = efl(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), "EFL backward produced non-finite gradients"
    assert (logits.grad.abs() > 0).any(), "EFL backward produced all-zero gradients"


def test_alpha_weighting_changes_loss():
    """When alpha != None, positives are scaled by alpha and negatives by (1-alpha).
    A run with alpha=0.5 should equal a run without alpha but scaled by 0.5."""
    torch.manual_seed(0)
    nc = 3
    n = 16
    logits = torch.randn(n, nc)
    targets = (torch.rand(n, nc) > 0.5).float()

    efl_no_alpha = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=0.0, alpha=None, reduction="none")
    efl_no_alpha.set_grad_ratios(torch.full((nc,), 0.5))
    efl_alpha_half = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=0.0, alpha=0.5, reduction="none")
    efl_alpha_half.set_grad_ratios(torch.full((nc,), 0.5))

    out_no_alpha = efl_no_alpha(logits, targets, update_ema=False)
    out_alpha_half = efl_alpha_half(logits, targets, update_ema=False)
    torch.testing.assert_close(out_alpha_half, 0.5 * out_no_alpha, rtol=1e-5, atol=1e-7)


def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        EqualizedFocalLoss(num_classes=0)
    with pytest.raises(ValueError):
        EqualizedFocalLoss(num_classes=4, reduction="bogus")
    with pytest.raises(ValueError):
        EqualizedFocalLoss(num_classes=4, momentum=1.5)


def test_shape_validation():
    nc = 4
    efl = EqualizedFocalLoss(num_classes=nc)
    logits = torch.randn(8, nc)
    targets_wrong = torch.zeros(8, nc + 1)
    with pytest.raises(ValueError):
        efl(logits, targets_wrong)


def test_extreme_imbalance_realistic_fathomnet_scale():
    """Smoke test at FathomNet's actual imbalance scale.

    Sets up a synthetic batch with 32 classes and gradient ratios mimicking
    the actual FathomNet train counts (urchin dominates, sea slug rare),
    then checks that effective gamma for sea slug class > effective gamma
    for urchin class by a meaningful margin.
    """
    nc = 32
    efl = EqualizedFocalLoss(num_classes=nc, gamma=2.0, gamma_b=4.0)

    g = torch.full((nc,), 0.5)
    g[31] = 0.95  # urchin (most frequent in real data)
    g[22] = 0.05  # sea slug (rarest, 7 instances)
    efl.set_grad_ratios(g)

    eff = efl.per_class_gamma()
    assert eff[22] > eff[31] + 3.0, (
        f"Sea slug effective gamma should be much higher than urchin's: "
        f"sea_slug={eff[22].item():.2f}, urchin={eff[31].item():.2f}"
    )
    assert eff[22].item() > 5.0
    assert eff[31].item() < 3.0
