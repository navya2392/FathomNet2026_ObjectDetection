"""Tests for src/hierarchical_loss.py.

Covers:
  - build_class_to_group_tensor() correctly partitions classes -> groups.
  - build_class_to_group_tensor() rejects overlapping/missing class assignments.
  - HierarchicalLoss reduces to species CE alone when group_weight=0.
  - HierarchicalLoss penalizes within-group errors LESS than cross-group
    errors of the same logit magnitude.
  - Reduction modes (mean/sum/none) have correct shapes.
  - Backward pass produces finite gradients.
  - Constructor and shape validation.

Run from repo root:
    pytest tests/test_hierarchical_loss.py -v
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.hierarchical_loss import HierarchicalLoss, build_class_to_group_tensor


def test_build_class_to_group_basic():
    groups = {"a": [0, 1], "b": [2, 3], "c": [4]}
    cls_to_grp, names = build_class_to_group_tensor(groups, num_classes=5)
    assert names == ["a", "b", "c"]
    assert cls_to_grp.tolist() == [0, 0, 1, 1, 2]


def test_build_class_to_group_missing_class():
    """Class 4 is unassigned -> ValueError."""
    groups = {"a": [0, 1], "b": [2, 3]}
    with pytest.raises(ValueError):
        build_class_to_group_tensor(groups, num_classes=5)


def test_build_class_to_group_overlapping():
    groups = {"a": [0, 1], "b": [1, 2]}
    with pytest.raises(ValueError):
        build_class_to_group_tensor(groups, num_classes=3)


def test_build_class_to_group_out_of_range():
    groups = {"a": [0, 1], "b": [2, 99]}
    with pytest.raises(ValueError):
        build_class_to_group_tensor(groups, num_classes=3)


def test_reduces_to_species_ce_when_group_weight_zero():
    torch.manual_seed(0)
    nc = 4
    groups = {"a": [0, 1], "b": [2, 3]}
    cls_to_grp, _ = build_class_to_group_tensor(groups, num_classes=nc)

    loss_fn = HierarchicalLoss(
        num_classes=nc, class_to_group=cls_to_grp, num_groups=2,
        group_weight=0.0, species_weight=1.0, reduction="none",
    )
    logits = torch.randn(8, nc)
    targets = torch.randint(0, nc, (8,))
    out = loss_fn(logits, targets)
    expected = F.cross_entropy(logits, targets, reduction="none")
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-7)


def test_within_group_error_costs_less_than_cross_group():
    """Setup: model is confident about class 1 but truth is class 0 (same group 'a').
    Compare to: confident about class 1 but truth is class 2 (group 'b').

    Both have the same SPECIES loss, but the second case ALSO incurs a
    group-loss (predicted group = 'a', truth group = 'b'). So total loss
    should be strictly higher in the cross-group case.
    """
    nc = 4
    groups = {"a": [0, 1], "b": [2, 3]}
    cls_to_grp, _ = build_class_to_group_tensor(groups, num_classes=nc)
    loss_fn = HierarchicalLoss(
        num_classes=nc, class_to_group=cls_to_grp, num_groups=2,
        group_weight=1.0, reduction="none",
    )

    logits = torch.tensor([
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
    ])
    targets_within = torch.tensor([0, 0])
    targets_cross = torch.tensor([2, 2])

    loss_within = loss_fn(logits, targets_within)
    loss_cross = loss_fn(logits, targets_cross)
    assert (loss_cross > loss_within).all(), (
        f"cross-group loss should exceed within-group: within={loss_within.tolist()}, "
        f"cross={loss_cross.tolist()}"
    )


def test_reduction_modes_shapes():
    nc = 4
    cls_to_grp, _ = build_class_to_group_tensor({"a": [0, 1], "b": [2, 3]}, num_classes=nc)
    logits = torch.randn(8, nc)
    targets = torch.randint(0, nc, (8,))
    for red, expected_ndim in [("mean", 0), ("sum", 0), ("none", 1)]:
        loss_fn = HierarchicalLoss(
            num_classes=nc, class_to_group=cls_to_grp, num_groups=2, reduction=red,
        )
        out = loss_fn(logits, targets)
        assert out.dim() == expected_ndim, f"reduction={red}: ndim={out.dim()}"
        if red == "none":
            assert tuple(out.shape) == (8,)


def test_backward_pass_no_nans():
    torch.manual_seed(0)
    nc = 6
    groups = {"a": [0, 1, 2], "b": [3, 4], "c": [5]}
    cls_to_grp, _ = build_class_to_group_tensor(groups, num_classes=nc)
    loss_fn = HierarchicalLoss(
        num_classes=nc, class_to_group=cls_to_grp, num_groups=3, group_weight=0.5,
    )
    logits = torch.randn(16, nc, requires_grad=True)
    targets = torch.randint(0, nc, (16,))
    loss = loss_fn(logits, targets)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_constructor_validation():
    cls_to_grp, _ = build_class_to_group_tensor({"a": [0, 1, 2, 3]}, num_classes=4)
    with pytest.raises(ValueError):
        HierarchicalLoss(num_classes=5, class_to_group=cls_to_grp, num_groups=1)
    with pytest.raises(ValueError):
        HierarchicalLoss(
            num_classes=4, class_to_group=cls_to_grp, num_groups=1, reduction="bogus",
        )


def test_shape_validation():
    cls_to_grp, _ = build_class_to_group_tensor({"a": [0, 1, 2, 3]}, num_classes=4)
    loss_fn = HierarchicalLoss(num_classes=4, class_to_group=cls_to_grp, num_groups=1)
    with pytest.raises(ValueError):
        loss_fn(torch.randn(8, 5), torch.zeros(8, dtype=torch.long))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(8, 4), torch.zeros(7, dtype=torch.long))
    with pytest.raises(ValueError):
        loss_fn(torch.randn(4), torch.zeros(1, dtype=torch.long))


def test_with_real_fathomnet_taxonomy():
    """Use the actual configs/taxonomy.py groups to ensure the math holds at FathomNet scale."""
    from configs.taxonomy import GROUPS
    cls_to_grp, names = build_class_to_group_tensor(GROUPS, num_classes=32)
    assert cls_to_grp.shape == (32,)
    assert len(names) == len(GROUPS)
    loss_fn = HierarchicalLoss(
        num_classes=32, class_to_group=cls_to_grp, num_groups=len(names),
        group_weight=0.3,
    )

    torch.manual_seed(0)
    logits = torch.randn(4, 32)
    targets = torch.tensor([22, 31, 6, 7])
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert loss.dim() == 0
