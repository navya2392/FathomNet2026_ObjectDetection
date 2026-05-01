"""Hierarchical (taxonomy-aware) classification loss for Phase 5 EXP 5.1.

Block reference: G.2 in master_checklist.txt.

What problem does hierarchical loss solve
-----------------------------------------
Phase 0 confusion-matrix EDA showed that many of YOLO's classification
errors are within-phylum (e.g., predicting "sea fan" instead of "sea pen"
-- both cnidarians). The standard cross-entropy loss treats these
within-phylum errors with the SAME magnitude as cross-phylum errors
(e.g., predicting "fish" instead of "amphipod"). For mAP@[.50:.95]
this matters because:

- Within-phylum confusions tend to be visually plausible mistakes (the
  model "sees" a marine invertebrate but picks the wrong species). The
  bbox is usually right, just the class label is wrong.
- Cross-phylum confusions are gross errors (the model fundamentally
  misjudged what KIND of organism it's looking at).

Penalizing both equally with cross-entropy gives no signal to the model
about "you got the kind right, just refine the species." The hierarchical
loss adds an extra term that explicitly rewards getting the GROUP right
even when the species is wrong, encouraging the model to converge to a
sensible group FIRST and then refine species.

The math
--------
Standard cross-entropy on 32 species:

    L_species(logits, target_class) = CE(softmax(logits), target_class)

Hierarchical loss adds a group-level term:

    p_group_g = sum over classes c in g of p_class_c
              (where p_class = softmax(logits))
    L_group(logits, target_group) = CE(p_group, target_group)

    L_hier = L_species + group_weight * L_group

where target_group is the group containing target_class.

The aggregation `p_group_g = sum(p_c for c in g)` is correct because if
the species-level distribution is properly normalized, the group-level
distribution is also properly normalized (the groups partition the
classes).

Status (May 1, 2026)
--------------------
SKELETON + TESTS. NOT yet integrated with Ultralytics' detection loss.
The integration plan: subclass v8DetectionLoss, replace its per-class
BCE classification head with this hierarchical loss. Or: keep the
species-level BCE intact and ADD the group-level term as an auxiliary
loss with its own coefficient (simpler, less risk of regressing
species-level accuracy).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def build_class_to_group_tensor(
    groups: dict[str, list[int]],
    num_classes: int,
) -> tuple[torch.Tensor, list[str]]:
    """Convert a {group_name: [class_idx, ...]} dict into a tensor mapping.

    Parameters
    ----------
    groups : dict[str, list[int]]
        From configs.taxonomy.GROUPS.
    num_classes : int
        Total number of classes (32 for FathomNet 2026).

    Returns
    -------
    class_to_group : torch.LongTensor of shape (num_classes,)
        class_to_group[c] = group_idx (0-indexed). Order of group_idx
        matches the second return value.
    group_names : list[str]
        Group names in the order matching class_to_group's group_idx.

    Raises
    ------
    ValueError
        If groups don't partition [0, num_classes) (i.e., a class is
        in zero or multiple groups).
    """
    group_names = sorted(groups.keys())
    name_to_idx = {n: i for i, n in enumerate(group_names)}

    out = torch.full((num_classes,), -1, dtype=torch.long)
    for name, members in groups.items():
        for c in members:
            if c < 0 or c >= num_classes:
                raise ValueError(f"Group {name!r} contains out-of-range class_idx {c}")
            if out[c] != -1:
                raise ValueError(
                    f"Class {c} appears in both groups {group_names[out[c]]!r} and {name!r}"
                )
            out[c] = name_to_idx[name]

    if (out == -1).any():
        missing = [int(i) for i in (out == -1).nonzero(as_tuple=True)[0]]
        raise ValueError(f"Classes not assigned to any group: {missing}")

    return out, group_names


class HierarchicalLoss(nn.Module):
    """Hierarchical (species + group) classification loss.

    Use as a drop-in replacement for `nn.CrossEntropyLoss` when the
    classes have a known coarser-level grouping (taxonomy). Combines:

      L_total = L_species + group_weight * L_group

    where L_species is standard CE on the per-class logits and L_group
    is CE on the GROUP probabilities (computed by summing per-class
    probabilities within each group).

    Parameters
    ----------
    num_classes : int
    class_to_group : torch.LongTensor of shape (num_classes,)
        Build via `build_class_to_group_tensor()`.
    num_groups : int
        Total number of groups (= max(class_to_group) + 1 typically).
    group_weight : float
        Coefficient on the group-level loss. Default 0.3 (paper-style
        starting point; ablate in Phase 5 EXP 5.1).
    species_weight : float
        Coefficient on the species-level loss. Default 1.0 (kept fixed;
        ablate group_weight relative to this).
    reduction : str
        'mean', 'sum', or 'none'.
    eps : float
        Numerical stability for log().

    Notes
    -----
    Targets are CLASS indices (0 to num_classes-1), NOT group indices.
    The class -> group mapping is applied internally.

    Aggregation: when reduction='none', returns L_species + L_group with
    matching reduction; the per-sample decomposition is well-defined
    because both terms are per-sample.
    """

    def __init__(
        self,
        num_classes: int,
        class_to_group: torch.Tensor,
        num_groups: int,
        *,
        group_weight: float = 0.3,
        species_weight: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        if num_classes != class_to_group.numel():
            raise ValueError(
                f"num_classes ({num_classes}) must equal class_to_group.numel() "
                f"({class_to_group.numel()})"
            )
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'/'sum'/'none', got {reduction!r}")

        self.num_classes = num_classes
        self.num_groups = num_groups
        self.group_weight = float(group_weight)
        self.species_weight = float(species_weight)
        self.reduction = reduction
        self.eps = float(eps)

        self.register_buffer("class_to_group", class_to_group.long(), persistent=True)

    def forward(self, logits: torch.Tensor, target_class: torch.Tensor) -> torch.Tensor:
        """Compute the hierarchical loss.

        Parameters
        ----------
        logits :
            Shape (N, num_classes). Raw logits, NOT softmaxed.
        target_class :
            Shape (N,). Integer class indices in [0, num_classes).

        Returns
        -------
        torch.Tensor
            Scalar (reduction in mean/sum) or (N,)-shape tensor (none).
        """
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                f"logits must be (N, {self.num_classes}), got shape {tuple(logits.shape)}"
            )
        if target_class.ndim != 1 or target_class.numel() != logits.shape[0]:
            raise ValueError(
                f"target_class must be 1-D with {logits.shape[0]} elements, "
                f"got shape {tuple(target_class.shape)}"
            )

        l_species = F.cross_entropy(logits, target_class, reduction="none")

        probs = F.softmax(logits, dim=-1)
        group_probs = torch.zeros(
            logits.shape[0], self.num_groups, device=logits.device, dtype=probs.dtype
        )
        group_probs = group_probs.index_add(1, self.class_to_group, probs)
        group_probs = group_probs.clamp_min(self.eps)

        target_group = self.class_to_group[target_class]
        l_group = F.nll_loss(torch.log(group_probs), target_group, reduction="none")

        per_sample = self.species_weight * l_species + self.group_weight * l_group

        if self.reduction == "mean":
            return per_sample.mean()
        if self.reduction == "sum":
            return per_sample.sum()
        return per_sample

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, num_groups={self.num_groups}, "
            f"group_weight={self.group_weight}, species_weight={self.species_weight}, "
            f"reduction={self.reduction!r}"
        )


__all__ = ["HierarchicalLoss", "build_class_to_group_tensor"]
