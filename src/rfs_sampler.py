"""Repeat Factor Sampling (RFS) for the Phase 3 EXP 3.4 experiment.

Block reference: E.5 in master_checklist.txt.

Reference
---------
Gupta, Agrim et al. "LVIS: A Dataset for Large Vocabulary Instance Segmentation."
CVPR 2019, Section 3.5 "Repeat factor sampling."
https://arxiv.org/abs/1908.03195

What problem does RFS solve
---------------------------
With FathomNet's 817:1 class imbalance (urchin: 5,723 vs sea slug: 7),
naive uniform sampling means rare classes barely appear in any batch.
Standard fix is to oversample images containing rare classes; LVIS's
RFS does this in a principled per-category way:

For each category c with frequency f_c (images containing c / total images):
    r_c = max(1, sqrt(t / f_c))
where t is a threshold (oversample categories below f_c = t).

For each image i:
    r_i = max over categories c present in i of r_c

During training, image i is sampled r_i times per epoch (with stochastic
rounding when r_i is not integer, so the expected count matches r_i).

Effect: rare-class images get up-weighted, common-class images stay at
baseline. A single image containing both urchin (common) and sea slug
(rare) is upweighted to the sea slug factor, so we kill two birds.

The threshold t controls aggressiveness
---------------------------------------
LVIS paper uses t = 0.001 (1000 images = the threshold). For FathomNet
2026 with ~6,463 train images:

  t = 0.001 -> threshold image count = 6 (oversample categories with <= 6 images)
  t = 0.01  -> threshold image count = 65 (oversample categories with <= 65 images)
  t = 0.05  -> threshold image count = 323 (oversample categories with <= 323 images)
  t = 0.1   -> threshold image count = 646 (oversample categories with <= 646 images)

Phase 0 expectation: t ~ 0.05 - 0.1 covers the 14 classes with <100 instances.
Phase 3 EXP 3.4 ablates over t = {0.01, 0.05, 0.1, 0.2}.

Status (May 1, 2026)
--------------------
SKELETON + TESTS. The compute_repeat_factors() function is unit-tested.
Integration with Ultralytics' DataLoader is the Phase 3 EXP 3.4 work
(E.5) and lives in a follow-up PR.

Why ship the skeleton now
-------------------------
The threshold-vs-frequency math is subtle (sqrt vs linear vs log-linear);
LVIS's specific choice has been retroactively justified, but it's easy
to silently use the wrong formula. Having tests that confirm the algorithm
matches the paper means we know what we're integrating with Ultralytics.

This module ALSO writes a pkl for use by the Ultralytics DataLoader at
training time, OR exposes a torch.utils.data.Sampler that other DataLoaders
can use directly.
"""
from __future__ import annotations

import json
import math
import pickle
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@dataclass
class RFSResult:
    """Outputs of compute_repeat_factors().

    Attributes
    ----------
    threshold : float
        The t hyperparameter used.
    n_images : int
        Total number of images.
    category_image_counts : dict[int, int]
        For each category_id, the number of distinct images containing it.
    category_image_frequencies : dict[int, float]
        For each category_id, image_count / n_images.
    category_repeat_factors : dict[int, float]
        For each category_id, r_c = max(1, sqrt(t / f_c)).
    image_repeat_factors : dict[int, float]
        For each image_id, r_i = max over categories present in i of r_c.
    """

    threshold: float
    n_images: int
    category_image_counts: dict[int, int] = field(default_factory=dict)
    category_image_frequencies: dict[int, float] = field(default_factory=dict)
    category_repeat_factors: dict[int, float] = field(default_factory=dict)
    image_repeat_factors: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        rcs = sorted(self.category_repeat_factors.items(), key=lambda x: -x[1])
        lines = [
            f"RFS computed at threshold t = {self.threshold}",
            f"  Total images:    {self.n_images:,}",
            f"  Categories:      {len(self.category_repeat_factors)}",
            f"  Images repeated >1x: "
            f"{sum(1 for v in self.image_repeat_factors.values() if v > 1.0):,} "
            f"({100 * sum(1 for v in self.image_repeat_factors.values() if v > 1.0) / max(len(self.image_repeat_factors), 1):.1f}%)",
            f"  Effective epoch size: {sum(self.image_repeat_factors.values()):,.0f} "
            f"(was {self.n_images:,} without RFS)",
            "",
            "Top 5 categories by repeat factor:",
        ]
        for cid, rc in rcs[:5]:
            n = self.category_image_counts.get(cid, 0)
            f = self.category_image_frequencies.get(cid, 0)
            lines.append(f"  cat {cid:3d}: r_c={rc:.3f}  (n_images={n:,}, f_c={f:.4f})")
        return "\n".join(lines)


def compute_repeat_factors(
    train_json_path: str | Path,
    *,
    threshold: float = 0.05,
    verbose: bool = True,
) -> RFSResult:
    """Compute LVIS-style RFS repeat factors from a COCO-format JSON.

    Parameters
    ----------
    train_json_path :
        Path to the COCO-format train_dataset.json.
    threshold : float
        The t hyperparameter (paper: 0.001 for LVIS).
    verbose :
        If True, print a per-category and per-image summary.

    Returns
    -------
    RFSResult
        See class docstring for fields.

    Notes
    -----
    "Image contains category c" means there is at least one annotation
    of category c in that image. Multi-instance counts within an image
    don't multiply — RFS is per-image, not per-instance.
    """
    train_json_path = Path(train_json_path)
    if not train_json_path.exists():
        raise FileNotFoundError(f"train JSON not found: {train_json_path}")
    if threshold <= 0 or threshold > 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    with train_json_path.open() as f:
        coco = json.load(f)

    image_ids: set[int] = {int(img["id"]) for img in coco["images"]}
    n_images = len(image_ids)
    if n_images == 0:
        raise RuntimeError(f"No images in {train_json_path}")

    image_to_categories: dict[int, set[int]] = defaultdict(set)
    for ann in coco["annotations"]:
        iid = int(ann["image_id"])
        cid = int(ann["category_id"])
        image_to_categories[iid].add(cid)

    category_image_counts: dict[int, int] = defaultdict(int)
    for cats in image_to_categories.values():
        for cid in cats:
            category_image_counts[cid] += 1

    category_image_frequencies: dict[int, float] = {
        cid: cnt / n_images for cid, cnt in category_image_counts.items()
    }

    category_repeat_factors: dict[int, float] = {
        cid: max(1.0, math.sqrt(threshold / max(f, 1e-12)))
        for cid, f in category_image_frequencies.items()
    }

    image_repeat_factors: dict[int, float] = {}
    for iid in image_ids:
        cats = image_to_categories.get(iid, set())
        if not cats:
            image_repeat_factors[iid] = 1.0
            continue
        image_repeat_factors[iid] = max(category_repeat_factors[cid] for cid in cats)

    result = RFSResult(
        threshold=threshold,
        n_images=n_images,
        category_image_counts=dict(category_image_counts),
        category_image_frequencies=category_image_frequencies,
        category_repeat_factors=category_repeat_factors,
        image_repeat_factors=image_repeat_factors,
    )

    if verbose:
        print(result.summary())

    return result


def write_repeat_factors(result: RFSResult, out_path: str | Path) -> Path:
    """Persist an RFSResult to disk via pickle for reuse across training runs."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(result, f)
    return out_path.resolve()


def load_repeat_factors(in_path: str | Path) -> RFSResult:
    in_path = Path(in_path)
    with in_path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, RFSResult):
        raise TypeError(f"{in_path} did not contain an RFSResult; got {type(obj).__name__}")
    return obj


class RepeatFactorSampler(torch.utils.data.Sampler[int]):
    """A torch sampler that yields image indices with RFS-weighted multiplicity.

    Stochastic rounding: an image with r_i = 1.7 is yielded twice with
    probability 0.7 and once with probability 0.3, so the expected count
    over many epochs matches r_i exactly.

    Parameters
    ----------
    image_ids_in_dataset_order : list[int]
        Image_ids in the order the underlying Dataset returns them. Used
        to map back from RFS's image_id keys to dataset indices.
    repeat_factors : RFSResult
        Output of compute_repeat_factors().
    shuffle : bool
        If True, the yielded indices are shuffled within each epoch
        (recommended for training; disable for deterministic evaluation).
    seed : int
        RNG seed for reproducibility.

    Notes
    -----
    The sampler length is the EXPECTED epoch size (sum of repeat factors,
    rounded). This matches what `len(DataLoader)` reports.

    Pairs naturally with `torch.utils.data.DataLoader(dataset, sampler=sampler)`.
    For Ultralytics integration, you'll need a Trainer subclass that swaps
    in this sampler -- see Phase 3 EXP 3.4 (E.5).
    """

    def __init__(
        self,
        image_ids_in_dataset_order: list[int],
        repeat_factors: RFSResult,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.image_ids = list(image_ids_in_dataset_order)
        self.image_id_to_idx = {iid: i for i, iid in enumerate(self.image_ids)}
        self.repeat_factors = repeat_factors
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        self._idx_factors: list[tuple[int, float]] = []
        for iid in self.image_ids:
            r = repeat_factors.image_repeat_factors.get(iid, 1.0)
            self._idx_factors.append((self.image_id_to_idx[iid], float(r)))

        self._expected_length = max(1, int(round(sum(r for _, r in self._idx_factors))))

    def set_epoch(self, epoch: int) -> None:
        """Call from the Trainer before each epoch for distinct shuffles."""
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator()
        g.manual_seed(self.seed + self._epoch)

        rounded: list[int] = []
        for idx, r in self._idx_factors:
            base = int(math.floor(r))
            frac = r - base
            extra = int(torch.rand((), generator=g).item() < frac)
            for _ in range(base + extra):
                rounded.append(idx)

        if self.shuffle:
            order = torch.randperm(len(rounded), generator=g).tolist()
            rounded = [rounded[i] for i in order]

        return iter(rounded)

    def __len__(self) -> int:
        return self._expected_length


__all__ = [
    "RFSResult",
    "compute_repeat_factors",
    "write_repeat_factors",
    "load_repeat_factors",
    "RepeatFactorSampler",
]
