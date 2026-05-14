"""Enumerate test images for inference scripts (Tier 1A / predict pipeline).

Keeps ordering deterministic (sorted by basename) so smoke subsets and hflip
runs stay comparable.
"""
from __future__ import annotations

from pathlib import Path

TEST_IMAGE_SUFFIXES = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)


def sorted_test_image_paths(test_dir: Path) -> list[Path]:
    paths = [
        p for p in test_dir.iterdir()
        if p.is_file() and p.suffix.lower() in TEST_IMAGE_SUFFIXES
    ]
    return sorted(paths, key=lambda p: p.name)


def capped_sorted_test_paths(
    test_dir: Path,
    max_images: int | None,
) -> list[Path]:
    """All sorted images, or first ``max_images`` if given."""
    paths = sorted_test_image_paths(test_dir)
    if max_images is None:
        return paths
    return paths[:max_images]


__all__ = [
    "TEST_IMAGE_SUFFIXES",
    "sorted_test_image_paths",
    "capped_sorted_test_paths",
]
