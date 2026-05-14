"""Underwater image preprocessing for the Phase 4 EXP 4.2 experiment.

Block reference: F.3 in master_checklist.txt.

What problem does this solve
----------------------------
Deep-sea images suffer from three coupled distortions:

1. **Color cast.** Water absorbs red/yellow wavelengths preferentially, so
   deep-sea images are dominated by blue-green tones. Standard
   ImageNet-pretrained backbones never saw this color distribution.
2. **Low contrast.** Forward-scattering of artificial lights creates a
   "veil" effect; histograms are compressed into a narrow band.
3. **Locally dark regions.** Shadows in benthic photography hide
   fine-grained taxa cues.

Two cheap, well-understood classical preprocessing tricks fix most of
this without GPU: **gray-world white balance** + **CLAHE** (contrast
limited adaptive histogram equalization).

Both operate on uint8 BGR or RGB images. Both are deterministic (not
augmentations) -- apply once per image, NOT per training step.

Phase 0 EDA cell 12 outputs the per-channel mean/std and triggers
the gray-world / CLAHE recommendation when the per-channel imbalance
exceeds 0.15 (red/blue ratio < 0.85, etc).

When NOT to apply
-----------------
- Test-time only? No -- if applied at training time, MUST also be applied
  at inference. Distribution shift between train and test wrecks mAP.
- All images? Phase 0 says ~73% of FathomNet images have severe color
  cast. The remaining ~27% (well-lit photogrammetry) are fine without.
  EXP 4.2 ablates: (a) apply to all, (b) apply only to images failing
  the per-channel test, (c) skip entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


@dataclass
class GrayWorldStats:
    """Per-channel mean before/after gray-world correction (for diagnostics)."""

    means_before: tuple[float, float, float]
    means_after: tuple[float, float, float]
    scales: tuple[float, float, float]
    color_cast_score: float


def gray_world_balance(
    img: np.ndarray,
    *,
    target: str = "mean",
    clip_uint8: bool = True,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, GrayWorldStats]:
    """Apply gray-world white balance to a BGR or RGB image.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8 or float. Channel order doesn't matter
        (the algorithm is symmetric in channels).
    target : str
        - 'mean': scale each channel so its mean equals the global mean.
                  This preserves overall brightness.
        - 'gray128': scale each channel so its mean equals 128 (mid-gray).
                     This re-centers brightness; can over-brighten dark images.
        Default 'mean' is safer for the FathomNet distribution.
    clip_uint8 : bool
        If True (default), clip output to [0, 255] and cast to uint8 IF
        the input was uint8. If input was float, no clipping or casting.
    return_stats : bool
        If True, also return a GrayWorldStats with diagnostics.

    Returns
    -------
    np.ndarray
        Same shape and dtype as input (or float64 if input was non-uint8
        and you don't pass clip_uint8=False).

    Notes
    -----
    Edge case: if a channel mean is zero (all-black channel), it cannot be
    scaled meaningfully. We skip the channel and warn via the stats.
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be HxWx3, got shape {img.shape}")

    is_uint8 = img.dtype == np.uint8
    img_f = img.astype(np.float64)

    means_before = (
        float(img_f[:, :, 0].mean()),
        float(img_f[:, :, 1].mean()),
        float(img_f[:, :, 2].mean()),
    )

    if target == "mean":
        target_value = sum(means_before) / 3.0
    elif target == "gray128":
        target_value = 128.0 if is_uint8 else 0.5
    else:
        raise ValueError(f"target must be 'mean' or 'gray128', got {target!r}")

    scales = []
    out = np.empty_like(img_f)
    for c in range(3):
        m = means_before[c]
        if m < 1e-6:
            scales.append(1.0)
            out[:, :, c] = img_f[:, :, c]
            continue
        s = target_value / m
        scales.append(s)
        out[:, :, c] = img_f[:, :, c] * s

    if clip_uint8 and is_uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)
    elif clip_uint8:
        out = np.clip(out, 0, 1) if img_f.max() <= 1.0 else np.clip(out, 0, 255)

    means_after = (float(out[:, :, c].astype(np.float64).mean()) for c in range(3))
    means_after_t = tuple(means_after)
    color_cast = max(means_before) / (min(means_before) + 1e-6) - 1.0

    if return_stats:
        stats = GrayWorldStats(
            means_before=means_before,
            means_after=means_after_t,  # type: ignore[arg-type]
            scales=tuple(scales),  # type: ignore[arg-type]
            color_cast_score=color_cast,
        )
        return out, stats
    return out


def apply_clahe(
    img: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (8, 8),
    color_space: str = "lab",
) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8.
    clip_limit : float
        CLAHE clip limit. Default 2.0 (OpenCV's default). Higher values
        produce more contrast but amplify noise; lower values are gentler.
        For underwater imagery, 1.5 - 3.0 is the useful range.
    tile_grid_size : (int, int)
        Tile grid for local histogram equalization. Default (8, 8) is
        OpenCV's default; smaller tiles -> more local contrast.
    color_space : str
        - 'lab' (default): equalize the L channel of LAB color space.
                Recommended -- preserves color while boosting luminance contrast.
        - 'gray':         equalize a grayscale version (3-channel output is
                grayscale repeated). Use only as ablation; throws away color.

    Returns
    -------
    np.ndarray
        Same shape and dtype as input (uint8).

    Notes
    -----
    Requires OpenCV (cv2). Standard install is via `pip install opencv-python-headless`.
    On RunPod's pytorch image, cv2 is preinstalled.
    """
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"img must be HxWx3 uint8, got shape={img.shape} dtype={img.dtype}")
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for CLAHE. Install via "
            "`pip install opencv-python-headless`."
        ) from exc

    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tile_grid_size)

    if color_space == "lab":
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    if color_space == "gray":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_eq = clahe.apply(gray)
        return cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"color_space must be 'lab' or 'gray', got {color_space!r}")


def underwater_preprocess(
    img: np.ndarray,
    *,
    apply_gray_world: bool = True,
    apply_clahe_step: bool = True,
    gray_world_target: str = "mean",
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Combined underwater pipeline: gray-world -> CLAHE.

    Order matters: gray-world FIRST (fixes the global color cast), then
    CLAHE (fixes local contrast on the corrected image). Reversed order
    over-amplifies the cast.

    Parameters mirror the individual functions; flags let you A/B-test
    each stage independently in Phase 4 EXP 4.2.
    """
    out = img
    if apply_gray_world:
        out = gray_world_balance(out, target=gray_world_target, clip_uint8=True)
    if apply_clahe_step:
        out = apply_clahe(out, clip_limit=clahe_clip_limit, tile_grid_size=clahe_tile_grid_size)
    return out


def _single_scale_retinex(img: np.ndarray, sigma: float) -> np.ndarray:
    """Single-scale retinex: log(I) - log(GaussianBlur(I, sigma)).

    Operates on float32 in [eps, +inf). Channels handled independently.
    """
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV required for MSRCR.") from exc
    eps = 1.0
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return np.log(img + eps) - np.log(blurred + eps)


def msrcr(
    img: np.ndarray,
    *,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
    G: float = 5.0,
    b: float = 25.0,
    alpha: float = 125.0,
    beta: float = 46.0,
    low_clip: float = 0.01,
    high_clip: float = 0.99,
) -> np.ndarray:
    """Multi-Scale Retinex with Color Restoration (MSRCR).

    Reference: Jobson et al., "A Multiscale Retinex for Bridging the Gap
    Between Color Images and the Human Observation of Scenes," IEEE TIP 1997.

    Springer 2024 (data-centric underwater detection framework) found MSRCR
    consistently beat CLAHE+gray-world by 3.2-13.4 mAP across underwater
    domain-shift settings — that's why we prefer it over CLAHE for E9.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3), dtype uint8 (BGR or RGB).
    sigmas : tuple of float
        Three Gaussian scales (small=local detail, large=global illumination).
        Defaults are the classic MSRCR triple (15, 80, 250).
    G, b : float
        Gain and bias applied after color restoration.
    alpha, beta : float
        Color restoration weights (Jobson defaults).
    low_clip, high_clip : float
        Output histogram clip percentiles for final dynamic-range stretch.

    Returns
    -------
    np.ndarray
        Same shape and dtype (uint8) as input.
    """
    if img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
        raise ValueError(f"img must be HxWx3 uint8, got shape={img.shape} dtype={img.dtype}")
    img_f = img.astype(np.float32) + 1.0

    msr = np.zeros_like(img_f)
    weight = 1.0 / len(sigmas)
    for sigma in sigmas:
        msr += weight * _single_scale_retinex(img_f, sigma)

    img_sum = np.sum(img_f, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img_f) - np.log10(img_sum))

    msrcr_img = G * (msr * color_restoration + b)

    out = np.zeros_like(msrcr_img)
    for c in range(3):
        ch = msrcr_img[:, :, c]
        lo = np.percentile(ch, low_clip * 100.0)
        hi = np.percentile(ch, high_clip * 100.0)
        if hi - lo < 1e-6:
            out[:, :, c] = 0
        else:
            out[:, :, c] = np.clip((ch - lo) / (hi - lo), 0.0, 1.0) * 255.0

    return out.astype(np.uint8)


def underwater_preprocess_msrcr(
    img: np.ndarray,
    *,
    sigmas: tuple[float, ...] = (15.0, 80.0, 250.0),
) -> np.ndarray:
    """Convenience wrapper: MSRCR with default Jobson params.

    This is the recommended preprocessing per Springer 2024 underwater
    domain-shift literature review. Use in place of `underwater_preprocess`
    (gray-world + CLAHE) for E9.
    """
    return msrcr(img, sigmas=sigmas)


def needs_underwater_preproc(
    img: np.ndarray, *, color_cast_threshold: float = 0.15
) -> bool:
    """Heuristic: should we apply underwater preprocessing to this image?

    Returns True if the per-channel imbalance exceeds the threshold:

        max(channel_means) / min(channel_means) - 1 > color_cast_threshold

    Phase 4 EXP 4.2 (b) uses this to apply preprocessing only to the
    ~73% of images that actually need it. The remaining ~27% are fine
    without and applying CLAHE would over-amplify their already-good
    contrast.

    Parameters
    ----------
    img : np.ndarray
        Shape (H, W, 3).
    color_cast_threshold : float
        Default 0.15 (15% imbalance). Phase 0 EDA suggested this value.

    Returns
    -------
    bool
    """
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"img must be HxWx3, got shape {img.shape}")
    img_f = img.astype(np.float64)
    means = [float(img_f[:, :, c].mean()) for c in range(3)]
    if min(means) < 1e-6:
        return True
    cast = max(means) / min(means) - 1.0
    return cast > color_cast_threshold


__all__ = [
    "GrayWorldStats",
    "gray_world_balance",
    "apply_clahe",
    "underwater_preprocess",
    "underwater_preprocess_msrcr",
    "msrcr",
    "needs_underwater_preproc",
]
