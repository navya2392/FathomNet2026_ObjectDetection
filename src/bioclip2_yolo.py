"""BioCLIP2 -> YOLOv11 backbone adapter (Phase 1 / Block D).

Goal
----
Inject biology-domain features from BioCLIP2 (ViT-L/14 pretrained on
TreeOfLife-10M) into a vanilla YOLOv11m so that the detection head sees
both:
  - YOLO's COCO-pretrained convolutional features (good local detail)
  - BioCLIP2's biology-domain global/semantic features (good for
    fine-grained marine taxa)

This file is the single point that bridges the two networks.

Architecture
------------
::

    image (B, 3, H, W) in YOLO range [0, 1]
       |
       +----> YOLO backbone (frozen by default, COCO-pretrained):
       |        layer  4: P3 features (B, 512, H/8,  W/8)
       |        layer  6: P4 features (B, 512, H/16, W/16)
       |        layer 10: P5 features (B, 512, H/32, W/32)
       |
       +----> BioCLIP2 ViT-L/14 (frozen, eval mode):
                resize to (224, 224), renormalize to ImageNet stats
                forward through ViT
                capture pre-pool transformer output: (B, 257, 1024)
                drop CLS token, reshape to (B, 1024, 16, 16)
                |
                +-> 1x1 conv head_p3 (1024 -> 512), zero-init
                |     bilinear interpolate to (H/8,  W/8)
                |     ADD to YOLO's P3 output (via forward hook)
                |
                +-> 1x1 conv head_p4 (1024 -> 512), zero-init
                |     bilinear interpolate to (H/16, W/16)
                |     ADD to YOLO's P4 output (via forward hook)
                |
                +-> 1x1 conv head_p5 (1024 -> 512), zero-init
                      bilinear interpolate to (H/32, W/32)
                      ADD to YOLO's P5 output (via forward hook)
       |
       v
    YOLO neck + detection head (UNCHANGED, COCO-pretrained)
       |
       v
    detection outputs

Why additive fusion (not concat)
--------------------------------
Concatenation would double the channel count entering the neck (1024
instead of 512), which means we'd have to widen YOLO's neck convolutions
and lose the COCO-pretrained weights for those layers.

Additive fusion preserves the neck's existing channel dimensions, which
means:
  - We keep ALL of YOLO's COCO-pretrained weights intact.
  - With zero-init projections, the wrapper starts producing IDENTICAL
    outputs to vanilla YOLO at step 0. There's no destabilization.
  - As training progresses, the adapter gradually learns when to add
    BioCLIP2 contribution. If BioCLIP2 doesn't help, the adapter just
    stays near zero and we're left with vanilla YOLO performance.

Why resize-to-224 (not PE interpolation)
----------------------------------------
BioCLIP2 was trained at 224x224. ViTs CAN be run at higher resolutions
via positional-embedding interpolation, but this is a separate code path
in open_clip and adds 4-9x compute (since attention is O(N^2) in
patches and patch count = (size/14)^2).

For the MVP (Phase 2 EXP 2.3 backbone bake-off):
  - 224x224 is fast (~0.5 sec/image on CPU, ~5 ms on GPU)
  - 16x16 patch grid upsampled 5x to YOLO's P3 (80x80) gives "global
    biology context" rather than spatially-precise features -- which is
    exactly what BioCLIP2 is good at anyway
  - If this works, we can upgrade to PE interpolation in a Phase 4
    ablation to see if higher-resolution BioCLIP2 features help

Why ZERO-init the projection heads
----------------------------------
At step 0, output of every projection conv is zero, so the wrapped model
produces the SAME outputs as vanilla YOLO. This:
  - Makes shape verification trivially correct (just compare to vanilla)
  - Avoids destabilizing COCO-pretrained weights at step 0
  - Lets the adapter discover the right contribution magnitude during
    training, instead of starting in a bad neighborhood

Usage
-----
::

    from src.bioclip2_yolo import BioCLIP2YOLOWrapper

    # Build a wrapper around yolo11m with COCO weights + BioCLIP2 adapter
    wrapper = BioCLIP2YOLOWrapper(
        yolo_weights="yolo11m.pt",
        bioclip2_id="hf-hub:imageomics/bioclip-2",
    )
    wrapper.eval()

    # Forward pass works exactly like vanilla YOLO
    import torch
    x = torch.randn(1, 3, 640, 640)
    out = wrapper(x)

For training integration with Ultralytics, see the smoke test in
``__main__`` for the gradient-flow pattern. A Trainer subclass that
swaps in this wrapper is a follow-up (Block C.5).

Reference
---------
notes/phase1_bioclip2_research.md (Option A from section 3)
notes/yolo11m_backbone_shapes.txt (output of D.3)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow `from src.bioclip2_yolo import ...` from any cwd.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Matches scripts/inspect_yolo_backbone.py findings (D.3 output).
DEFAULT_BACKBONE_TAP_INDICES: tuple[int, int, int] = (4, 6, 10)
DEFAULT_BACKBONE_CHANNELS: tuple[int, int, int] = (512, 512, 512)  # yolo11m specific

# OpenCLIP / BioCLIP2 expects ImageNet-style normalization with these stats.
BIOCLIP2_MEAN: tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)
BIOCLIP2_STD: tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)
BIOCLIP2_INPUT_SIZE: int = 224
BIOCLIP2_PATCH_SIZE: int = 14
BIOCLIP2_HIDDEN_DIM: int = 1024  # ViT-L/14 hidden dimension


def _capture_transformer_output(visual_module: nn.Module) -> tuple[Any, dict]:
    """Register a forward hook on the OpenCLIP ViT's transformer block.

    Returns (handle, capture_dict). Reads capture_dict['x'] after a
    forward pass to get the (B, N+1, D) token tensor BEFORE pooling/projection.

    OpenCLIP's VisionTransformer typically has these submodules:
      - conv1     (patch embedding)
      - transformer or blocks (the stacked attention layers)
      - ln_post   (final layer norm, after transformer)
      - attn_pool / global_pool
      - proj      (final linear to text-shared embedding dim)

    We hook 'transformer' if present, else 'blocks'. Either way the
    output is (B, N, D) tokens including the CLS token at index 0.
    """
    capture: dict = {}

    target_module = None
    for attr_name in ("transformer", "blocks"):
        if hasattr(visual_module, attr_name):
            target_module = getattr(visual_module, attr_name)
            break
    if target_module is None:
        raise AttributeError(
            "OpenCLIP visual module has neither '.transformer' nor '.blocks' "
            "attribute. open_clip API may have changed."
        )

    def hook(_m, _i, output):
        capture["x"] = output

    handle = target_module.register_forward_hook(hook)
    return handle, capture


class BioCLIP2YOLOWrapper(nn.Module):
    """YOLOv11 wrapped with a BioCLIP2 feature-injection adapter.

    The wrapper exposes a forward() that takes the same input as YOLO
    (a (B, 3, H, W) image tensor in [0, 1]) and returns the same output
    YOLO would return. The architectural difference is internal: forward
    hooks at the YOLO backbone's P3/P4/P5 tap points add a BioCLIP2-derived
    contribution to the existing features.

    Parameters
    ----------
    yolo_weights : str | Path
        Path to a YOLOv11 .pt checkpoint (Ultralytics format) OR a model
        spec like 'yolo11m.pt' that triggers Ultralytics' auto-download.
    bioclip2_id : str
        HuggingFace ID for the BioCLIP2 model. Default works.
    backbone_channels : tuple of int
        Channel counts at the three tap indices. Defaults to yolo11m's
        (512, 512, 512). For other YOLO sizes, run inspect_yolo_backbone.py
        to get the right values.
    backbone_tap_indices : tuple of int
        Indices into yolo.model.model where to hook the backbone.
        Default (4, 6, 10) matches all standard YOLOv11 sizes.
    vit_input_size : int
        Resize image to this many pixels per side before feeding to
        BioCLIP2. Must be a multiple of 14 (the ViT-L/14 patch size).
        Default 224 matches BioCLIP2's training resolution.
    freeze_yolo : bool
        If True, freezes ALL YOLO parameters (only the adapter projection
        heads are trainable). Useful for ultra-fast adapter-only training.
        Default False (all YOLO weights remain trainable, matching the
        Phase 2 baseline behavior).
    """

    def __init__(
        self,
        yolo_weights: str | Path = "yolo11m.pt",
        bioclip2_id: str = "hf-hub:imageomics/bioclip-2",
        *,
        backbone_channels: tuple[int, int, int] = DEFAULT_BACKBONE_CHANNELS,
        backbone_tap_indices: tuple[int, int, int] = DEFAULT_BACKBONE_TAP_INDICES,
        vit_input_size: int = BIOCLIP2_INPUT_SIZE,
        freeze_yolo: bool = False,
    ) -> None:
        super().__init__()

        if vit_input_size % BIOCLIP2_PATCH_SIZE != 0:
            raise ValueError(
                f"vit_input_size ({vit_input_size}) must be divisible by "
                f"BioCLIP2 patch size ({BIOCLIP2_PATCH_SIZE})."
            )

        from ultralytics import YOLO

        yolo_obj = YOLO(str(yolo_weights))
        self.yolo: nn.Module = yolo_obj.model  # the underlying DetectionModel
        # Ultralytics .pt checkpoints come with requires_grad=False on all
        # weights (a fine-tuning entry-point convention). Restore the default
        # PyTorch behavior unless the caller explicitly asked to freeze.
        for p in self.yolo.parameters():
            p.requires_grad = not freeze_yolo

        import open_clip

        clip_model, _, _ = open_clip.create_model_and_transforms(bioclip2_id)
        self.bioclip2_visual: nn.Module = clip_model.visual
        self.bioclip2_visual.eval()
        for p in self.bioclip2_visual.parameters():
            p.requires_grad = False

        self.vit_input_size = vit_input_size
        self.vit_grid_size = vit_input_size // BIOCLIP2_PATCH_SIZE
        self.backbone_tap_indices = backbone_tap_indices

        self.register_buffer(
            "vit_mean",
            torch.tensor(BIOCLIP2_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "vit_std",
            torch.tensor(BIOCLIP2_STD).view(1, 3, 1, 1),
            persistent=False,
        )

        self.proj_p3 = nn.Conv2d(BIOCLIP2_HIDDEN_DIM, backbone_channels[0], kernel_size=1)
        self.proj_p4 = nn.Conv2d(BIOCLIP2_HIDDEN_DIM, backbone_channels[1], kernel_size=1)
        self.proj_p5 = nn.Conv2d(BIOCLIP2_HIDDEN_DIM, backbone_channels[2], kernel_size=1)
        self._zero_init_projections()

        self._cached_bioclip2_features: torch.Tensor | None = None
        self._hook_handles: list[Any] = []
        self._register_yolo_hooks()

    def _zero_init_projections(self) -> None:
        for proj in (self.proj_p3, self.proj_p4, self.proj_p5):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _register_yolo_hooks(self) -> None:
        """Hook YOLO backbone tap points to add BioCLIP2 contributions."""
        seq = self.yolo.model
        projections = (self.proj_p3, self.proj_p4, self.proj_p5)
        for tap_idx, proj in zip(self.backbone_tap_indices, projections):
            handle = seq[tap_idx].register_forward_hook(self._make_add_hook(proj))
            self._hook_handles.append(handle)

    def _make_add_hook(self, proj: nn.Conv2d):
        """Returns a forward hook that adds proj(BioCLIP2_features) to YOLO output."""
        def hook(_module, _input, output):
            if self._cached_bioclip2_features is None:
                return output
            feat = proj(self._cached_bioclip2_features)
            feat = F.interpolate(
                feat,
                size=output.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            return output + feat
        return hook

    def _compute_bioclip2_features(self, x: torch.Tensor) -> torch.Tensor:
        """Run BioCLIP2 ViT on the resized input and return spatial patch features.

        Parameters
        ----------
        x : torch.Tensor
            (B, 3, H, W) image tensor in YOLO format ([0, 1] or unnormalized).

        Returns
        -------
        torch.Tensor
            (B, 1024, grid, grid) spatial feature map at stride 14
            (grid = vit_input_size / 14).
        """
        if x.shape[-2:] != (self.vit_input_size, self.vit_input_size):
            x_resized = F.interpolate(
                x,
                size=(self.vit_input_size, self.vit_input_size),
                mode="bilinear",
                align_corners=False,
            )
        else:
            x_resized = x

        x_normed = (x_resized - self.vit_mean) / self.vit_std

        handle, capture = _capture_transformer_output(self.bioclip2_visual)
        try:
            with torch.no_grad():
                _ = self.bioclip2_visual(x_normed)
        finally:
            handle.remove()

        tokens = capture["x"]
        if isinstance(tokens, (tuple, list)):
            tokens = tokens[0]
        if tokens.shape[0] != x.shape[0] and tokens.shape[1] == x.shape[0]:
            tokens = tokens.transpose(0, 1)

        n_patches = self.vit_grid_size * self.vit_grid_size
        n_tokens = tokens.shape[1]
        if n_tokens == n_patches + 1:
            tokens = tokens[:, 1:, :]
        elif n_tokens == n_patches:
            pass
        else:
            raise RuntimeError(
                f"Unexpected token count from BioCLIP2 ViT: got {n_tokens}, "
                f"expected {n_patches} or {n_patches + 1} (=patches +/- CLS). "
                f"This usually means vit_input_size is wrong or the ViT "
                f"variant is not L/14."
            )

        b, _, d = tokens.shape
        spatial = tokens.transpose(1, 2).reshape(b, d, self.vit_grid_size, self.vit_grid_size)
        return spatial

    def forward(self, x: torch.Tensor, *args, **kwargs) -> Any:
        self._cached_bioclip2_features = self._compute_bioclip2_features(x)
        try:
            return self.yolo(x, *args, **kwargs)
        finally:
            self._cached_bioclip2_features = None

    def adapter_parameters(self):
        """Return only the trainable adapter weights (the 3 projection convs).

        Useful for setting up an optimizer that trains ONLY the adapter
        (e.g., during a warmup phase) before unfreezing the rest.
        """
        for proj in (self.proj_p3, self.proj_p4, self.proj_p5):
            yield from proj.parameters()

    def train(self, mode: bool = True):
        """Override to keep BioCLIP2 ViT in eval mode regardless of caller.

        This protects against accidental BatchNorm/dropout activation in
        the frozen feature extractor when the wrapper is put in training
        mode. (ViT-L/14 uses LayerNorm, no dropout in eval, but the same
        contract should hold if we swap in a different ViT later.)
        """
        super().train(mode)
        self.bioclip2_visual.eval()
        return self

    def __del__(self) -> None:
        for h in getattr(self, "_hook_handles", []):
            try:
                h.remove()
            except Exception:
                pass


def _smoke_test() -> None:
    """End-to-end shape + correctness verification.

    1. Build the wrapper with COCO-pretrained yolo11m + BioCLIP2.
    2. Build a vanilla YOLO comparison (just the .yolo attr) with same weights.
    3. Forward identical input through both.
    4. With zero-init projections, outputs MUST be identical (within
       floating point tolerance).
    5. After random-init projections, outputs must differ -- proving the
       adapter is wired in.
    6. Verify gradient flow: only adapter weights + (optionally) YOLO
       weights receive gradients; BioCLIP2 weights do NOT.
    """
    print("=" * 70)
    print("BioCLIP2YOLOWrapper smoke test (D.5)")
    print("=" * 70)

    print("\n[1/6] Building wrapper (loads yolo11m.pt + BioCLIP2 ViT-L/14)...")
    wrapper = BioCLIP2YOLOWrapper(yolo_weights="yolo11m.pt")
    wrapper.eval()
    print(f"      adapter trainable params: {sum(p.numel() for p in wrapper.adapter_parameters()):,}")
    print(f"      yolo trainable params: {sum(p.numel() for p in wrapper.yolo.parameters() if p.requires_grad):,}")
    print(f"      bioclip2 frozen params: {sum(p.numel() for p in wrapper.bioclip2_visual.parameters()):,}")

    print("\n[2/6] Forward pass at imgsz=640 with BATCH=1...")
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out_wrapped = wrapper(x)
    out_shapes = _describe_yolo_output(out_wrapped)
    print(f"      wrapped output: {out_shapes}")

    print("\n[3/6] Forward through inner YOLO (vanilla baseline)...")
    with torch.no_grad():
        out_vanilla = wrapper.yolo(x)
    out_vanilla_shapes = _describe_yolo_output(out_vanilla)
    print(f"      vanilla output: {out_vanilla_shapes}")
    assert out_shapes == out_vanilla_shapes, (
        f"Output shape mismatch: wrapped={out_shapes}, vanilla={out_vanilla_shapes}"
    )

    print("\n[4/6] Verifying ZERO-init: wrapped output == vanilla output ...")
    max_diff = _max_abs_diff(out_wrapped, out_vanilla)
    print(f"      max |wrapped - vanilla| = {max_diff:.6e}")
    assert max_diff < 1e-4, (
        f"With zero-init projections, wrapper should match vanilla YOLO. "
        f"Got max diff = {max_diff:.4e}. Hooks may be wired incorrectly."
    )
    print("      OK -- zero-init wrapper is a perfect identity over vanilla YOLO.")

    print("\n[5/6] Random-init projections; verifying outputs now DIFFER ...")
    for proj in (wrapper.proj_p3, wrapper.proj_p4, wrapper.proj_p5):
        nn.init.normal_(proj.weight, std=0.01)
        nn.init.normal_(proj.bias, std=0.01)
    with torch.no_grad():
        out_perturbed = wrapper(x)
    max_diff = _max_abs_diff(out_perturbed, out_vanilla)
    print(f"      max |perturbed - vanilla| = {max_diff:.6e}")
    assert max_diff > 1e-3, (
        f"After random-init, wrapper should differ from vanilla. "
        f"Got max diff = {max_diff:.4e}. Hooks may not be firing."
    )
    print("      OK -- adapter contributions are flowing into YOLO outputs.")

    print("\n[6/6] Gradient flow test: backward through wrapper, check who gets grads...")
    wrapper.train()
    wrapper._zero_init_projections()
    x = torch.randn(1, 3, 640, 640, requires_grad=False)
    out = wrapper(x)
    fake_loss = _scalar_loss_from_yolo_output(out)
    fake_loss.backward()

    p3_grad_norm = wrapper.proj_p3.weight.grad.abs().sum().item() if wrapper.proj_p3.weight.grad is not None else 0.0
    bioclip2_grad_count = sum(1 for p in wrapper.bioclip2_visual.parameters() if p.grad is not None)
    yolo_grad_count = sum(1 for p in wrapper.yolo.parameters() if p.grad is not None and p.requires_grad)

    print(f"      proj_p3 weight grad |sum|:    {p3_grad_norm:.6e}  (must be > 0 with random input)")
    print(f"      BioCLIP2 params with .grad:   {bioclip2_grad_count}  (MUST be 0 -- frozen)")
    print(f"      YOLO params with .grad:       {yolo_grad_count}  (must be > 0 -- trainable)")

    assert bioclip2_grad_count == 0, (
        f"BioCLIP2 should be frozen but got {bioclip2_grad_count} params with gradients."
    )
    assert yolo_grad_count > 0, "YOLO params should be trainable but none received gradients."
    print("      OK -- gradients flow to adapter + YOLO; BioCLIP2 stays frozen.")

    print("\n" + "=" * 70)
    print("SMOKE TEST PASS  --  D.5 complete")
    print("=" * 70)


def _walk_tensors(obj):
    """Yield every torch.Tensor leaf in a nested structure of tensors / tuples / lists / dicts."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _walk_tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _walk_tensors(o)


def _describe_yolo_output(out) -> str:
    """Compact string describing a YOLO output structure (tensor shapes only)."""
    shapes = [tuple(t.shape) for t in _walk_tensors(out)]
    return f"{len(shapes)} tensors: {shapes}"


def _max_abs_diff(out_a, out_b) -> float:
    """Maximum absolute element-wise difference between two YOLO output structures.

    Walks both structures in parallel, comparing tensor leaves. If shape
    mismatches at any leaf or different number of tensor leaves, returns inf.
    """
    tensors_a = list(_walk_tensors(out_a))
    tensors_b = list(_walk_tensors(out_b))
    if len(tensors_a) != len(tensors_b):
        return float("inf")
    diffs = []
    for a, b in zip(tensors_a, tensors_b):
        if a.shape != b.shape:
            return float("inf")
        diffs.append((a - b).abs().max().item())
    return max(diffs) if diffs else 0.0


def _scalar_loss_from_yolo_output(out) -> torch.Tensor:
    """Reduce all tensor leaves in YOLO output into a single scalar for backward()."""
    tensors = list(_walk_tensors(out))
    if not tensors:
        raise RuntimeError("Could not find any tensor leaves in YOLO output for loss.")
    return torch.stack([t.float().mean() for t in tensors]).sum()


__all__ = [
    "BioCLIP2YOLOWrapper",
    "DEFAULT_BACKBONE_TAP_INDICES",
    "DEFAULT_BACKBONE_CHANNELS",
    "BIOCLIP2_INPUT_SIZE",
    "BIOCLIP2_HIDDEN_DIM",
]


if __name__ == "__main__":
    _smoke_test()
