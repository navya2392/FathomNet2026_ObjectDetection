# Phase 1 — BioCLIP2-as-YOLOv11-backbone: design notes

**Block reference:** D.1 - D.5 in `master_checklist.txt`.
**Status:** research / plan only. **No weights downloaded yet** — some
shape numbers are TODOs to fill in once the checkpoint is on disk.
**Goal:** by EOD Apr 30, have a clear path from "default YOLOv11
backbone" to "BioCLIP2-features-feeding-YOLOv11-neck", with at least
two viable options ranked by risk/effort.

---

## TL;DR

YOLOv11 expects a **multi-scale convolutional feature pyramid** (P3/P4/P5
at strides 8/16/32). BioCLIP2 is a **ViT-L/14** that outputs a single
feature map at stride 14 (or per-token features at stride 14). They don't
plug together natively. We need an **adapter** that bridges the gap.

Three options ranked from lowest-risk to highest-payoff:

| Option | Effort (hr) | Expected Δ-mAP vs YOLO+ImageNet | Risk |
|---|---|---|---|
| A. Frozen-BioCLIP2 features → small adapter → YOLO neck | 6-10 | **+2 to +5 pts** | LOW |
| B. Trainable BioCLIP2 features → adapter → YOLO neck (LoRA) | 12-18 | +4 to +8 pts | MEDIUM |
| C. Distillation: BioCLIP2 logits as soft labels for YOLO | 4-6 | +1 to +3 pts | LOW |

**Recommended starting point: Option A.** Smallest blast radius, fastest
to validate, and we can promote to B if it pays off. Option C is a
side-bet to run in parallel cheaply if GPU time allows.

---

## 1. What BioCLIP2 actually IS at the tensor level

BioCLIP2 is an **OpenCLIP ViT-L/14** trained on TreeOfLife-10M with
contrastive image-text loss (image ↔ taxonomic-name pairs). Concretely:

- **Architecture:** ViT-Large/14 (24 transformer layers, 1024 hidden, 16 heads)
- **Input:** 224×224 RGB images, normalized with OpenCLIP stats
  (mean `[0.48145466, 0.4578275, 0.40821073]`, std `[0.26862954, 0.26130258, 0.27577711]`)
- **Patch size:** 14 → produces **16×16 = 256 patch tokens** + 1 CLS token at 224 input
- **Output options** (we choose at adapter time):
  - `cls_token` only — `(B, 1024)` global feature, useless for detection
  - all patch tokens — `(B, 256, 1024)` reshapeable to `(B, 1024, 16, 16)` spatial map at stride 14
  - intermediate layer features (e.g., layer 12 and 18) — same shape as above per layer
- **Total params:** ~304M (frozen) — bigger than YOLOv11x (~70M)

> **TODO at runtime (D.1):** confirm exact ViT variant from the HuggingFace
> checkpoint metadata. The repo also lists ViT-B/16 BioCLIP2 variants;
> the L/14 is the "main" model and what we want.

### Why BioCLIP2 over plain CLIP

Plain CLIP is trained on 400M web image-text pairs (LAION). Its features
are great for general semantics ("dog", "car", "beach") but mediocre for
fine-grained biology ("amphipod" vs "isopod" — both small crustaceans,
visually nearly identical to non-experts).

BioCLIP2 was trained on TreeOfLife-10M, where the text targets ARE
taxonomic labels at multiple ranks (kingdom → species). The resulting
features cluster by taxonomic similarity, which means classes that look
similar but ARE similar (e.g., two anemone species) get close embeddings,
and classes that look similar but ARE different (anemone vs. tube
anemone) get distant ones. Exactly the inductive bias we need for
FathomNet's 32 marine taxa.

---

## 2. What YOLOv11 needs from a backbone

YOLOv11 (and v8/v5) follow this overall structure:

```
input (B, 3, H, W)
   ↓
backbone — produces multi-scale features:
   P3 at stride 8:   (B, C3, H/8,  W/8)   — small objects
   P4 at stride 16:  (B, C4, H/16, W/16)  — medium objects
   P5 at stride 32:  (B, C5, H/32, W/32)  — large objects
   ↓
neck (PANet/FPN) — fuses P3/P4/P5
   ↓
head — predicts box + class per anchor
```

For YOLOv11x, the channel widths at P3/P4/P5 are roughly
`(C3, C4, C5) = (256, 512, 1024)` (varies per model size: n/s/m/l/x).

> **TODO (D.3):** dump exact backbone output shapes for the YOLOv11 size
> we choose (probably `yolo11m` for speed/quality trade-off on a 4090):
> ```python
> from ultralytics import YOLO
> m = YOLO("yolo11m.pt")
> # use a hook to capture P3/P4/P5 shapes
> ```

The trick: YOLO's backbone is convolutional, so its receptive fields and
feature scales naturally span small/medium/large objects. ViT-L/14 only
outputs a single scale. We have to **synthesize the missing scales**.

---

## 3. Option A — Frozen BioCLIP2 + adapter (RECOMMENDED START)

### Architecture sketch

```
input (B, 3, 640, 640)              <- YOLO standard size
   ↓
   ├── BioCLIP2 ViT-L/14 (frozen, eval mode)
   │      input resized to 224×224 internally
   │      output: (B, 256, 1024) patch tokens, reshape -> (B, 1024, 16, 16)
   │
   ↓                          ↓
   YOLO conv stem (kept)      Adapter (NEW, trainable):
   ↓                            - 1x1 conv: 1024 -> 512
   YOLO P3/P4/P5 (kept)         - upsample to 80x80 (P3 res for 640 input)
   ↓                            - upsample to 40x40 (P4 res)
   ↓                            - keep at 20x20 (P5 res)
   ↓
   Concat at neck level: YOLO_P3 ⊕ Adapter_P3, etc.
   ↓
   Neck (PANet) — slightly widened to absorb extra channels
   ↓
   Head — unchanged
```

### Why this design

- **Frozen ViT** = zero added training cost, BioCLIP2 only runs once per
  forward, no gradient through 304M params, no risk of catastrophic
  forgetting biology priors.
- **Concat at neck** instead of replacing YOLO's backbone entirely =
  the YOLO backbone still gives us local convolutional features tuned by
  COCO pretraining, BioCLIP2 contributes global semantic context.
  Two complementary signals.
- **Adapter is small** (~5M params, 1x1 conv + upsamples): trains fast,
  no weird gradient dynamics.

### Implementation gotchas

1. **Image size mismatch:** YOLO trains at 640×640, BioCLIP2 wants 224×224.
   Two options:
   (i) resize input twice (once for YOLO, once for ViT) — simpler, +5%
       wallclock per forward. Recommended.
   (ii) train YOLO at 224 to match — kills small-object recall, NOT
        recommended for this dataset.
2. **Different normalizations:** YOLO uses `[0,1]` raw, BioCLIP2 uses
   ImageNet-style normalization. Apply both transforms separately to the
   ViT branch, leave YOLO branch alone.
3. **Adapter init:** start with adapter weights ≈ 0 so the first forward
   pass behaves like vanilla YOLO. Then the model gradually learns to
   weight in the BioCLIP2 features. Avoids destabilizing the COCO-pretrained
   weights at step 0.
4. **Frozen ViT in DDP / mixed precision:** make sure to set
   `requires_grad=False` BEFORE wrapping in DDP, or PyTorch will sync
   gradients of frozen params (slow + warns).
5. **Memory:** ViT-L/14 at 224 in fp16 = ~3 GB activation. Plus YOLO at
   640 batch 16 = ~12 GB. Total ~15 GB. Fits on a 24 GB 4090 comfortably,
   tight on a 16 GB card. RunPod plan: A40 (48 GB) or 4090 (24 GB).

### How to wire it into Ultralytics

Ultralytics doesn't have a clean "swap the backbone" API. Two paths:

(i) **Subclass `DetectionModel` and override `_predict_once()`:** capture
    YOLO's intermediate features, run the ViT branch separately, fuse at
    the neck. ~150 lines, contained in one file. **Recommended.**

(ii) **Edit the YAML model definition** (`yolo11m.yaml`) to add a custom
     module. Cleaner Ultralytics-native, but requires registering a new
     `nn.Module` in their module zoo and wading through their YAML parser.

Both options keep YOLOv11's training loop, augmentations, loss, etc.
unchanged — only the model graph differs.

---

## 4. Option B — Trainable BioCLIP2 (LoRA) + adapter

Same as Option A, but instead of freezing the ViT, attach **LoRA adapters**
(rank-8) to its attention layers. LoRA adds only ~1M trainable params on
top of the 304M frozen base, so it's cheap, and lets the ViT specialize
on marine imagery.

**When to upgrade from A to B:** if Option A converges and gives a
clear win (>+2 mAP over baseline), B is worth the time. If A doesn't
help or hurts, jumping to B is unlikely to rescue it.

**Effort:** ~6 extra hours over Option A. Mostly spent debugging
gradient flow through the LoRA wrappers under DDP + mixed precision.

---

## 5. Option C — BioCLIP2 as soft-label distillation

Completely different angle: don't change the YOLO architecture at all.
Instead:

1. Run BioCLIP2 zero-shot classification on every train image's GT crops:
   for each annotated bbox, crop, embed, similarity-match against the
   32 class names → soft probability distribution.
2. Use that soft distribution as an additional CE target (or KL term) for
   YOLO's classification head, with weight λ ≈ 0.3.

**Pros:**
- No architecture changes → can run as a drop-in addon to ANY of the
  other phases.
- Cheap to compute (one BioCLIP2 forward per crop, done ONCE offline).
- Works alongside Option A or B.

**Cons:**
- Pure regularization signal, no actual feature improvement.
- Smaller expected gain (+1 to +3 mAP).

**Recommendation:** queue it as a Phase 4-5 enhancement, not Phase 1.

---

## 6. Risks and fallbacks

| Risk | Likelihood | Fallback |
|---|---|---|
| BioCLIP2 features give zero benefit | medium | Roll back to ImageNet, no time lost (Option A is independent) |
| Adapter overfits on small classes | medium | Add dropout to adapter, freeze adapter for last 5 epochs |
| Memory blows up at batch=16 | low | Drop YOLO batch to 12, accumulate grads to keep effective batch |
| BioCLIP2 weights don't load (HF api change) | low | Vendor weights to local + load manually |
| Training is 2× slower than vanilla YOLO | high | Accept it; we have GPU budget for 1 BioCLIP2 5-fold run |

---

## 7. What to do tomorrow morning (when you wake up)

In priority order:

1. **D.1 — Download BioCLIP2 weights** (~15 min, ~2.5 GB):
   ```bash
   huggingface-cli login            # uses HUGGINGFACE_HUB_TOKEN from .env
   huggingface-cli download imageomics/bioclip-2 --local-dir weights/bioclip2
   ```
2. **D.2 — Smoke-test loading:**
   ```python
   import open_clip
   model, _, preprocess = open_clip.create_model_and_transforms(
       "hf-hub:imageomics/bioclip-2"
   )
   model.eval()
   import torch
   with torch.no_grad():
       feats = model.encode_image(torch.randn(2, 3, 224, 224))
   print(feats.shape)  # expect (2, 768) or similar — confirm
   ```
3. **D.3 — Capture YOLOv11 P3/P4/P5 shapes** with a forward hook. Fill
   in the TODOs in section 2 of this doc.
4. **D.4 — Implement Option A adapter** as a single file
   `src/models/bioclip2_yolo.py`. ~150 lines.
5. **D.5 — Smoke-test:** train for 100 steps on fold 0, confirm loss
   decreases and shapes flow end-to-end. NO target mAP yet, just "did
   it not crash?"

Estimated total time to "trains end-to-end": ~6 hours. Reserve evening
of Apr 30 for the first real fold0 training run.
