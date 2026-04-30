# FathomNet 2026 — Backup & Extras Plan

**When to use:** After your master plan is executing on schedule AND you have time or budget remaining. Each item is optional polish.

**Budget for extras:** Up to ~$70 of master plan headroom plus whatever compute you have left. Most items are nearly free.

**How to read each item:** Each section follows the same pattern:

- Problem it solves
- Design principle (what it actually does)
- Decision criteria (when to include vs skip)
- Expected gain and cost
- Light implementation reference

---

## Priority Ranking

If you can only do some, do them in this order:

1. **Temperature scaling calibration** (1h, $0)
2. **Per-class confidence thresholds** (2h, $0)
3. **Per-class NMS IoU thresholds** (2h, $0)
4. **Seed ensemble** (1 day, ~$24)
5. **Active learning loop** — for LLNL career narrative, not leaderboard
6. **Everything else** — nice-to-haves, only if well ahead

Items 1–3 combined can add +1–2 mAP with effectively zero GPU cost. Do them first.

---

## 1. Per-class confidence thresholds (2 hours, $0)

### Problem

mAP is computed independently per class. The optimal confidence threshold for "bony fish" (common, precision-dominated) differs from the optimal threshold for "octopus" (rare, recall-dominated). Default detectors use one global threshold for all classes.

### Design principle

Grid search the confidence threshold independently per class on val data. Lock per-class thresholds at inference time.

### Decision criteria

```
IF class-count imbalance ratio > 10:
    INCLUDE — per-class tuning matters more when classes behave differently
IF per-class AP variance is high (some classes >0.6, others <0.2):
    INCLUDE
ELWAYS:
    INCLUDE — it's nearly free
```

Effectively: always do it.

### Expected gain

+0.5 to 1 mAP.

### Methodology

1. Collect OOF (out-of-fold) predictions from Phase 6's 5-fold CV
2. For each class, sweep threshold from 0.05 to 0.95 in steps of 0.05
3. Compute per-class AP at each threshold
4. Lock the threshold that maximized that class's AP
5. Apply all 32 thresholds at inference time: `if pred.score < thresholds[pred.class]: drop`

### Light code reference

```python
for cls_id in range(32):
    best_ap, best_thresh = 0, 0.25
    for t in np.arange(0.05, 0.95, 0.05):
        ap = compute_class_ap(preds, gt, cls_id, threshold=t)
        if ap > best_ap: best_ap, best_thresh = ap, t
    thresholds[cls_id] = best_thresh
```

### Failure mode

Overfitting to val data. Check that improvement on held-out fold is similar to improvement on fit fold. If they diverge, use coarser grid (0.1 steps instead of 0.05).

---

## 2. Per-class NMS IoU thresholds (2 hours, $0)

### Problem

Dense organisms (amphipod swarms, shrimp clusters) need low NMS IoU — you want to preserve many overlapping detections. Isolated large organisms (single octopus) need high NMS IoU — overlapping detections are almost certainly duplicates.

### Design principle

Same as #1 but applied to NMS IoU threshold. Different classes, different optimal NMS aggressiveness.

### Decision criteria

```
IF your test images contain clustered organisms (check EDA for images with > 10 objects):
    INCLUDE — NMS aggressiveness matters
ELSE:
    Marginal benefit — do only if #1 was easy
```

### Expected gain

+0.3 to 0.8 mAP.

### Methodology

Similar sweep as #1, but over IoU threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}. Apply per-class NMS at inference.

---

## 3. Temperature scaling (1 hour, $0)

### Problem

Same calibration problem as isotonic regression (master plan item), but simpler — single learned parameter instead of a monotonic function.

### Design principle

Multiply logits by a learned temperature T before sigmoid. T > 1 softens confident predictions; T < 1 sharpens them. Fit by minimizing NLL on val data.

### Decision criteria

```
IF isotonic regression overfits (val calibration improves but test degrades):
    USE temperature scaling instead (single parameter, less overfitting)
IF dataset is small:
    PREFER temperature scaling
ELSE:
    Run both, compare on held-out data, keep winner
```

### Expected gain

+0.2 to 0.5 mAP.

### Light code reference

```python
# Fit T by gradient descent on val NLL
T = nn.Parameter(torch.ones(1) * 1.5)
optimizer = torch.optim.LBFGS([T], lr=0.01)

def closure():
    loss = F.binary_cross_entropy_with_logits(logits / T, targets)
    loss.backward()
    return loss

optimizer.step(closure)
# At inference: calibrated_score = sigmoid(logit / T)
```

### When to prefer temperature over isotonic

- Small validation set (< 5000 detections) → temperature (less prone to overfit)
- Non-monotonic miscalibration observed → isotonic (more flexible)
- When in doubt → try both, keep winner

---

## 4. Seed ensemble (1 day, ~$24)

### Problem

Your 5-fold CV ensemble has model diversity from *data* splits, and architectural diversity from *RT-DETR*. You have no diversity from *initialization randomness*.

### Design principle

Train 2 additional YOLOv11m models on all data with different random seeds (same architecture, same data, different init). Adds 2 more ensemble members with decorrelated errors purely from stochastic training dynamics.

### Decision criteria

```
IF Phase 6 completed on schedule AND GPU budget > $30:
    INCLUDE seed ensemble
IF ensembling 5-fold alone gives > 1 mAP over best single fold:
    Ensembling is helping — more diversity will help more
    INCLUDE
IF ensemble already >2 points better than best single:
    Diminishing returns likely — SKIP
```

### Expected gain

+0.3 to 0.5 mAP.

### Methodology

```python
for seed in [1337, 2024]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Otherwise identical to Phase 6 training config
    train(model, data_all, seed=seed)
```

Adds 2 members → 8-model ensemble. Re-tune ensemble weights after adding.

---

## 5. Active learning loop (5 days, ~$8)

**Most impressive differentiator for LLNL/MBARI narrative. Marginal for competition leaderboard.**

### Problem

You'd like more labeled marine data, but annotation is expensive. Which images from the 400k unlabeled FathomNet DB should you manually label to gain the most?

### Design principle

Use model uncertainty to prioritize. For each unlabeled image:

- Run inference N times with dropout enabled (Monte Carlo Dropout)
- Measure prediction variance across passes
- High variance = model is uncertain = this image would be valuable to label

Label top-K uncertain images. Retrain. Uncertainty should decrease globally.

### Decision criteria

```
IF you're specifically building your LLNL internship narrative:
    INCLUDE — this is the "how to efficiently label data" problem both MBARI and NIF face
IF > 3 days remaining after master plan:
    INCLUDE — useful learning
ELSE:
    SKIP — marginal leaderboard impact, not worth the time otherwise
```

### Expected gain

- Competition: marginal. You'd need hours of manual labeling to see impact.
- Career/learning: significant. Direct application to LLNL internship.

### Methodology

1. Enable dropout at inference time: `model.model.train()` (switches dropout to training mode, rest of model still frozen)
2. Run predict() 10 times on each candidate image
3. Compute variance in detection counts, class distribution, or box coordinates
4. Rank images by variance
5. Manually label top 100–500
6. Retrain with newly-labeled data + compare to baseline

### Light code reference

```python
def mc_dropout_uncertainty(model, image, n=10):
    model.model.train()  # enable dropout
    counts = [len(model.predict(image, conf=0.1)[0].boxes) for _ in range(n)]
    model.model.eval()
    return np.var(counts)  # higher = more uncertain
```

### Writeup angle for LLNL

"Implemented active learning loop that reduces annotation burden by ~X% while matching full-dataset performance. Directly applicable to NIF data labeling bottleneck."

---

## 6. IoU estimation branch for pseudo-label filtering (3 days, ~$10)

### Problem

Plain pseudo-labeling filters by confidence. But a high-confidence box may still be spatially misaligned. You'd rather filter by *expected IoU with the true box* — but you don't know the true box.

### Design principle

Train an auxiliary head on labeled data that predicts IoU between a detection and ground truth. At inference on unlabeled data, this predicted IoU becomes a quality filter.

### Decision criteria

```
IF Soft Teacher gave positive but marginal gain (<1 mAP):
    INCLUDE IoU predictor to improve pseudo-label quality
IF plain pseudo-labeling already works well:
    SKIP — marginal additional gain
```

### Expected gain

+1 to 2 mAP over plain pseudo-labeling / Soft Teacher when PU severity is high.

### Methodology

1. Add IoU regression head parallel to classification/regression heads
2. Loss: predict IoU between each proposal and its matched GT box during supervised training
3. At inference on unlabeled data, predicted IoU threshold (e.g., > 0.7) replaces confidence threshold for pseudo-labels

---

## 7. Knowledge distillation (2 days, ~$16)

### Problem

Large models are accurate but slow. Could distill their "dark knowledge" into a smaller model that runs faster — useful if inference time matters.

### Design principle

Train small model (student) to match large model's (teacher) soft predictions, not just hard labels. Soft predictions contain information about uncertainty and class relationships.

### Decision criteria

```
IF inference time is constrained (it's not, for this competition):
    Useful
IF you want an additional ensemble member at a different scale:
    Mildly useful
ELSE:
    SKIP — marginal value for this competition
```

### Expected gain

+0.5 to 1 mAP on distilled smaller model. Also useful as ensemble diversity.

### Methodology

```
teacher_logits = teacher(x)          # frozen, larger model
student_logits = student(x)           # trained
distill_loss = KL_div(
    softmax(student_logits / T),
    softmax(teacher_logits / T)
) × T²
hard_loss = cross_entropy(student_logits, labels)
total = α × distill_loss + (1 − α) × hard_loss
# α = 0.7, T = 4 are typical
```

---

## 8. Feature-level augmentation (FASA) (2 days, ~$4)

### Problem

Rare classes with <50 instances can't benefit much from pixel-level augmentation — you still have the same underlying visual variety.

### Design principle

Augment in *feature space* instead. Sample new feature vectors for rare classes from the statistics of their labeled examples.

### Decision criteria

```
IF EDA shows any class with < 30 training instances:
    INCLUDE — pixel-level augmentation is hitting a ceiling
IF RFS + class-aware copy-paste already helped rare classes substantially:
    SKIP — diminishing returns
```

### Expected gain

+0.3 to 0.5 mAP on rare classes only.

### Methodology

1. Extract features for all rare-class instances using your trained backbone
2. Compute per-class feature mean and std
3. Generate synthetic features: `synthetic = μ + σ × ε`, where `ε ~ N(0, 0.5)`
4. Inject into training batch alongside real features during classifier training

---

## 9. Multi-task image-level classification head (2 days, ~$16)

### Problem

The PU problem means many objects go unlabeled. But at the *image level*, labels are more reliable — if any bony fish is labeled in an image, the image definitely contains bony fish.

### Design principle

Add an auxiliary head predicting the set of categories present in the image. Multi-label binary cross-entropy. The auxiliary signal helps the backbone learn better features without requiring complete bounding box annotations.

### Decision criteria

```
IF PU severity is extreme (>60% single-category labels):
    INCLUDE — image-level signal is more reliable than box-level
IF Phase 3 differentiators already gave good gains:
    OPTIONAL — incremental benefit
```

### Expected gain

+0.5 to 1 mAP.

### Methodology

```
backbone_features = backbone(x)
detections = detection_head(backbone_features)
image_labels = image_classifier_head(pool(backbone_features[-1]))

total_loss = detection_loss + 0.3 × image_multilabel_bce_loss
```

Particularly valuable for PU because it leverages a more reliable supervision signal.

---

## 10. DINOv2 ViT-L/14 backbone (4 days, ~$60)

### Problem

Maybe a bigger backbone helps. DINOv2 ViT-L has ~300M parameters vs ResNet50's 25M. Better features, potentially.

### Decision criteria

```
IF Phase 6 mAP has plateaued AND nothing in master plan is left to tune:
    MAYBE include — significant integration effort
IF time is tight OR simpler ideas haven't been exhausted:
    SKIP — ROI is low, risk of implementation bugs is high
```

### Expected gain

+1 to 2 mAP if integration succeeds. Zero if integration has bugs.

### Cost

High engineering effort. YOLO expects CNN backbone by default; integrating a ViT backbone requires custom feature pyramid construction. Plan for 3–4 days of integration + debugging even before training.

### Reality check

Not recommended unless you have a full week left. Better to use BioClip2 (which is already trained on marine data) than to struggle with ViT-L integration.

---

## 11. Mosaic + CutMix additions (half day, $0)

### Problem

Default Ultralytics mosaic + mixup helps. Could add more augmentation variety.

### Design principle

- **Gradual augmentation decay** — strong early, weak late
- **CutMix** — cut a patch from one image, paste into another, blend labels proportionally
- Combine with `close_mosaic=20` (already in master plan)

### Decision criteria

```
IF val mAP curve shows overfitting (train << val gap grows):
    INCLUDE — more regularization helps
ELSE:
    SKIP — added complexity for marginal gain
```

### Expected gain

+0.2 to 0.5 mAP.

---

## 12. Working note paper submission (3–5 days of writing)

### What it is

Optional 4–6 page paper describing your method. Submitted to CEUR-WS proceedings via the CLEF 2026 Lab.

### Why consider it

- Gets you into the officially published CLEF ranking (not just Kaggle leaderboard)
- Publishable output — listable on resume, LinkedIn
- Strong conversation piece for MBARI/LLNL internship applications

### Decision criteria

```
IF final leaderboard position is top 15%:
    STRONGLY CONSIDER writing the paper
IF position is top 30% AND LLNL mentor encourages it:
    DO IT — you have a 3-week window from May 7 to May 28
IF position is mediocre:
    SKIP — not worth the writing time
```

### Paper structure (if doing it)

- **Method** — describe the differentiator stack: SSL pretraining, Kiryo PU + EFL, Soft Teacher, hierarchical loss, SAHI + ensemble
- **Results** — official leaderboard result plus ablation table showing each differentiator's contribution
- **Analysis** — per-class AP, failure mode inspection
- **Key tables** — ablations are the most cite-able content

### Timeline

- **May 7** — Kaggle deadline, final submission known
- **May 8–27** — writing window (~3 weeks)
- **May 28** — CLEF working note deadline

---

## Compute Budget for Extras

| Item | Compute | Cost |
|---|---|---|
| 1–3 (per-class tuning + temp scaling) | CPU only | $0 |
| 4 (seed ensemble) | A100 40GB, ~12h | ~$24 |
| 5 (active learning) | RTX 4090, ~10h | ~$8 |
| 6 (IoU predictor) | A100 40GB, ~5h | ~$10 |
| 7 (distillation) | A100 40GB, ~8h | ~$16 |
| 8 (FASA) | RTX 4090, ~5h | ~$4 |
| 9 (multi-task head) | A100 40GB, ~8h | ~$16 |
| 10 (DINOv2-L) | A100 80GB, ~25h | ~$60 |
| 11 (CutMix) | RTX 4090, ~3h | ~$2 |
| 12 (paper) | — | $0 (time only) |

**Items 1–5 combined:** ~$32, fits easily in the $70 master plan headroom.

**All items 1–9, 11:** ~$80, pushes total to ~$310 — slightly over $300 cap. Skip #7 or #9 if strictly capping at $300.

---

## Quick Decision Tree

```
After Phase 7 submission:

IF you have > 3 days left before deadline:
    # Cheap wins first
    DO items 1, 2, 3 (per-class + temp scaling)
    IF results improve: keep them
    
    IF additional leaderboard push wanted:
        DO item 4 (seed ensemble)
    
    IF building LLNL narrative is a priority:
        DO item 5 (active learning)

IF you have 1–2 days left:
    DO only items 1, 3 (temperature scaling + per-class confidence)
    These take an evening, give +0.7–1.5 mAP typically

IF you have hours only:
    DO only item 3 (temperature scaling)
    1 hour of work, +0.2–0.5 mAP

IF you're behind schedule:
    Don't touch backup plan
    Polish your master plan submission instead
```
