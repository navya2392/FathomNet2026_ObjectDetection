# FathomNet 2026 — Marine Species Detection

**Finished 14th on the public leaderboard** of the FathomNet 2026 / CLEF 2026 underwater object detection competition (mAP@[.50:.95] = 0.1143).

![Train vs Test contact sheet](figures/train_vs_test_contact_sheet.jpg)

*Train (left): NOAA + SOI ROV imagery. Test (right): MBARI VARS framegrabs. Different research institutions, different ROVs, different oceans, different decades.*

## TL;DR

- The competition's central trick is that **train and test come from completely different research institutions**. The training set is NOAA + SOI imagery; the test set is 100% MBARI footage from different submarines, oceans, and decades. The detector you train has never seen anything like the test set.
- Anything that made the model "fit the training data better" — fancier loss functions, sampling tricks, pseudo-labels, underwater color correction — **hurt the leaderboard**.
- Things that worked were all **architecture- and domain-agnostic**: bigger input resolution, multi-scale test-time inference, ensembling two different architectures.
- The biggest single win came *after* training was done: re-classifying each detected box using a foundation-model classifier that only kicks in when two independent voters agree. That alone added +0.0226 to the score with no retraining.

## Headline numbers

| Metric | Value | Notes |
|---|---|---|
| **Public leaderboard rank** | **14th** | FathomNet 2026 / CLEF 2026 |
| **Best LB** | **0.1143 mAP@[.50:.95]** | Final composite pipeline |
| Best pure detector | 0.0917 | YOLOv8x + RT-DETR-l cross-architecture ensemble |
| Single-model anchor | 0.0863 | YOLOv8x with high-res training + multi-scale TTA |
| Submissions scored | 28 | 17 during training, 11 post-submission |

## The problem

32 classes of deep-sea marine species, ~6,500 training images, ~1,400 test images. The catch:

- **Train = 100% NOAA and SOI imagery** — ROVs and cameras from two specific oceanographic institutions.
- **Test = 100% MBARI VARS framegrabs** — a completely different institution, different ROVs, different oceans, different decades, different camera resolutions.
- Most damning: 14% of the test images are 720×486 — a resolution that does not appear *anywhere* in the training set.

This means a model can be near-perfect on a held-out slice of the training data and still fail catastrophically on the test set. We saw this directly — validation mAP was around 0.43, leaderboard mAP was around 0.09. **The 8× gap isn't overfitting; it's domain shift.**

![Sample test images](figures/sample_test_images.png)

*Sample test images. Different lighting, different equipment, lower resolution, different class distributions from the training set.*

## How we approached it

The thinking evolved in three phases.

### Phase 1 — Try the obvious things

Train a decent YOLO model, pick a sensible pretrained backbone, run inference at multiple scales, ensemble with another architecture. This got us to **0.0917** — a respectable result built from three reliable techniques:

- **Higher input resolution.** Going from imgsz=640 to 1024 added +0.0137 by itself, because higher-resolution training gives the model more flexibility about which image scales it can handle at test time.
- **Multi-scale test-time inference.** Running the detector at six different image scales and merging the predictions with Weighted Boxes Fusion. Added +0.0228 cumulatively.
- **Cross-architecture ensemble.** Fusing predictions from YOLOv8x and RT-DETR-l (a transformer-based detector). RT-DETR-l alone scored *worse* than YOLOv8x, but their errors disagreed enough that combining them lifted the score by another +0.0054.

We also tried five different pretrained initializations. The **MBARI 315k** checkpoint (a model pretrained on related marine imagery) beat the COCO and ImageNet baselines by a wide margin — that one decision was worth +0.078 mAP, the largest single jump in the whole project.

### Phase 2 — Realize the "make the model better" tricks all hurt

This was the most important and counterintuitive finding. We tried a long list of techniques designed for **long-tail class distributions** (a 1000× imbalance across 32 classes) and **domain shift**:

- Loss functions tuned for rare classes (positive-unlabeled loss, equalized focal variants, frequency-weighted losses)
- Sampling strategies that oversample rare classes
- Augmentation tricks (class-aware copy-paste)
- Pseudo-labeling — train on the model's own confident predictions on the test set
- BatchNorm running-statistic adaptation
- Underwater color correction at inference time

**Every one of them hurt the leaderboard.** Not by a little — by 1–2 percentage points each.

The pattern took a while to see, but once it clicked, the project pivoted:

> Every losing technique makes the model fit the **source** (training) distribution more tightly. Pseudo-labeling teaches the model to trust its own predictions on test images — but those predictions came from a model that has never seen MBARI footage. Loss reweighting helps rare classes within the training set, not across the institutional gap. Sampling and augmentation make the same mistake.

We tested this directly: three mechanistically independent attacks on class imbalance (loss reweighting, sampling, augmentation) all hurt. **Class imbalance wasn't the bottleneck. Domain shift was.**

### Phase 3 — Stop training, start optimizing the predictions

Once it was clear that no training-time intervention would help, we pivoted to **post-submission optimization** — operations that work on the detector's predictions, not on the detector itself. Three composable ideas, applied in order:

**1. Two-classifier consensus relabeling — +0.0174**

For every detected box on the test set, crop the region and run *two independent classifiers* on the crop:

- A nearest-neighbor lookup in DINOv2 (a general-purpose vision foundation model) feature space against labeled training crops.
- A 32-way classifier trained on the same DINOv2 features.

If both classifiers unanimously agree that the detector predicted the wrong class, we replace the label. Otherwise we leave it alone. About 3,000 of ~290,000 predictions got relabeled.

**Why this worked when YOLO-on-YOLO pseudo-labeling didn't:** DINOv2 was pretrained on a much broader image distribution than any YOLO checkpoint. It "sees" MBARI imagery through a different lens than a NOAA-trained YOLO does. Two voters reading from genuinely different feature spaces actually filter mistakes; two voters reading from correlated representations just compound them.

**2. Weighted Boxes Fusion stacking — +0.0043**

Generate two more variants of the relabel (one with a fine-tuned DINOv2, one with the MLP alone at a higher confidence threshold) and WBF-merge them with the detector ensemble. Asymmetric weights — proven winner heavy, variants light — preserved the best result while picking up a small diversity lift.

**3. Letterbox-border noise filter — +0.0002**

The 14% of test images at 720×486 are letterboxed inside black bars. The detector was producing thousands of low-confidence false positives inside those bars. A simple geometric rule (drop predictions where the box's interior mean RGB is below 15) removed ~7,000 of them, all with score < 0.10. Tiny lift, free win.

The composite pipeline (detector ensemble → consensus relabel → WBF stack → border filter) ended at **0.1143 mAP@[.50:.95]**, a **+0.0226 gain over the best pure detector** — entirely from inference-time post-processing.

## What didn't work — at a glance

| Category | What we tried | Why it hurt |
|---|---|---|
| **Long-tail-aware training** | EQLv2, equalized focal loss, repeat-factor sampling, class-aware copy-paste | Class imbalance isn't the bottleneck — domain shift is |
| **Pseudo-labeling** | Single-teacher, multi-teacher consensus | Both teachers see test images through the same NOAA-trained lens; they agree about the wrong things |
| **Inference-time domain tricks** | BatchNorm running-stat adaptation, underwater CLAHE color correction at inference only | Train/inference mismatch, or simply not enough lever |
| **Source-side custom losses** | Positive-unlabeled (PU) loss, federated loss | Tightens fit to source domain |
| **Aggressive post-submission variants** | Crop-only classifier (no detector), full retrained SSL, broad relabeling thresholds, size-based relabel | Same lesson as Phase 2 — conservative + consensus wins, aggressive + greedy loses |

The full numerics for all 28 scored submissions live in [`notebooks/final_pipeline.ipynb`](notebooks/final_pipeline.ipynb) (scoreboard chart, per-class AP chart, full pipeline diagram).

## The bigger lesson

**Institutional domain shift is the dominant variable, and the techniques that won were the ones that don't care about it.** Bigger input resolution, multi-scale inference, cross-architecture ensembling, foundation-model consensus relabeling — none of these "know" anything about NOAA vs MBARI. They just work on images, regardless of which institution captured them.

The techniques that hurt were the ones that implicitly assume train and test come from the same distribution: loss reweighting, sampling, augmentation, pseudo-labeling. They're not bad techniques in general — they just need an assumption that this competition broke.

## What we'd do next

- **Re-run two blocked experiments**: a domain-adversarial training run (DANN) and a hierarchical-classification auxiliary head — both implemented in `src/` but blocked by a framework regression we ran out of time to patch around.
- **Add a third ensemble member** under the same consensus-relabel pipeline.
- **SAHI tile-based inference** at higher resolution for the 720×486 letterboxed images, which are particularly exposed to small-object recall failures.

## Tech stack

PyTorch, Ultralytics YOLO11/YOLOv8, RT-DETR, DINOv2 (ViT-L/14), Weighted Boxes Fusion, MBARI 315k pretrained weights. See `src/` for the supporting modules — losses, samplers, classifiers, and the relabel pipeline.

## Repository

| Path | What's there |
|---|---|
| [`SETUP.md`](SETUP.md) | Install + commands to run training, inference, and the demo notebook |
| [`notebooks/final_pipeline.ipynb`](notebooks/final_pipeline.ipynb) | Full results notebook with the **28-experiment scoreboard chart**, per-class AP chart, and pipeline diagram — committed with cached outputs, browseable on GitHub |
| `src/`, `scripts/`, `tests/`, `configs/` | Source modules, training and inference entry points, unit tests, dataset configs |
| `data/`, `weights/` | Folder placeholders — see each README for download instructions |

## License

Code under Apache 2.0 (see `LICENSE`). Pretrained weights subject to their upstream licenses (Ultralytics AGPL-3.0, MBARI 315k CC-BY 4.0).
