# FathomNet 2026 — Marine Species Detection

Object detection on a long-tail, cross-domain underwater dataset for the FathomNet 2026 / CLEF 2026 Kaggle competition. The training set is 100% NOAA + SOI imagery; the test set is 100% MBARI VARS framegrabs from different ROVs, oceans, and decades. **The dominant challenge isn't class imbalance or rare species — it's institutional domain shift, and every modeling decision in this project follows from that observation.**

![Train vs Test contact sheet](figures/train_vs_test_contact_sheet.jpg)

*Train (NOAA + SOI ROV imagery, left) vs test (MBARI VARS framegrabs, right). Different ROVs, oceans, decades, and image resolutions. This gap drove every modeling decision.*

## TL;DR

- **Best result: 0.1143 mAP@[.50:.95]** on the public leaderboard via a composite pipeline — E27 cross-architecture detector ensemble, then DINOv2-consensus class relabeling, then WBF stacking of peer variants, then a letterbox-border noise filter.
- Best **pure-model** result was 0.0917 (E27 = YOLOv8x + RT-DETR-l cross-arch ensemble). **Post-submission optimization added +0.0226 on top** by treating relabeling and stacking as a separable inference-time phase.
- The train/test split is **institutionally disjoint**. This is the dominant variable.
- Of 28 scored submissions across both phases, **8 lifted the leaderboard** and **20 hurt or were noise**. The pattern is the same in both phases: domain-agnostic, conservative interventions help; source-fit, aggressive ones hurt.
- The val → LB ratio for the pure detector is **8×**, far beyond what classical overfitting predicts. The pipeline was audited end-to-end (pycocotools matches Ultralytics within 0.004); the gap is structural.

## Headline numbers

| Metric | Value | Notes |
|---|---|---|
| **Best public LB** | **0.1143 mAP@[.50:.95]** | Slot 9 — E27 + Path A consensus + WBF stack + border filter |
| Best pure-model result | 0.0917 | E27 cross-architecture ensemble (YOLOv8x + RT-DETR-l) |
| Anchor (single model) | 0.0863 | YOLOv8x + MBARI 315k init + 6-scale TTA WBF |
| Post-submission lift over E27+P3 (0.0924) | **+0.0219** | +23.7% relative; consensus relabel → WBF stack → border filter |
| Cross-arch ensemble lift over anchor | +0.0054 | WBF(YOLOv8x + RT-DETR-l), equal weights |
| Resolution lift (640 → 1024) | +0.0137 | Biggest single training lever |
| TTA lift (1-scale → 6-scale WBF) | +0.0228 | Composes cleanly with resolution |
| Experiments executed | 28 scored | Model training (17) + post-submission (11) |

The competition leader scored 0.321; finishing at 0.1143 places this work below the top tier — but the experimental story, particularly the systematic refutation of source-side optimizations under institutional domain shift, is the substance of the project.

## The problem

FathomNet 2026 is object detection on deep-sea marine imagery: 32 classes, ~6,500 train images, ~1,400 test images. The dominant challenge is **institutional disjoint** between train and test:

- **Train:** 100% NOAA + SOI imagery (4,733 from `d2l7vcm2vanphr.cloudfront.net`, 1,730 from `hurlimage.soest.hawaii.edu`)
- **Test:** 100% MBARI VARS framegrabs (86% Doc Ricketts ROV, 10% Ventana, 3% Tiburon, 1% other)

Different ROVs, different oceans, different decades. Resolution mismatch is structural as well: train is only 3840×2160 and 1920×1080, while 14% of the test set is 720×486 — a resolution the model has never seen in training. The competition is fundamentally an **out-of-distribution generalization** problem, not a within-distribution optimization problem.

![Sample test images](figures/sample_test_images.png)

*Sample MBARI test imagery. Lower resolution, different lighting, different ROV equipment, and class distributions that don't match the NOAA + SOI training set.*

## Approach

Detection is solved as a YOLO-style supervised problem with **four orthogonal levers** attacked in sequence:

1. **The right pretrained init** — a 5-way bake-off identified MBARI 315k YOLOv8 as the only init worth promoting (+0.064 mAP over Megalodon, +0.078 over vanilla COCO).
2. **High-resolution training + multi-scale TTA + WBF** — the only consistently positive training lever family. imgsz=1024 + 6-scale TTA + Weighted Boxes Fusion is the anchor recipe.
3. **Cross-architecture ensemble** — the winning training-phase lever. WBF of YOLOv8x and RT-DETR-l adds +0.0054 over the best single model — pure architectural diversity, since RT-DETR-l solo underperforms the anchor by 0.018.
4. **Post-submission optimization (Phase 7)** — once retraining stops paying, the project pivots to an inference-time layer: re-classify each E27 detection with a frozen DINOv2 + k-NN + MLP consensus rule (relabel only on unanimous disagreement), then WBF-stack peer variants, then drop letterbox-border noise. This phase added **+0.0226 over the pure-model best**.

A wide range of source-side optimizations (PU loss, equalized focal loss, long-tail samplers, pseudo-labeling, BatchNorm adaptation, underwater preprocessing) were tested and **rejected on leaderboard evidence**. The pattern is that any technique fitting tighter to the source domain (NOAA + SOI) hurts on the target domain (MBARI).

## What worked

### 1. Resolution: imgsz=640 → 1024 retrain, **+0.0137 LB**

The single biggest lever in the entire training phase. Higher-resolution training transfers more useful features to a test set that contains image scales the model never saw during training (14% of test is 720×486, a resolution absent from train). All subsequent training used imgsz=1024 as the default.

### 2. Multi-scale TTA + WBF: 4-scale → 6-scale, **+0.0228 cumulative LB**

Inference at scales {480, 640, 832, 1024, 1280, 1536} with Weighted Boxes Fusion gives the model multiple effective resolutions at test time. 6 scales beat 4 scales by +0.0088 — diminishing returns are not yet hit. This is the anchor recipe (E1 = LB 0.0863) for every subsequent ensemble experiment.

### 3. Cross-architecture ensemble: **+0.0054 over anchor**, training-phase best 0.0917

WBF of YOLOv8x (E1, LB 0.0863) and RT-DETR-l (T2, LB 0.0684) at equal weights produced the project's best **pure-model** LB. The lift is pure architectural diversity: T2 solo is 0.018 *below* the anchor, but its predictions disagree with E1's on enough images that fusing the two corrects errors on both sides.

### 4. MBARI 315k pretrained init beats COCO by +0.078

A 5-way pretrained-init bake-off ranked candidates after 10 epochs on fold-0 val:

| Rank | Candidate | mAP@[.50:.95] |
|---|---|---|
| 1 | **MBARI 315k YOLOv8** | **0.389** |
| 2 | Megalodon YOLOv8x (single-class FathomNet objectness) | 0.325 |
| 3 | YOLOv11m + COCO (control) | 0.310 |

MBARI 315k beat the COCO baseline on 27 of 32 classes. The biggest wins were on rare classes (isopod +0.312, calycophoran siphonophore +0.291, hydroid +0.281) — exactly where mAP@[.50:.95] is most punishing. BioCLIP2 (ViT-L/14) and Megafishdetector (YOLOv5 format) were incompatible with the Ultralytics 8.x stack and dropped.

### 5. Post-submission optimization (Phase 7): **+0.0226 over E27**, FINAL BEST 0.1143

After the training pipeline plateaued, the project moved to a separable inference-time phase. Three composable mechanisms, applied in sequence:

**(a) Path A consensus relabel — Slot 1, LB 0.1098 (+0.0174 over E27+P3):**
- Extract crops from every E27 predicted box on the test set.
- Pass each crop through frozen DINOv2 (ViT-L/14) features.
- Predict a class with **two independent classifiers** trained on labeled train crops: a k-NN over DINOv2 features and a 32-way MLP softmax.
- **Relabel a prediction only when both classifiers unanimously agree on a different class** (~3,062 relabels). The unanimity rule is the load-bearing element — it filters confirmation bias by requiring two architecturally distinct classifiers to converge.

**(b) WBF stacking of peer relabel variants — Slot 8, LB 0.1141 (+0.0043 over Slot 1):**
- Generate two more relabel variants alongside Slot 1: Slot A (fine-tuned DINOv2 + STRICT consensus, LB 0.1038) and Slot B (MLP-only with prob ≥ 0.85, LB 0.1031).
- WBF-fuse {E27, Slot 1, Slot A, Slot B} with **asymmetric weights** (1.5 / 0.5 / 0.7 / 0.7 — Slot 1 is the proven winner, others contribute diversity).
- Net lift is small but real: WBF surfaces consensus across peers without overwriting Slot 1's choices.

**(c) Letterbox-border noise filter — Slot 9, LB 0.1143 (+0.0002 over Slot 8) — ★ FINAL BEST:**
- 14% of test images are 720×486 — letterboxed inside a black bar.
- A geometric rule (mean RGB > 15, min black-band depth > 5 px) drops ~7,022 low-confidence predictions (all score < 0.10, median 0.008) that sit entirely inside the bars.
- Tiny but free lift; confirms the analysis from the contact-sheet figure.

The total chain (E27 → Path A consensus → WBF stack → border filter) lifts the project from 0.0917 to 0.1143 — a **+23.7% relative improvement** without touching detector weights.

## What didn't work

### Every source-side training optimization

These were all attempted as solo retrains with the anchor's 6-scale TTA recipe; all underperformed the anchor:

| Method | LB | Δ vs anchor |
|---|---|---|
| PU loss (Kiryo) | 0.0600 | -0.0175 |
| Underwater preprocessing (CLAHE + gray-world, inference only) | 0.0665 | -0.0198 |
| RT-DETR-l solo | 0.0684 | -0.0179 |
| Repeat-Factor Sampling | 0.0689 | -0.0174 |
| Multi-teacher consensus pseudo-labels | 0.0695 | -0.0168 |
| imgsz=1280 retrain | 0.0730 | -0.0133 |
| EQLv2 long-tail loss | 0.0765 | -0.0098 |
| Single-teacher pseudo-labels | 0.0773 | -0.0090 |
| AdaBN BN re-stats | 0.0811 | -0.0052 |
| Class-aware copy-paste | 0.0827 | -0.0036 |
| Hflip TTA | 0.0844 | -0.0019 |

### Pseudo-labeling is structurally broken under institutional disjoint

Both single-teacher (T1, anchor self-distillation) and multi-teacher consensus (E28, E1 + T2 with IoU ≥ 0.5 and conf ≥ 0.7) hurt. Both teachers were trained on NOAA + SOI imagery and see test (MBARI) images through the same domain-shift lens — they *agree about the wrong things*. **Note the contrast with Phase 7:** Path A consensus also uses two classifiers, but they're DINOv2 features (not the YOLO detector) trained on labeled crops — a completely separate signal source from the detector, so the unanimity rule actually filters domain-correlated mistakes instead of compounding them.

### Class imbalance is not the bottleneck

EQLv2 (loss reweighting), RFS (sampling), and class-aware copy-paste (augmentation) are three mechanistically independent attacks on the long tail. All three hurt the leaderboard. The 1000× class-count range is real, but it isn't what's costing us mAP — the domain gap is.

### Domain shift is not BatchNorm-fixable

AdaBN measured a real distribution shift in BN running statistics (top |Δvar| = 1.48 across backbone BN layers on test images), but applying the adapted stats *lowered* LB by 0.005. Either (a) the EMA-trained anchor's BN is already near-optimal for the test distribution, or (b) detection (high-res feature maps) is less BN-sensitive than classification. TENT was skipped on the same logic.

### Aggressive Phase 7 variants

Six of the ten post-submission slots also hurt — and they hurt for the same reason aggressive training interventions did:

| Slot | LB | Δ vs Slot 1 (0.1098) | Why it hurt |
|---|---|---|---|
| Slot 4 — Path A2 (MLP-only argmax, ~50% relabel) | 0.0981 | -0.0117 | Too aggressive; relabeling without consensus overshoots |
| Slot 2 — k-NN only (no MLP) | 0.0945 | -0.0153 | k-NN alone insufficient; MLP needed for veto |
| Slot 6 — Path B (MLP classifier on crops + TTA, replace boxes) | 0.0622 | -0.0476 | Crop-level classification doesn't substitute for detection |
| Slot A STRICT — FT DINOv2 + STRICT consensus | 0.1038 | -0.0060 | Fine-tuning DINOv2 didn't beat frozen — extra source-fit |
| Slot B 0.85 — MLP-only (prob ≥ 0.85) | 0.1031 | -0.0067 | MLP without k-NN consensus weaker (still useful in WBF stack) |
| Slot D SSL — SSL t35 + YOLO fine-tune + multiscale TTA | 0.0504 | -0.0594 | End-to-end SSL pipeline failed on MBARI imagery |
| Slot 10 — pairwise size-violation relabel on Slot 9 | 0.1086 | -0.0057 vs Slot 9 | Size-based MLP relabel too noisy |

### Two domain-adaptation experiments were blocked by an Ultralytics 8.4.39 regression

DANN (`src/dann.py`) and the hierarchical aux head (`src/hierarchical_loss.py`) both depend on the trainer's `self.hyp` being attribute-accessible (`self.hyp.box`, `.cls`, `.dfl`). In 8.4.39, `self.hyp` is a `dict`, breaking the access pattern. The fix is mechanical (wrap with `getattr` / `dict.get` helpers, defaults 7.5 / 0.5 / 1.5) but the competition deadline arrived before either retrain could happen. Both modules remain in `src/` as methodology evidence.

## All 28 scored experiments

<details>
<summary><b>Click to expand the full experiments table</b></summary>

Each row is a scored Kaggle submission. **INCLUDE** = composable winner; **EXCLUDE** = solo loser; **EXCLUDE-S** = solo hurt but useful as ensemble component; **BLOCKED** = framework bug prevented training. Δ is vs anchor (E1 = 0.0863).

### Phase 1–6 — Model training

| # | Exp | Recipe | LB | Δ vs anchor | Verdict | Takeaway |
|---|---|---|---|---|---|---|
| 1 | Baseline | Single-scale 640 | 0.0498 | — | baseline | 8× val→LB gap confirmed |
| 2 | aug-640 | Heavy aug at 640 | 0.0502 | +0.0004 | NOISE | Aug alone at 640 does nothing |
| 3 | aug-1024 | Heavy aug at 1024 | 0.0635 | +0.0137 | INCLUDE | **Resolution is the biggest lever** |
| 4 | 4-scale TTA | 4-scale TTA WBF | 0.0775 | +0.0140 | INCLUDE | TTA composes with resolution |
| 5 | PU loss | Kiryo PU at imgsz=1024 | 0.0600 | -0.0175 | EXCLUDE | PU Kiryo hurts; deprioritized |
| 6 | **E1 (anchor)** | 6-scale TTA WBF | **0.0863** | +0.0088 | INCLUDE | 6 scales > 4 scales |
| 7 | E2 hflip | E1 + hflip TTA | 0.0844 | -0.0019 | EXCLUDE | Hflip hurts; marine orgs have natural orientation |
| 8 | E4 preproc | E1 + CLAHE/gray-world at inference | 0.0665 | -0.0198 | EXCLUDE | Train/infer preprocessing must match |
| 9 | E5 imgsz=1280 | imgsz=1280 retrain + 6-scale TTA | 0.0730 | -0.0133 | EXCLUDE | Fewer effective updates at batch=4 |
| 10 | E6 AdaBN | E1 + BN re-stats refit | 0.0811 | -0.0052 | EXCLUDE | Domain shift is not BN-fixable |
| 11 | E9 EQLv2 | EQLv2 long-tail retrain | 0.0765 | -0.0098 | EXCLUDE | Long-tail loss didn't help |
| 12 | T1 pseudo R1 | Single-teacher pseudo-labels (conf≥0.5) | 0.0773 | -0.0090 | EXCLUDE | Single-teacher confirmation bias |
| 13 | T2 RT-DETR | RT-DETR-l retrain | 0.0684 | -0.0179 | EXCLUDE-S | Solo hurt, but **wins as ensemble member** |
| 14 | **E27** | **WBF(E1 + T2)**, equal weights | **0.0917** | +0.0054 | INCLUDE | Training-phase best — cross-arch ensemble |
| 15 | E28 consensus | E1+T2 consensus pseudo-labels (IoU≥0.5, conf≥0.7) | 0.0695 | -0.0168 | EXCLUDE | Multi-teacher consensus also hurts |
| 16 | E17 RFS | Repeat-Factor Sampling retrain | 0.0689 | -0.0174 | EXCLUDE | Class imbalance is not the bottleneck |
| 17 | E19 copy-paste | Class-aware copy-paste retrain | 0.0827 | -0.0036 | EXCLUDE-S | Smallest hurt of recent retrains; ensemble candidate |
| 18 | T3 hier-aux | Hierarchical aux head retrain | N/A | N/A | BLOCKED | Ultralytics 8.4.39 hyp-dict regression |
| 19 | E8 DANN | Domain-adversarial retrain | N/A | N/A | BLOCKED | Same Ultralytics regression |

### Phase 7 — Post-submission optimization

| # | Exp | Recipe | LB | Δ vs anchor | Verdict | Takeaway |
|---|---|---|---|---|---|---|
| 20 | E27 + P3 | E27 + laser-dot geometric suppression | 0.0924 | +0.0061 | INCLUDE | Cheap geometric filter; defensive baseline |
| 21 | **Slot 1** | E27+P3 + Path A consensus relabel (frozen DINOv2 + k-NN + MLP unanimous) | **0.1098** | +0.0235 | INCLUDE | **+0.0174 over E27+P3 — test-time relabel works** |
| 22 | Slot 2 | E27+P3 + k-NN only (no MLP consensus) | 0.0945 | +0.0082 | EXCLUDE | k-NN alone insufficient |
| 23 | Slot 4 | E27+P3 + MLP-only argmax (~50% relabel) | 0.0981 | +0.0118 | EXCLUDE | Aggressive relabeling overshoots |
| 24 | Slot 6 | Path B — MLP classifier on crops + TTA | 0.0622 | -0.0241 | EXCLUDE | Crop classification doesn't substitute for detection |
| 25 | Slot A STRICT | E27+P3 + FT DINOv2 + STRICT consensus | 0.1038 | +0.0175 | EXCLUDE-S | FT DINOv2 doesn't beat frozen; useful in WBF stack |
| 26 | Slot B 0.85 | E27+P3 + MLP-only (prob ≥ 0.85) | 0.1031 | +0.0168 | EXCLUDE-S | Useful in WBF stack |
| 27 | Slot D SSL | SSL t35 + YOLO fine-tune + multiscale TTA | 0.0504 | -0.0359 | EXCLUDE | End-to-end SSL failed |
| 28 | **Slot 8** | WBF(E27, Slot 1, Slot A, Slot B), asymmetric weights (1.5/0.5/0.7/0.7) | **0.1141** | +0.0278 | INCLUDE | **+0.0043 — WBF stacks peer relabel variants** |
| 29 | **Slot 9 ★** | **Slot 8 + letterbox-border noise filter** | **0.1143** | **+0.0280** | **INCLUDE ★** | **★ FINAL BEST. Drops 7,022 low-conf bar predictions** |
| 30 | Slot 10 | Slot 9 + pairwise size-violation relabel | 0.1086 | +0.0223 | EXCLUDE | Size-based MLP relabel too noisy |

</details>

## Key insights

### Institutional disjoint dominates everything

Train = 100% NOAA + SOI; test = 100% MBARI VARS. Different decades, different ROVs, different oceans. Every winning lever in this project — high-resolution training, multi-scale TTA, cross-architecture ensembling, **and the Phase 7 consensus relabel built on a separately-trained classifier** — is **domain-agnostic** or domain-orthogonal. Every losing lever fit tighter to the source domain. This is the load-bearing observation.

### Val mAP and LB mAP measure different distributions

Val mAP@[.50:.95] = 0.43 on fold-0 (SOI/NOAA holdout); pure-model LB = 0.09 on test (MBARI). The 8× ratio is structural, not a pipeline bug — verified by re-scoring with pycocotools using the competition's exact eval notebook (0.40706 ≈ Ultralytics' 0.40310). Val improvements that don't survive the domain shift are leaderboard losses.

### Consensus rules that pull from independent signal sources actually work

Pseudo-labeling between two YOLO detectors trained on the same source domain hurt (E28, -0.0168). Consensus relabeling between two DINOv2-feature-space classifiers (k-NN + MLP, trained on labeled train crops, applied to E27's box crops) lifted +0.0174. The difference isn't the consensus mechanism — it's whether the two voters are reading from the same domain-correlated representation. Two YOLO heads see MBARI through the same NOAA-trained lens; DINOv2 sees crops through a representation pretrained on a much broader image distribution.

### "Useful solo" and "useful in an ensemble" are different questions

T2 (RT-DETR-l) solo scored 0.0684, an 18-point regression from the anchor. As an ensemble member, T2 lifted the project to its training-phase best score. Slot A STRICT (LB 0.1038) and Slot B 0.85 (LB 0.1031) also hurt solo but lifted +0.0043 when WBF-stacked with Slot 1 and E27. Once the diversity question is decoupled from the standalone-strength question, the candidate pool of techniques worth running widens.

### Class imbalance is not what's costing us mAP

Three mechanistically independent attacks on the long tail (EQLv2 / RFS / copy-paste) all hurt the leaderboard. The 1000× class-count range across 32 classes is real, but it isn't the lever. The domain gap is.

## Final composition

The submitted pipeline is composed in two phases — model training (Phase 6, E27) and post-submission optimization (Phase 7, Slot 9):

```python
# Phase 6 — model training endpoint
E27 = WBF(
    E1 = YOLOv8x + MBARI 315k init + imgsz=1024 + 6-scale TTA WBF,
    T2 = RT-DETR-l + imgsz=1024 + 6-scale TTA WBF,
    weights = (1.0, 1.0),
    iou_thr = 0.55,
)                                                              # LB 0.0917

# Phase 7 — post-submission optimization on top of E27
# (a) Path A consensus relabel — class-relabel E27 boxes where two
#     DINOv2-feature classifiers (k-NN + MLP) unanimously disagree
slot_1 = path_a_consensus_relabel(
    base       = E27 + P3,                  # P3 = laser-dot suppression
    features   = DINOv2_ViT_L14_frozen,
    classifiers= [kNN, MLP_32way],
    rule       = "unanimous_disagreement",  # ~3,062 relabels
)                                                              # LB 0.1098

# (b) WBF stack of peer relabel variants — asymmetric weights
slot_8 = WBF(
    [E27, slot_1, slot_a_strict, slot_b_085],
    weights = [0.5, 1.5, 0.7, 0.7],
    iou_thr = 0.55,
    skip_box_thr = 1e-4,
) + P3                                                         # LB 0.1141

# (c) Letterbox-border noise filter — drop low-conf preds in black bars
slot_9 = letterbox_border_filter(
    slot_8,
    mean_rgb_thresh = 15,   # interior pixels must average > 15 RGB
    min_depth       = 5,    # bar must be ≥ 5 px deep
)                                                              # LB 0.1143 ★
```

**Public LB: 0.1143 mAP@[.50:.95].** Competition leader scored 0.321; gap = 0.207.

Per-class metrics on the proxy validation set (`figures/per_class_metrics.csv`) show wide AP variance: pyrosome, larvacean, and calycophoran siphonophore are near 1.0 AP; benthic worm, anemone, and brittle star sit in the 0.4–0.5 range. The long-tail shape of the per-class AP distribution mirrors the per-class instance count distribution — high-frequency classes are not necessarily high-AP, and rare classes with strong visual signatures (pyrosomes, larvaceans) score well even with few train examples.

## Limitations and what we'd do with more time

- **Re-run the two blocked experiments** once the Ultralytics 8.4.39 hyp-dict regression is patched. DANN (feature-level domain invariance via GRL on the backbone) and the hierarchical aux head (semantic regularization via taxonomic predictions) both attack the domain gap from angles not covered by the winning levers. The fix is ~30 minutes of code; the retrains are each ~3 hours of GPU.
- **SAHI tile-based inference at imgsz=1920** is implemented (`src/sahi_inference.py`) but never made it onto the leaderboard. The 14% of test images at 720×486 are particularly exposed to small-object recall failures — and the border filter already shows that the letterboxed scale is its own regime.
- **Generalize Phase 7 to a 3rd ensemble member.** A real 3-model E27 (E1 + T2 + a fixed T3 hierarchical head) under the same Path A consensus relabel is the natural next step; the +0.0054 gain from 1 → 2 detectors plus +0.0226 from the relabel chain suggests compounding is plausible.
- **Background-swap augmentation** (paste detected ROV equipment from training images onto MBARI test backgrounds for augmented training) was pre-generated to 68% before a tmux crash interrupted it. Re-doing this with the equipment crops already extracted in `data/equipment_verification/` is cheap if the ensemble path stalls.

## Tech stack

- **PyTorch** + **Ultralytics YOLO11** (anchor model, YOLOv8x backbone)
- **RT-DETR** (cross-architecture ensemble member)
- **MBARI 315k** pretrained FathomNet weights (chosen via 5-way bake-off)
- **DINOv2 (ViT-L/14)** frozen features for Phase 7 crop classification
- **Weighted Boxes Fusion** for multi-model + multi-scale aggregation
- **SAHI** for tile-based small-object inference (implemented; not in final pipeline)
- **Weights & Biases** for experiment tracking
- Custom modules in `src/`: PU loss (Kiryo), EFL, EQLv2, federated loss, hierarchical aux head, DANN, soft teacher, RFS, class-aware copy-paste, underwater preprocessing, context rescorer

## Repository

| Path | Contents |
|---|---|
| [`SETUP.md`](SETUP.md) | Install + how to run the demo notebook, training, and inference |
| [`notebooks/final_pipeline.ipynb`](notebooks/final_pipeline.ipynb) | End-to-end demo of the final Slot 9 pipeline, **run with cached outputs** |
| `src/` | Loss functions, augmentations, domain-adaptation modules, submission utilities |
| `scripts/` | Training, inference, ensemble, and evaluation entry points |
| `tests/` | Unit tests for `src/` modules |
| `configs/` | YOLO dataset config + class taxonomy |
| `figures/` | Curated figures referenced from this README |
| `data/`, `weights/` | Folder placeholders (gitignored) — see their READMEs for download instructions |

## License

Code: see [`LICENSE`](LICENSE) (Apache 2.0). Pretrained weights are subject to their upstream licenses (Ultralytics AGPL-3.0, MBARI 315k CC-BY 4.0).
