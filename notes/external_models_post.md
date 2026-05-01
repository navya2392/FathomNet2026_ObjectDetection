# Kaggle Discussion-board declaration — external models / data

**Status:** REVISED May 1, 2026 ~00:25 PT — adds marine-specific
pretrained models per host clarification (see "Why this version exists" below).
**Block:** B.0.5 — HARD GATE before any leaderboard submission that uses
a NEWLY-DECLARED model below. The original Apr 30 post (declaring YOLOv11
+ BioCLIP2) is sufficient for any Phase 2 submission that uses ONLY those
two. To use any of the new models in Section 4-8 below, **EDIT the
existing Kaggle thread with the additional declarations BEFORE submitting
any prediction generated using them.**
**Where to edit:** the existing Kaggle Discussion thread you posted on
Apr 30 ~23:45 PT (`External Data Declaration — navya2392`).

---

## Why this version exists

On the Kaggle Discussion board, host Laura Chrobak responded to a question
from user `robbrock` clarifying Rule 6 ("External Data and Tools"). Key
points from the host's response (paraphrased; see the original thread for
verbatim language):

1. "Classification" in Rule 6 covers **object detection** (any task-specific
   training for this problem).
2. **FathomNet Megalodon model — ALLOWED** as a starting point (similar
   to ImageNet pretraining). Disclosure required.
3. **Other public marine-life detection models (e.g. Megafishdetector) —
   ALLOWED** for initialization or feature extraction if publicly available.
4. **Additional FathomNet imagery (even unlabeled) — NOT ALLOWED.** Cannot
   augment the dataset, even by stripping labels.
5. **Relabeling provided data — ALLOWED.** Self-pseudo-labeling on the
   competition train set is permitted (validates our Soft Teacher path).

This UNLOCKS our use of any publicly available marine-specific pretrained
detector as a starting checkpoint. Major strategic upgrade — these models
were trained on FathomNet imagery so their weights are vastly closer to
our target distribution than generic COCO/ImageNet weights.

Filing this updated declaration BEFORE any submission that uses a model
listed in Section 3+ below.

---

## Title (use the existing thread)

```
External Data Declaration — navya2392
```

(Edit the existing Kaggle thread; do not start a new one. One canonical
record per the host's preferred convention.)

## Body to paste into the EDIT (replaces existing body)

```markdown
Hi all,

UPDATE (May 1): Adding marine-specific pretrained detectors per Laura's
clarification of Rule 6 (Megalodon and similar marine-life detectors are
allowed for initialization/feature extraction with disclosure).

Per the rules, here is my full external pretrained model list. I am NOT
using any external image data — only the official FathomNet 2026
train + test splits.

### 1. Ultralytics YOLOv11 with COCO pretraining

- Repo: https://github.com/ultralytics/ultralytics  (v8.4.39)
- Pretrained checkpoint: `yolo11{n,s,m,l,x}.pt` (auto-downloaded by Ultralytics)
- Pretraining data: MS-COCO 2017 (Microsoft, CC-BY 4.0)
- License: AGPL-3.0 (Ultralytics)
- Use: Phase 2 baseline anchor + one of several candidate inits in
  Phase 2 EXP 2.3 multi-init bake-off

### 2. BioCLIP2 backbone (replacing the default CSP-Darknet/ResNet)

- Repo: https://github.com/Imageomics/bioclip
- Checkpoint: HuggingFace `imageomics/bioclip-2`
- Pretraining data: TreeOfLife-10M (curated biology images from
  iNaturalist, EOL, BIOSCAN, etc.; NOT FathomNet)
- License: MIT (model weights)
- Use: feature-extraction backbone via additive adapter; one of several
  candidate inits in Phase 2 EXP 2.3 multi-init bake-off

### 3. FathomNet Megalodon detector

- Card: https://huggingface.co/FathomNet/megalodon
- Checkpoint: `mbari-megalodon-yolov8x.pt` (HuggingFace `FathomNet/megalodon`)
- Architecture: Ultralytics YOLOv8x
- Pretraining data: ALL publicly-available FathomNet localizations (single
  "object" class — generic salient-object detector for marine imagery)
- Maintained by: MBARI (Monterey Bay Aquarium Research Institute)
- License: see model card (CC-BY 4.0 per FathomNet's standard practice;
  will pin license link in working notes paper)
- Use: candidate init for fine-tuning on the 32 FathomNet 2026 classes

### 4. FathomNet MBARI 315k detector

- Card: https://huggingface.co/FathomNet/MBARI-315k-yolov8
- Checkpoint: `mbari_315k_yolov8.pt`
- Architecture: Ultralytics YOLOv8
- Pretraining data: 315,000 MBARI deep-sea benthic images with
  taxonomic-level localizations (multi-class detector; broader than the
  32 classes in this competition but same data distribution)
- Maintained by: MBARI / FathomNet team
- License: see model card
- Use: candidate init for fine-tuning. STRONGEST EXPECTED CANDIDATE because
  it is multi-class and trained on the same MBARI distribution as the
  competition test imagery.

### 5. Megafishdetector

- Card: https://huggingface.co/FathomNet/megafishdetector  (mirror)
- Repo: https://github.com/warplab/megafishdetector
- Checkpoint: `megafishdetector_v0_yolov5m_1280p`
- Architecture: Ultralytics YOLOv5m
- Pretraining data: AIMs Ozfish, FathomNet (subset), VIAME FishTrack,
  NOAA Puget Sound Nearshore Fish, DeepFish, NOAA Labelled Fishes in
  the Wild — all publicly-available datasets
- License: see repo
- Use: candidate init for fine-tuning, especially helpful for the few
  fish classes in our 32-class set. Also available as an ensemble
  member if it adds diversity.

### 6. Open-vocabulary detection models (planned for Phase 5+ stretch goal)

The following may be added during Phase 5-7. They are listed here in
advance per the rule that disclosure must precede use. If we do not end
up using them, no harm done; if we do, we are already covered.

- **GroundingDINO** — IDEA-Research (https://github.com/IDEA-Research/GroundingDINO)
  open-vocabulary detector trained on Objects365, GoldG, V3Det.
  License: Apache 2.0. Use: zero-shot detection of rare classes
  (sea slug, isopod) by class-name prompting.
- **OWL-ViT** — Google Research (https://huggingface.co/google/owlvit-large-patch14)
  open-vocabulary ViT-based detector trained on COCO + LVIS + OpenImages.
  License: Apache 2.0. Use: same as GroundingDINO; alternative or ensemble.

### 7. No additional external image data

- Training images: official `train_dataset.json` only (6,463 images)
- Test images: official `test_dataset.json` only (1,425 images)
- I am NOT using FathomNet's broader public image archive, iNaturalist
  scrapes, web-scraped marine imagery, or any unlabeled data outside the
  competition.
- "Pseudo-labels" generated by Soft Teacher (Phase 3 EXP 3.5) are derived
  from MY OWN MODEL'S predictions on the COMPETITION TRAIN images, which
  Laura confirmed is allowed (relabeling provided data is OK).

If I add anything later I'll EDIT this post.

Thanks,
navya2392
```

---

## What to do AFTER editing the Kaggle post

1. Copy the public Kaggle URL of the (updated) post into `master_checklist.txt`
   next to the B.0.5 line if not already there.
2. The B.0.5 line already says `[x] done`. After this edit goes live, also
   note the timestamp of the edit in B.0.5 for audit-trail completeness.
3. Until the EDIT is published, **do not** click "Submit" on any
   leaderboard probe that uses Megalodon, MBARI 315k, Megafishdetector,
   GroundingDINO, or OWL-ViT. The original post DOES cover any submission
   using only YOLOv11 + BioCLIP2, so vanilla Phase 2 baselines (item C.4
   in checklist) are fine to submit immediately.

## Future additions

The post can be edited freely up to the entry deadline. Cost of an extra
disclosure is zero; cost of a missed disclosure is disqualification.
Always update HERE first (this draft) for clean version control, then
sync the change to the live Kaggle thread.

If during Phase 5-7 you decide to use:
- **An additional pretrained model** (e.g. SAM2 for refining boxes,
  DINOv2 for features, RT-DETR-X for ensemble diversity) → add a new
  numbered section to this draft, edit the Kaggle post.
- **Test-time augmentation that imports a 3rd-party model** → declare it.
- **Any pretrained weights even if not used in the final ensemble** →
  declare them anyway. The rule says "use" and any weights touching
  the training pipeline count.
