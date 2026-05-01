# Kaggle Discussion-board declaration — external models / data

**Status: DRAFT, do not post until reviewed.**
**Block:** B.0.5 — HARD GATE before any leaderboard submission.
**Where to post:** https://www.kaggle.com/competitions/fathomnet-2026/discussion (new thread, prefix title with `[External Data Declaration]`)

---

## Why this exists

The official Kaggle rules (Section 7, "External Data") say:

> If you use external data or pretrained models, you must disclose them publicly on the discussion forum **before** the entry deadline.

Failing to declare = automatic disqualification per the rules. Two pretrained models are baked into my pipeline, so I file ONE post that covers both, before my first leaderboard submission.

If I add anything later (a new ImageNet checkpoint to compare, an extra augmentation that uses pretrained weights, ANYTHING), I EDIT this same thread to keep one canonical record.

---

## Title

```
[External Data Declaration] navya2392 — YOLOv11 (COCO-pretrained) + BioCLIP2 (TreeOfLife-10M)
```

## Body (copy-paste, edit name/team)

```markdown
Hi all,

Per the rules, declaring my external pretrained models up front. I am
NOT using any external image data — only the official FathomNet 2026
train + test splits.

### 1. Ultralytics YOLOv11 with COCO pretraining

- Repo: https://github.com/ultralytics/ultralytics  (v8.4.39)
- Pretrained checkpoint: `yolo11{n,s,m,l,x}.pt` (auto-downloaded by Ultralytics)
- Pretraining data: MS-COCO 2017 (Microsoft, CC-BY 4.0; widely used in object
  detection literature, available to all Kaggle competitors)
- License: AGPL-3.0 (Ultralytics)
- Why: standard, fast, well-tested baseline for COCO-format detection

### 2. BioCLIP2 backbone (replacing the default CSP-Darknet/ResNet)

- Repo: https://github.com/Imageomics/bioclip
- Checkpoint: HuggingFace `imageomics/bioclip-2`
- Pretraining data: TreeOfLife-10M — 10M labeled images of organisms across
  the tree of life (curated from iNaturalist, EOL, BIOSCAN, etc.)
- License: MIT (model weights), CC-BY-NC 4.0 (some training images, but the
  WEIGHTS are MIT and the weights are what I use)
- Why: features pretrained on biology imagery transfer better to marine
  organisms than ImageNet-pretrained features

### 3. No additional external image data

- Training images: official `train_dataset.json` only (6,463 images)
- Test images: official `test_dataset.json` only (1,425 images)
- I am NOT using FathomNet's broader public image archive, iNaturalist
  scrapes, web-scraped marine imagery, or any unlabeled data outside the
  competition.

If I add anything later I'll EDIT this post.

Thanks,
[YOUR NAME / TEAM]
```

---

## What to do AFTER posting

1. Copy the public Kaggle URL of the post into `master_checklist.txt` next to the B.0.5 line.
2. Mark B.0.5 as `[x] done` with the URL.
3. Until then, **do not** click "Submit" on any leaderboard probe — a single submission before declaration could be grounds for DQ at audit time.

## Edits to make if I add anything

If during Phase 5/6 I decide to use:
- **An additional pretrained model** (e.g. SAM2 for refining boxes, DINOv2 for features) → add a section 4, edit the post.
- **Test-time augmentation that imports a 3rd-party model** → declare it.
- **Any pretrained weights even if not used in the final ensemble** → declare them anyway, the rule says "use", and any weights touching my training pipeline count.

I can ALWAYS edit the post freely up to the entry deadline. The cost of an extra disclosure is zero; the cost of a missed disclosure is disqualification.
