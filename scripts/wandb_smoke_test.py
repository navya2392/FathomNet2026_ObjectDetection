"""W&B authentication + logging smoke test.

Verifies that:
  1. wandb is installed
  2. WANDB_API_KEY env var (or ~/.netrc) is valid
  3. The configured entity/project is reachable
  4. Both config and per-step metric logging work end-to-end

Run anywhere we'll be training (local laptop, RunPod, Kaggle) to confirm
W&B works BEFORE kicking off a multi-hour training job.

Usage:
    python scripts/wandb_smoke_test.py

Expected output: a wandb run URL like
    https://wandb.ai/navya2392-usc/fathomnet-2026/runs/<random_id>
Click it. You should see a live dashboard with `acc` going up and
`loss` going down across 8 steps.

After the smoke test passes, you can delete the run from the W&B UI
(Settings -> Delete run) to keep the project tidy.

Sourced from the W&B Quickstart sample, adapted with our entity/project.
"""
from __future__ import annotations

import random

import wandb

WANDB_ENTITY = "navya2392-usc"
WANDB_PROJECT = "fathomnet-2026"


def main() -> None:
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name="auth-smoke-test",
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )

    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        run.log({"acc": acc, "loss": loss})

    run.finish()
    print(f"OK -- run logged to https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}")


if __name__ == "__main__":
    main()
