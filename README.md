# FathomNet 2026

Positive-unlabeled object detection on marine imagery.

- **Competition:** https://kaggle.com/competitions/fathomnet-2026
- **Deadline:** May 7, 2026 (11:59 PM CET)
- **Plan:** see `docs/fathomnet_2026_master_plan_v3.pdf` (or `.md`)
- **Backup ideas:** see `docs/fathomnet_2026_backup_extras_v2.pdf` (or `.md`)

## Status

- [x] CLEF registration
- [x] Cursor setup
- [x] Project structure
- [x] Python environment (`venv/`, Python 3.11)
- [x] Kaggle API authenticated (`KAGGLE_API_TOKEN` env var) + competition rules accepted
- [ ] Kaggle data files downloaded to `data/raw/kaggle/`
- [ ] FathomNet images downloaded via `download.py` (~50 GB)
- [ ] Phase 0 EDA

## Quickstart

```powershell
# Activate the venv (PowerShell)
.\venv\Scripts\Activate.ps1

# Verify Kaggle auth
kaggle competitions files fathomnet-2026
```

If `KAGGLE_API_TOKEN` is not set in this shell:

```powershell
[Environment]::SetEnvironmentVariable("KAGGLE_API_TOKEN", "KGAT_your_token", "User")
# then close + reopen the terminal
```

Copy `.env.example` to `.env` and fill in any other secrets (W&B, HF) as needed.
