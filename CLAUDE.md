# CLAUDE.md

Low-data isolated ISL recognition paper code. Study design and status: `docs/study_design.md`. Baseline architectures: `docs/baselines.md`. KDF history: `docs/kdf_architecture.md`.

## Commands (run from the repo root)
- Install: `pip install -r requirements.txt` (optionally `pip install -e .`)
- End-to-end check (the only "test"): `python lowdata.py smoke --fast`. Full version: `python lowdata.py smoke` (~15 min CPU). Skip steps with `--skip ddp sweep ...`, and pick models with `--models partformer kdf_transformer`.
- One study run: `python lowdata.py run --stores S1 S2 --model partformer --protocol scarce --shots 4 --seed 0 --out runs`
- Sweep plan: `python lowdata.py sweep --grid islr/lowdata/configs/scarce_legacy8.json --plan --members 4`
- One member's data (builds, copies or merges the stores a grid lists; no waiting on teammates): `python lowdata.py data --grid islr/lowdata/configs/isl40_scarce.json --dest stores --inputs /kaggle/input`. Kaggle flow: `kaggle/README.md`.
- Legacy few-shot: `python -m islr.fewshot.train --list`, then `python -m islr.fewshot.train --data-dir ISL_DATASET_40WORDS --models stgcn --epochs 5`
- No linter config and no pytest suite.

## Conventions / gotchas
- Imports are absolute (`from islr.models.registry import ...`). Entry points add the repo root to `sys.path`, so they work without `pip install -e .`.
- A new model needs a builder + `BaselineSpec` in `islr/models/registry.py`. Non-model config keys must be in `TRAIN_KEYS`, or they get passed to the constructor. The study runner resolves models through the same registry.
- `kdf_stgcn` is a registry alias for `kdf_transformer` (the file is `islr/models/kdf.py`).
- The study (`islr/lowdata`) uses a fixed step budget with no validation or early stopping. The legacy `islr/fewshot` loop does early-stop on a tiny validation set. Don't mix their results.
- No horizontal flips in augmentation, because a flip swaps the dominant hand.
- Grid JSONs read store paths as `$STORES/...`, so set `STORES`.
- `islr/common/__init__.py` defines output dirs (`outputs/cache|weights|checkpoints`). Datasets sit at the repo root (`ISL_DATASET*`, gitignored).
- `dataset/scripts/*` resolve paths relative to `dataset/`. Only needed to rebuild the HF corpus.
- GISLR data must never be committed or redistributed. CISLR is gated (needs `HF_TOKEN`).
