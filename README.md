# Low-data Isolated Sign Language Recognition (ISL)

Code for the paper on **low-resource sign language recognition, using Indian Sign Language (ISL) as a proxy**. The paper asks three questions:
- How does accuracy scale with K, the number of training clips per word?
- What happens to scarce words inside a larger vocabulary?
- Which inductive biases pay off when K is tiny? The candidates are graphs, spectral features, Koopman/DMD dynamics, part-factorised encoders and prototype heads.

Every comparison is signer-independent, with a fixed test set per seed.

Study design, protocols, compute plan and paper outline: [`docs/study_design.md`](docs/study_design.md).

## Layout

| path | what |
|---|---|
| `islr/models/` | All model architectures and the registry (`registry.py`): 11 baselines, **KDF** (Hankel-DMD + class-wise Koopman head), the **PartFormer** family, and ablations. |
| `islr/lowdata/` | **Main study.** Landmark stores, corpus ingestion, signer-independent protocols, the fixed-budget runner, sweeps, and the report with paired tests. Grids are in `configs/`. |
| `islr/fewshot/` | The earlier 8-word, 7-shot experiment: locked test split, train/eval/report. |
| `islr/common/` | Output paths, landmark extraction/cache, and the train/eval engine. |
| `lowdata.py` | Launcher for the study: `extract`, `sources`, `inspect`, `run`, `sweep`, `report`, `smoke`. |
| `kaggle/` | The single notebook that runs extraction and sweeps across 4 Kaggle accounts, with its guide. |
| `scripts/` | Hugging Face download, the cloud pipeline for the few-shot experiment, and a protocol check. |
| `dataset/` | How the 40-word corpus was built (scripts, word lists, licence and duplicate reports). See [`dataset/README.md`](dataset/README.md). |
| `docs/` | `baselines.md` (architecture and citations for each baseline), `kdf_architecture.md` (KDF revision history and ablations), `study_design.md`. |
| `outputs/` | Everything generated: caches, weights, stores, sweeps. Gitignored. |

## Setup

```bash
pip install -r requirements.txt
pip install -e .          # optional; every entry point also works from the repo root without it
```

## Main study (low-data)

```bash
python lowdata.py smoke --fast                    # synthetic end-to-end check, a few minutes on CPU
python lowdata.py sources isl40 --store outputs/stores/isl40
python lowdata.py inspect --stores outputs/stores/isl40 --grid islr/lowdata/configs/scarce_legacy8.json
python lowdata.py sweep --grid islr/lowdata/configs/scarce_legacy8.json --plan --members 4
python lowdata.py report --out sweeps --dest outputs/report
```

Grid files read store paths as `$STORES/<name>`, so set `STORES` (for example `STORES=outputs/stores`). The full runs are done on Kaggle; see [`kaggle/README.md`](kaggle/README.md).

## Earlier few-shot experiment (8 words, 7-shot)

```bash
python scripts/download_hf_dataset.py --repo vidit031/isl-isolated-40words --out ISL_DATASET_40WORDS
python -m islr.fewshot.extract_landmarks --num-frames 30 --data-dir ISL_DATASET_40WORDS
python -m islr.fewshot.train --data-dir ISL_DATASET_40WORDS --n-words 8 --draws 3 --train-shots 7 --test-per-class 15
python -m islr.fewshot.eval  --data-dir ISL_DATASET_40WORDS --models all
python -m islr.fewshot.train --list                # registered models
```

Or run everything in one go on a cloud GPU: `bash scripts/run_pipeline_baselines.sh`. Weights, metrics and `baselines_report.md` are written to `outputs/weights/`.

## Data

The videos are not stored in this repo.

| corpus | where |
|---|---|
| 40-word ISL corpus (642 clips) | HF `vidit031/isl-isolated-40words` |
| 8-word smoke subset (56 clips) | HF `vidit031/isl-isolated-8words` |
| INCLUDE, ISLRTC dictionary, CISLR, GISLR | fetched by `lowdata.py sources ...` (licences are listed in `docs/study_design.md` §4) |
