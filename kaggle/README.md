# Running the low-data study on Kaggle (4 accounts)

All heavy work (landmark extraction, the sweeps) runs on Kaggle, and each of the four team
members runs it on their own account. Everything goes through one notebook,
`kaggle/lowdata_study.ipynb`, controlled by two variables: `STAGE` and `MEMBER` (0–3).

## 0. Once per account

* Import the notebook: Kaggle → Create → Notebook → File → Import.
* Settings: Accelerator **GPU T4 ×2**, Internet **on**.
* Code: either attach a private dataset `isl-lowdata-slr-code` holding this repository, or
  set `REPO_URL`/`BRANCH` to the pushed branch.
* Hugging Face data that is gated or private (CISLR, and the 40-word corpus if private):
  Add-ons → Secrets → `HF_TOKEN`. CISLR also needs its terms accepted on Hugging Face first.

## 1. Landmark extraction: sharded by clip

| who | setting | output |
|---|---|---|
| member *i* | `STAGE="extract"`, `MEMBER=i`, `EXTRACT_SOURCE="include"` | `stores/include_shard<i>` |
| member 0 (small) | `EXTRACT_SOURCE="isl40"`, `MEMBERS=1` | `stores/isl40_shard0` |

* INCLUDE shards are whole zips, largest first and dealt round-robin, so each member downloads about 1/4 of the ~50 GB.
* Other corpora shard by a stable hash of the clip key.
* A session that runs out of time can be continued: attach the notebook's own previous output as an input and rerun. Finished clips are skipped.
* Then one member runs `STAGE="merge"` with the four extract outputs attached. The merged `stores/` output is then saved as a **private** dataset `isl-lowdata-stores`.

MediaPipe is installed into `/tmp/mp` and is visible only to the extraction subprocess, so the notebook's torch and numpy are not changed.

## 2. Sweeps: split by K (training videos per word)

Every member runs the same grid with `STAGE="sweep"`, `MEMBER=i`, `MEMBERS=4`.

* **How the split works.** `sweep.py` groups the jobs by K and deals the groups to members by estimated GPU time. The rule is deterministic, so no coordination is needed.
* **Checking the split before starting.** Anyone can run it locally without a GPU:

  ```
  python lowdata.py sweep --grid islr/lowdata/configs/scarce_legacy8.json --plan --members 4
  ```

* **Fixing the assignment by hand.** Put an `"assign": {"0": [1, 16], "1": [2], "2": [4], "3": [8]}` entry in the grid file.
* **How the two T4s are used.** Each T4 runs its own queue of whole runs. Low-data runs are too small to benefit from DDP. Grids marked `"ddp": true` (for example `pretrain_full.json`) use `torchrun` over both GPUs instead.
* **Resuming after the 12 h limit.** Runs checkpoint about 10 minutes before the limit and the notebook prints `NOT FINISHED`. Save the version, attach its output as an input, and run again: finished runs are skipped and interrupted runs resume from their checkpoint.

| grid | question | jobs |
|---|---|---|
| `scarce_legacy8.json` | 8 words with K clips alongside a well-resourced vocabulary, for K = 1…16 | 18 models × 5 K × 3 seeds |
| `uniform_k.json` | every word has K clips | 9 × 6 × 3 |
| `vocab.json` | how vocabulary size affects the scarce words | 9 × 2 × 3 × 5 |
| `cross_source.json` | domain shift between corpora | 9 × 4 × 3 |
| `pretrain_full.json` | all data with DDP; checkpoints can be used for `init_from` | 3 |

## 3. Report

`STAGE="report"` with the members' final sweep outputs attached writes `summary.md`, `paired.md`, `checks.md` and `curve_*.png`.

* `checks.md` confirms that every model saw the same test clips for each (protocol, seed).
* It also lists any grid cells that are missing seeds.

## Data rules

* Keep the stores dataset **private**. It contains landmarks derived from INCLUDE (CC BY 4.0), the ISLRTC dictionary and the team's own recordings.
* Do not add GISLR (Kaggle competition data) to any dataset. Load it only from the competition input.
* Never put `kaggle.json` or tokens in the repository. Use Kaggle Secrets.
