# Running the low-data study on Kaggle (4 accounts)

All heavy work (landmark extraction, the sweeps) runs on Kaggle, and each of the four team
members runs it on their own account. Everything goes through one notebook,
`kaggle/lowdata_study.ipynb`, controlled by `STAGE` (`member` or `report`) and `MEMBER` (0–3).

**Nobody waits for anyone.** Each member builds the data on their own, then runs their share of
the sweep. The only step that needs everyone is the final report.

```
member i:  data (own copy, minutes) ──► sweep (share i of each grid) ──► output
                                                                            │
anyone:    report  ◄── all four outputs attached ◄──────────────────────────┘
```

## 0. Once per account

* Import the notebook: Kaggle → Create → Notebook → File → Import (upload `lowdata_study.ipynb`).
  You don't need to clone anything: the notebook clones `REPO_URL`/`BRANCH` itself. Alternatively,
  it uses a private dataset `isl-lowdata-slr-code` holding this repository, if one is attached.
* Settings: Accelerator **GPU T4 ×2**, Internet **on**.
* Only for gated or private Hugging Face data (CISLR, or the 40-word corpus if you make it private):
  Add-ons → Secrets → `HF_TOKEN`. CISLR also needs its terms accepted on Hugging Face first.

## 1. `STAGE="member"`: data, then sweep

Set `MEMBER` to your number, keep `MEMBERS = 4` and the same `GRIDS` as everyone else, then
*Save Version → Save & Run All*.

**Data.** `lowdata.py data` gets every store the grids list into `/kaggle/working/stores`. For each store:

1. keep it if it is already complete;
2. otherwise copy it from `/kaggle/input`, if a complete copy with the same data version is attached;
3. otherwise build it.

The default grids use two small stores, and every member builds them in about 20 minutes:

* `isl40`: the 40-word corpus (`vidit031/isl-isolated-40words` on Hugging Face, pinned to one commit
  in `"data": {"isl40": {"revision": ...}}`). ~640 clips, downloaded, then MediaPipe on the CPUs.
* `include_words`: INCLUDE's Brother and I clips (42 clips, ~0.55 GB), renamed to the corpus's
  thin words brother and me. Each clip is read straight out of its zip on Zenodo with
  HTTP byte ranges, so the category zips (9 GB) are never downloaded. That's under 10 minutes at the
  observed ~1.5 MB/s. The word mapping is in `"data": {"include_words": {"words": ...}}`.

Attaching a teammate's output only saves those minutes.

* INCLUDE (only for the `scarce_legacy8*` grids) is ~57 GB from a slow Zenodo server. Each member
  starts on a different set of zips. If you attach teammates' partial outputs, their stores are
  merged first and the zips they finished are skipped.
* Every complete store records its data version and a fingerprint of its clip keys. They are
  written to `sweeps/_sweep/data_member<i>.json`. The report checks that all members have the same
  fingerprint, which guarantees identical train/test splits.
* MediaPipe is installed into `/tmp/mp` and only the data subprocess sees it, so the notebook's
  torch and numpy are not changed.

**Sweep.** For each grid in `GRIDS`, in order, this member's share runs on both T4s, with results in
`/kaggle/working/sweeps`.

* **How the split works.** Jobs are grouped by K (training videos per word) and the groups are
  dealt to members by estimated GPU time. The rule is deterministic, so no coordination is needed.
  Check it locally without a GPU:

  ```
  python lowdata.py sweep --grid islr/lowdata/configs/isl40_scarce.json --plan --members 4
  ```

  To fix it by hand, put `"assign": {"0": [1], "1": [2], "2": [4], "3": [8]}` in the grid file.
* **How the two T4s are used.** Each T4 runs its own queue of whole runs, because low-data runs are
  too small for DDP. Grids marked `"ddp": true` use `torchrun` over both GPUs instead.
* **Shared jobs.** A job that appears in two grids (same protocol, model, K, seed) runs once.
* **Resuming after the 12 h limit.** When the notebook prints `NOT FINISHED`, save the version, attach
  its output (Add Input → Your Work → this notebook) and run again. The stores are copied back,
  finished runs are skipped, and interrupted runs resume from their checkpoint.

**Sharing the stores (optional).** Your output's `stores/` folder is a ready-made store. To let
teammates skip the data step, share the notebook with them, or make a **private** dataset from its
output. Attaching it is optional for them.

| grid | question | jobs (default) |
|---|---|---|
| `isl40_scarce.json` | the 8 legacy words have K clips each, next to a well-resourced 40-word-corpus vocabulary | 18 models × K{1,2,4,8} × 3 seeds = 216 |
| `isl40_uniform.json` | every word has K clips | 9 × 4 × 3 |
| `isl40_vocab.json` | how vocabulary size affects the scarce words | 9 × 2 × 3 × 3 |
| `scarce_legacy8.json`, `uniform_k.json`, `vocab.json`, `cross_source.json` | the same questions with INCLUDE and other corpora (needs the INCLUDE store) | see the file |
| `pretrain_full.json` | all data with DDP; checkpoints can be used for `init_from` | 3 |

The grids fix the vocabulary at **30 words** (`base.words`): the corpus's words with at least 16
clips and at least 4 signers, including brother and me topped up from INCLUDE. Left out are the 8 thin
words (sorry, come, stop, goodbye, read, write, stand, when) and home and hospital (2 signers each).
After the test split, each rich word has about 10 training clips, so K stops at 8.

### Adding the team's own recordings

For the thin words (stand, when, goodbye, read, write, come, stop, sorry), no open dataset has more
than one or two clips. To add them later, extend `base.words` in the grids. FDMSE-ISL (RKMVERI) has about 20
per word for all of them except goodbye. Access is by request with an institutional email, and its
terms forbid redistribution, so those clips may only go into a private store, never the public
Hugging Face corpus. Otherwise, record them:

1. Record them: several signers, a few repetitions each, framed from the head to the waist.
   Name each signer `team_<name>` in `metadata.csv` (this keeps them as a separate identity for the
   signer-independent test split).
2. Add the clips to the Hugging Face dataset. This creates a new commit.
3. Put that commit in `"data": {"isl40": {"revision": "<new commit>"}}` in **all** grids, and push.
   Stores from the old commit are then not reused, so every member rebuilds from the same new data.

## 2. `STAGE="report"`

Attach the members' final outputs (their notebooks must be shared with you, or published as private
datasets). The report writes `summary.md`, `paired.md`, `checks.md` and `curve_*.png`.

* `checks.md` confirms that every model saw the same test clips for each (protocol, seed).
* It also confirms that every member's data had the same fingerprint.
* It lists any grid cells that are missing seeds.

A member can also run the report on their own output alone; the notebook does this at the end
of each `member` run.

## Data rules

* Keep any stores dataset **private**. It contains landmarks derived from the 40-word corpus,
  INCLUDE (CC BY 4.0), the ISLRTC dictionary and the team's own recordings.
* Do not add GISLR (Kaggle competition data) to any dataset. Load it only from the competition input.
* Never put `kaggle.json` or tokens in the repository. Use Kaggle Secrets.
