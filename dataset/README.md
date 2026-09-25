# Building the 40-word ISL corpus

These scripts built `vidit031/isl-isolated-40words` (642 clips, 40 glosses) from ISL500, INCLUDE, CISLR and the ISLRTC dictionary. They also produced the reports used in the paper's data section.

Run them from this folder. Paths resolve relative to `dataset/`, and they write `raw_datasets/`, `cache/`, `logs/` and `ISL_DATASET/` here (all gitignored).

| file | role |
|---|---|
| `scripts/isl_dataset_agent.py` | Full pipeline: search sources, download, normalise glosses, deduplicate, materialise `ISL_DATASET/`, write reports. |
| `scripts/finalize_local.py` | Rebuilds the corpus and reports from raw videos already downloaded, with no network access. |
| `scripts/download_include_*.py`, `_check_include_zips.py` | INCLUDE (Zenodo 4010759) zip downloads, in different orders. |
| `scripts/extract_cislr_low_words.py`, `watch_extract_people.py` | CISLR and INCLUDE extraction for words with few clips. |
| `scripts/upload_to_hf.py` | Pushes `ISL_DATASET/` to the Hugging Face Hub. |
| `config/` | Target vocabulary (`target_words.json`, `words.txt`) and the 8-word test list. |
| `reports/` | Snapshot of the build outputs: inventory, licences, duplicates, gloss normalisation, per-word statistics, `metadata.csv`, `summary.md`. |

```bash
cd dataset
python scripts/isl_dataset_agent.py      # or: python scripts/finalize_local.py
```

The low-data study does **not** need this. It ingests the published corpus through `python lowdata.py sources isl40`.
