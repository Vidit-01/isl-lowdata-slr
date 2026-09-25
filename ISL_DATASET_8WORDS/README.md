---
pretty_name: ISL Isolated Word Dataset (8 words, test subset)
license: other
license_name: multi-source-aggregate
license_link: https://huggingface.co/datasets/vidit031/isl-isolated-8words
task_categories:
  - video-classification
language:
  - en
  - sgn
tags:
  - indian-sign-language
  - isl
  - sign-language
  - isolated-signs
  - video
  - smoke-test
size_categories:
  - n<1K
configs:
  - config_name: default
    data_files: metadata.csv
---

# ISL Isolated Word Dataset — 8-word test subset

Smaller subset of [vidit031/isl-isolated-40words](https://huggingface.co/datasets/vidit031/isl-isolated-40words) for **pipeline smoke tests** and **data-efficient model training** (Landmark TCN, MediaPipe Transformer).

- **56** H.264 MP4 clips
- **8** glosses (visually distinct, 6–7 clips each)
- Same layout and `metadata.csv` schema as the full dataset
- Derived from the curated 7-clip-per-word release

## Vocabulary

yes, no, hello, water, eat, go, help, please

## Layout

```
ISL_DATASET/
  README.md
  metadata.csv
  <word_slug>/*.mp4
  <word_slug>/*.json
  <word_slug>/sources.txt
```

`video_path` in `metadata.csv` is relative to the dataset root (forward slashes).

## Download for training

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="vidit031/isl-isolated-8words",
    repo_type="dataset",
    local_dir="ISL_DATASET",
)
```

Or with the project helper:

```bash
python scripts/download_hf_dataset.py --repo vidit031/isl-isolated-8words
```

## Intended use

Quick end-to-end validation: landmark extraction → train → eval → live inference.
Not intended as a benchmark for final model accuracy.

## License

Same multi-source aggregate terms as the [full 40-word dataset](https://huggingface.co/datasets/vidit031/isl-isolated-40words). Respect upstream licenses (ISL500, CISLR, INCLUDE, ISLRTC).
