# IIPD: Baselines for Indian Sign Language Recognition

Baseline models for **isolated Indian Sign Language (ISL) word recognition**. They are the comparison points for a proposed Koopman / Hankel-DMD spectral pipeline with Kalman-filtered landmark preprocessing. All eleven models share one training loop, so they can be compared on the same data splits.

Full architecture write-ups and citations are in [`baselines/arch.md`](baselines/arch.md).

## Models

| Tier | Name | Modality | Model |
|---|---|---|---|
| 1 | `cnn_bilstm` | RGB frames | CNN + 1D-CNN + BiLSTM |
| 1 | `mp_bilstm` | MediaPipe landmarks | BiLSTM |
| 1 | `mp_transformer` | MediaPipe landmarks | Transformer encoder |
| 2 | `stgcn` | 27-joint skeleton | ST-GCN |
| 2 | `ctr_gcn`, `td_gcn` | skeleton | CTR-GCN / TD-GCN |
| 2 | `hwgat` | skeleton | Hierarchical windowed graph attention |
| 3 | `fft_bilstm` | FFT + kinematics | BiLSTM |
| 3 | `cwt_bilstm`, `cwt_transformer` | CWT bands + kinematics | BiLSTM / Transformer |
| 3 | `pgf_slr` | skeleton | Part-wise graph-Fourier attention |

## Data

The data is not stored in this repo. Download it from Hugging Face:

```python
from huggingface_hub import snapshot_download
snapshot_download("vidit031/isl-isolated-8words", repo_type="dataset", local_dir="ISL_DATASET_8WORDS")
```

- `vidit031/isl-isolated-8words`: 56 clips, 8 glosses, for smoke tests ([dataset card](ISL_DATASET_8WORDS/README.md))
- `vidit031/isl-isolated-40words`: the full 40-word set

## Usage

These scripts expect to live at `models/baselines/` in the parent IPD repo. They import a shared `common` package from `models/common` (metadata loading, splits, landmark extraction, the train/eval engine, weight paths), and that package is **not included here**.

```bash
python models/baselines/train.py --list
python models/baselines/train.py --data-dir ISL_DATASET_8WORDS --models mp_bilstm stgcn
python models/baselines/train.py --data-dir ISL_DATASET_8WORDS --models all --epochs 20
python models/baselines/eval.py  --data-dir ISL_DATASET_8WORDS --models all
```
