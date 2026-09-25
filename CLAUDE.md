# CLAUDE.md

ISL isolated-sign recognition baselines. Architecture reference: `baselines/arch.md`.

## Commands (run from the parent IPD repo root, where this lives at `models/baselines/`)
- List models: `python models/baselines/train.py --list`
- Train a subset: `python models/baselines/train.py --data-dir ISL_DATASET_8WORDS --models mp_bilstm stgcn`
- Train one model quickly: `... --models stgcn --epochs 5`
- Eval saved weights: `python models/baselines/eval.py --data-dir ISL_DATASET_8WORDS --models all`
- No test suite, linter config, or requirements file exists.

## Gotchas
- `train.py`/`eval.py` do `sys.path.insert(ROOT / "models")` with `ROOT = parents[2]`, and import `common` (`common.engine`, `common.landmarks`). That package is NOT in this repo, so the scripts won't run standalone.
- New models: add a builder and a `BaselineSpec` in `baselines/registry.py`. Any config key that isn't a model kwarg must go in `TRAIN_KEYS`, or it gets passed to the model constructor.
- `pgf_slr` uses a custom `forward_fn` (aux loss) and `use_amp=False`.
- Datasets, landmark stores (`lowdata_work/`) and the `*.task` MediaPipe model are gitignored. The data lives on HF (`vidit031/isl-isolated-8words`, `-40words`).
