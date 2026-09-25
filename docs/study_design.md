# Low-data sign language recognition — ISL as a proxy for low-resource sign languages

Status: 25 Sep 2026.

The code is written and has passed a synthetic smoke test on CPU. **There are no results on real data yet.** All heavy compute (extraction and sweeps) will run on Kaggle, spread over the four team members' accounts. Nothing heavy has been run locally.

---

## 1. Motivation

* Most of the world's sign languages have a few hundred labelled clips, not tens of thousands.
* Indian Sign Language (ISL) has several small public corpora, of different sizes, signers and recording conditions. That makes it a realistic stand-in for studying:
  1. **How accuracy scales with data.** Training clips per word (K) run from 1 to all.
  2. **Scarce words inside a larger vocabulary.** A few words have K clips while many others are well resourced. This generalises the earlier "8 videos for a few words vs many videos for many words" experiment.
  3. **Which inductive biases pay off when K is tiny.** The candidates are graphs, spectral features, Koopman/DMD dynamics, part-factorised encoders, prototype/cosine heads, and mixup.
* Every comparison is **signer-independent**, with a test set that is fixed per seed.

## 2. Code base

| path | content |
|---|---|
| `islr/models/` | the 12 existing model families plus the new models, and the registry |
| `islr/fewshot/` | the legacy few-shot data pipeline and loop (`loop.py`) |
| `islr/common/` | landmark extraction/cache, training engine |
| `islr/lowdata/` | **new study package** (below) |
| `lowdata.py` | launcher: `extract`, `sources`, `inspect`, `run`, `sweep`, `report`, `smoke` |
| `kaggle/` | `lowdata_study.ipynb` (the whole pipeline) and `README.md` (4-account workflow) |

Files in `islr/lowdata/`:

| file | role |
|---|---|
| `store.py` | landmark store format: raw Holistic output `(T, 543, 3)` float16 with NaN for missing points. Also layout conversion (GISLR order is verified), identity rules, deduplication, and merging shard stores. |
| `extract.py`, `sources.py` | MediaPipe extraction (Tasks and legacy APIs), resumable and shardable (`--shard i/n`). Ingestion for the 40-word corpus, INCLUDE (Zenodo, zip by zip), the ISLRTC dictionary, CISLR (gated) and GISLR. |
| `protocol.py` | signer-independent protocols with nested K subsets |
| `bank.py`, `augment.py` | per-modality feature banks, computed once per clip table and kept in memory, with augmentation done on the GPU |
| `run.py` | one run: fixed step budget, fp16, DDP, checkpoint/resume, metrics, prototype evaluation |
| `sweep.py` | grid → jobs, split across members by K, one queue per GPU, resume across Kaggle sessions |
| `report.py` | merges the members' outputs: mean ± std tables, curves, McNemar and paired bootstrap tests, consistency checks |
| `inspect_data.py`, `smoke_test.py`, `configs/*.json` | data checks, the one-command test, and the grids |

### The earlier experiment (legacy pipeline)

The legacy pipeline downloaded `vidit031/isl-isolated-40words` (642 clips, 40 glosses, sourced from ISL500, INCLUDE, CISLR and ISLRTC). It kept the 8 glosses with the most clips and trained a 7-shot / 15-test protocol over 3 draws.

The new study keeps those 8 words as the **target set** (`LEGACY8` = eat, go, hello, help, no, please, water, yes) and embeds them in larger vocabularies.

## 3. Models (all trained under the same recipe)

| name | input | family | params (100 classes) |
|---|---|---|---|
| mp_bilstm | pose+hands landmarks | tier 1 | 2.62 M |
| mp_transformer | pose+hands landmarks | tier 1 | 0.44 M |
| stgcn | 27-joint skeleton graph | tier 2 | 2.07 M |
| ctr_gcn / td_gcn | skeleton | tier 2 | 1.47 M |
| hwgat | skeleton | tier 2 | 1.22 M |
| fft_bilstm | FFT + kinematics | tier 3 | 2.99 M |
| cwt_bilstm / cwt_transformer | CWT bands | tier 3 | 3.32 / 0.48 M |
| pgf_slr | graph-Fourier attention | tier 3 | 0.84 M |
| kdf_transformer | landmarks + Hankel-DMD spectrum + class-wise Koopman head | novel | 0.68 M |
| mp_transformer_reg | KDF backbone + mixup/LS, no DMD, no Koopman head | ablation | 0.45 M |
| kdf_transformer_nodmd / _nokc | KDF minus one component | ablation | 0.66 / 0.48 M |
| **partformer** | part tokens: body, shared mirrored hand encoder, lips+nose, presence mask, part dropout | new | 0.41 M |
| **partformer_cos** | the same with a cosine (normalised) classifier | new | 0.41 M |
| **conv1d_former** | part front end + depthwise-conv/transformer blocks | new | 0.61 M |
| **kdf_partformer** | PartFormer + Hankel-DMD + class-wise Koopman head | new (novel) | 0.64 M |
| cnn_bilstm | RGB frames (ResNet-18) | tier 1 | 22.6 M; smoke-tested only (the stores hold landmarks, not video) |

Every run also reports **prototype accuracy**: nearest class mean, by cosine, in the classifier's input space. This separates the representation from the linear head, which is poorly estimated at K = 1–2.

## 4. Data

| source | size | identities | licence / redistribution | status |
|---|---|---|---|---|
| 40-word corpus (`vidit031/isl-isolated-40words`) | 642 clips, 40 words | ISL500 sessions, INCLUDE sessions, ISLRTC, CISLR clips | mixed, per clip (`license` column) | ingestion written; local extraction stopped part-way (not needed; redo on Kaggle) |
| INCLUDE (Zenodo 4010759) | ~4.3k clips, 263 words, ~50 GB of zips | no signer ids; **recording sessions derived from MVI camera counters** (jump > 200 = new session) | CC BY 4.0 | shardable ingestion, zip by zip |
| ISLRTC dictionary (HF re-encode) | 1 clip per word | one signer | MIT on the HF copy (check the original ISLRTC terms before redistribution) | written, restricted to the study vocabulary |
| CISLR (Exploration-Lab) | ~7k clips, one per word | YouTube | gated, research only | needs terms accepted + `HF_TOKEN`; the CSV schema is detected at run time (untested) |
| GISLR (Kaggle ASL Signs) | 94k sequences (ASL) | 21 signers | competition data, **must not be redistributed** | loader verifies row order; optional transfer source |
| iSign, ISL-CSLTR | continuous / sentence-level | — | — | not ingested (sentence-level; out of scope for isolated recognition) |

**Identity** is signer, else session, else clip. Duplicate INCLUDE clips that appear in more than one corpus are removed by (word, MVI number).

**Pilot extraction** (4 clips, CPU): pose and face were found in 100 % of frames and the right hand in 29–65 %. The left hand was found only in the INCLUDE clip (70 %). Hands are often missing, which is why the new models use presence masks.

## 5. Protocols

* **Test set.** Identities are held out greedily in a seeded order until each word reaches about 25 % test clips (at least 2 and at most 20 per word, never more than 50 % of a word's clips).
  * The test set depends only on (clip table, seed). It is identical for every K, model and protocol that shares the vocabulary, and `report` checks this.
* **Training subsets.** Each word's pool is ordered once per (seed, word), round-robin over identities. The K = 1 subset is therefore contained in K = 2, and so on (nested), and small K already spans several signers.
* **Protocol variants:**

  | protocol | training data |
  |---|---|
  | `uniform` | K clips for every word |
  | `scarce` | the 8 target words get K clips; every other word keeps its whole pool |
  | `cross_source` | train on one corpus, test on another |
  | `full` | all clips |
  | `n_words` | restricts the vocabulary for any protocol (targets always kept) |

* **Reporting.** Target (scarce-word) and rich-word accuracy are reported **separately**, along with how often a target clip is predicted as a rich word.
* **Seeds.** 3 seeds for everything, with mean ± std across seeds, plus exact McNemar tests and paired bootstrap CIs on the pooled test clips against a reference model.

### Training recipe (identical for all models)

* **No validation split and no early stopping.** With K ≤ 8 a validation split would consume the data and select on noise.
* **Fixed budget:** steps = clamp(epochs · n_train / batch, 300, 6000).
* **Optimiser:** AdamW with 5 % warmup and cosine decay, gradient clip 1.0, fp16 + GradScaler on CUDA.
* **Augmentation** on GPU: rotation ±15°, scale, anisotropy, shift, noise on detected points only, and a non-circular time warp. No flips, since a flip would swap the dominant hand.
* **Batches** come from a stateless seeded stream. A resumed run therefore sees exactly the same batches, and all DDP ranks agree on the global batch.

## 6. Compute plan (Kaggle, 4 accounts, 2 × T4 each)

* **Extraction.** Each member extracts shard *i*/4. INCLUDE is sharded by zip (~12 GB of downloads each); other corpora are sharded by a clip-key hash. One member merges the shards into a private dataset.
* **Sweeps.** Jobs are split **by K (videos per word)**.
  * The groups are dealt to members by estimated cost. When whole K values would be unbalanced (> 25 %), the split falls back to (K, seed).
  * The split is deterministic, so each member only needs their own `MEMBER` number.
* **GPU use.** Low-data runs are too small for DDP to help, so each T4 runs its own queue of whole jobs. `pretrain_full.json` (all data) uses torchrun + NCCL + SyncBatchNorm on both GPUs.
* **12 h limit.** Runs checkpoint about 10 minutes before the limit. The next session attaches the previous output and continues.

### GPU-time estimates

These are rough. Per-step times are T4 guesses to be recalibrated from the first finished runs (`sweep --plan` does this automatically). Clip counts are also assumed: ~280 words and ~3.6k rich-word pool clips.

| grid | jobs | GPU-h | wall per member (2 × T4) |
|---|---|---|---|
| scarce_legacy8 (18 models × K∈{1,2,4,8,16} × 3 seeds) | 270 | ~12 | ~1.5 h |
| uniform_k (9 models × K∈{1..32} × 3) | 162 | ~6 | ~0.8 h |
| vocab (9 × 2 K × 3 × 5 sizes) | 270 | ~7 | ~1 h |
| cross_source | 108 | ~4 | ~0.6 h |
| pretrain_full (DDP) | 3 | ~0.4 | — |
| **total** | 813 | **~30** (plausible range 20–80) | about one Kaggle session per member |

Extraction dominates: INCLUDE on Kaggle CPUs at roughly 1–2 videos/s would take about 1–2 h per member for 4.3k clips, plus download time.

## 7. Results so far

* **Real data:** none. The local pilot sweep was dropped when the plan moved all compute to Kaggle.
* **Synthetic smoke test** (CPU; a pipeline check, *not evidence about the models*). All 18 models beat chance (0.08). The first 7 ran for 80 steps and the other 11 for 150, so the two groups are not comparable:

  | model | steps | top-1 |
  |---|---|---|
  | mp_bilstm | 80 | 0.60 |
  | stgcn | 80 | 0.25 |
  | cwt_transformer | 80 | 0.85 |
  | kdf_transformer | 80 | 0.38 |
  | partformer | 80 | 0.21 |
  | conv1d_former | 80 | 0.42 |
  | kdf_partformer | 80 | 0.25 |
  | mp_transformer | 150 | 1.00 |
  | ctr_gcn | 150 | 0.27 |
  | td_gcn | 150 | 0.25 |
  | hwgat | 150 | 0.56 |
  | fft_bilstm | 150 | 0.96 |
  | cwt_bilstm | 150 | 0.98 |
  | pgf_slr | 150 | 0.40 |
  | mp_transformer_reg | 150 | 0.96 |
  | kdf_transformer_nodmd | 150 | 0.88 |
  | kdf_transformer_nokc | 150 | 0.77 |
  | partformer_cos | 150 | 0.21 |

  * The graph models (ctr_gcn, td_gcn and hwgat) take 6–10 min on CPU at 150 steps; the rest take seconds.
  * Protocol invariants, checkpoint resume, sweep member split and resume, report, and shard/merge also passed.
  * The 2-rank DDP check (gloo, CPU) passed: kdf_transformer 0.25, partformer 0.21.

## 8. Known bugs and limitations

Fixed in the legacy code:

* the multi-layer LSTM dropout / state-dict layout (a legacy loader is kept);
* circular `np.roll` time augmentation, which wrapped the end of a sign onto its start;
* noise added to missing (zero) landmarks;
* the landmark cache keyed by absolute path (the new key is portable, with a fallback to the old one);
* feature caches keyed by stem only (name clashes);
* INCLUDE clips treated as independent identities, which leaked signers between train and test;
* the deprecated `torch.cuda.amp`.

Remaining, and documented rather than changed:

* the legacy loop selects the best epoch on an 8-clip validation set (the new runner does not);
* the PGF frequency gate;
* in the legacy features, missing points equal the origin (the new part models use presence masks);
* ISL500 "sessions" are assumed to be single signers;
* INCLUDE sessions are inferred from camera counters;
* the CISLR schema is untested;
* the ISLRTC dictionary has one signer, so it can only train.

## 9. What remains

1. Push the code (after approval) and upload a private code dataset; extract INCLUDE and the 40-word corpus on Kaggle (4 shards) and merge.
2. `inspect` the merged stores: check per-seed feasibility for every K and the detection rates.
3. Run `scarce_legacy8` first (the main result), then `uniform_k`, `vocab` and `cross_source`. Optionally pre-train on `full` or GISLR and fine-tune through `init_from`.
4. `report`, then write up.

## 10. Paper outline

1. Introduction: low-resource sign languages and why ISL is a proxy.
2. Related work: ISLR benchmarks (INCLUDE, WLASL, GISLR), few-shot SLR, skeleton models, Koopman/DMD.
3. Data: unified landmark stores, identity inference, deduplication, licences.
4. Protocols: signer-independent fixed tests, nested K, scarce-in-rich, cross-source, a fixed budget with no validation.
5. Models: baselines, KDF, PartFormer family, ablations.
6. Results:
   * accuracy vs K curves;
   * scarce vs rich accuracy and absorption rates;
   * vocabulary-size effect;
   * domain shift;
   * prototype vs linear head;
   * paired tests.
7. Analysis: which inductive biases help at K ≤ 4, and the effect of missing hands.
8. Limitations and ethics: signer diversity, licences, no redistribution of competition data.
