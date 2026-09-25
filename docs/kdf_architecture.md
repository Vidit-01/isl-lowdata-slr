# Architectural Evolution of the Koopman / Hankel-DMD Isolated-SLR Model

**Document type:** internal architecture-and-ablation report for the paper  
**Model family:** kinematic–dynamical fusion (KDF)  
**Final CLI name:** `kdf_transformer`  
**Earlier CLI name:** `kdf_stgcn` (kept as an alias)

This note records every architectural revision of the proposed Koopman model, why it was made, and what it did to few-shot accuracy. The Koopman / Hankel-DMD *front-end* is the intended scientific contribution. The *classifier* was allowed to change: the comparison-deck pipeline originally named ST-GCN / HWGAT as the fusion backbone, but the few-shot numbers showed that a large graph net is the wrong capacity for this protocol. The final model therefore attaches the same dynamical features to the best data-efficient backbone on this split (`mp_transformer`), following Neural Koopman Pooling (Wang et al., CVPR 2023) and DMD-as-empirical-feature concatenation (Zhang et al., 2021).

---

## 1. Experimental protocol (held fixed)

All four KDF variants were trained and tested under the **same locked few-shot protocol**. Baseline checkpoints (`mp_transformer`, `stgcn`, `td_gcn`, spectral models, …) were **not retrained** between KDF revisions, so McNemar tests against those models remain paired on an identical 114-clip test set.

| Item | Setting |
|---|---|
| Corpus | `vidit031/isl-isolated-40words` (642 clips, 40 glosses) |
| Class selection | 8 highest-count glosses |
| Words | thank you (39), friend (38), school (37), hello (36), market (36), okay (23), hospital (21), sit (19) |
| Protocol seed | 42 |
| Train / val / test target | 7 / 1 / 15 clips per word |
| Realised test sizes | hospital 13, sit 11; others 15 |
| Locked test \(n\) | 114 clips, identity-disjoint from train/val |
| Train-set draws | 3 (same locked test, resampled 7-shot train) |
| Headline metric | macro-F1 (equal weight per sign) |
| Uncertainty | Wilson 95% CI on overall accuracy; bootstrap 95% CI on macro-F1; McNemar exact + paired bootstrap \(\Delta\) macro-acc |
| Landmarks | MediaPipe Holistic, \(T=30\) frames, 27-joint HWGAT layout (7 upper-body + 10 per hand) |

**How to read the tables.** “Last draw” is the canonical checkpoint used for McNemar (paired clip-level decisions on the locked test). “Draws” is mean \(\pm\) std over the three 7-shot training draws. A large **val−test gap** is the overfitting signal that matters under 7-shot training with an 8-clip validation set.

**Baselines that define the target.** On this split the strongest model is `mp_transformer` (macro-F1 0.679, acc 0.684, 427,912 params, val−test gap 0.066). The strongest graph model is `td_gcn` (0.574 / 0.605). Plain `stgcn` is 0.459 / 0.491. FFT/CWT spectral models sit in the 0.61–0.64 F1 band. The proposed model has to clear ST-GCN to justify a graph-fusion story, and has to match or beat `mp_transformer` to justify Koopman features as a *sample-efficient* dynamical prior rather than as decoration on a weaker backbone.

---

## 2. What is held constant: the Koopman / Hankel-DMD front-end

From revision **v2** onward the dynamical feature extractor is frozen (feature cache tag `kdfv2`). Later gains come from the **classifier and the fusion rule**, not from changing DMD.

### 2.1 Pose cleanup (pipeline stages 1–2)

1. MediaPipe Holistic \(\to\) 27 joints \((T, V, 3)\).
2. Occlusion fill: a joint with \(\lVert xyz \rVert < 10^{-6}\) is treated as missing and linearly interpolated along time.
3. Forward–backward constant-velocity Kalman smoother on every coordinate (process noise \(q=8\times10^{-4}\), measurement noise \(r=1.5\times10^{-2}\)).
4. Signer normalisation: subtract the nose (root) and divide by mean shoulder width.

This is the “noise and occlusion filter” in the comparison-deck figure. It is *not* learned.

### 2.2 Part-wise Hankel-DMD (pipeline stage 3)

ISL motion is hand-dominant. A single DMD on the flattened 81-D pose (\(27\times 3\)) with \(T=30\) is rank-starved. From v2 the operator is estimated **separately** on three anatomical parts, matching the HWGAT body-part windows:

| Part | Joints | State dim |
|---|---:|---:|
| Upper body | 7 | 21 |
| Left hand | 10 | 30 |
| Right hand | 10 | 30 |

For each part:

1. Build a delay-embedded Hankel matrix with \(d=8\) delays.
2. Exact DMD: SVD of \(H_0\), fit Koopman operator \(K\), eigendecomposition \(K\Phi=\Phi\Lambda\).
3. Keep \(r=3\) modes, ordered oscillatory-first (\(|\arg\lambda|\) then \(|\lambda|\)).
4. Amplitudes from \(b_0=\Phi^+ x_0\) with \(|\lambda|\) clipped to \([0.5, 1.5]\) so \(\lambda^t\) does not explode.

**Clip-level spectrum** (39-D), per mode and part:

- \(\log(1+|\lambda|)\) — growth / decay  
- \(\sin\theta,\cos\theta\) of \(\arg\lambda\) — frequency, wrap-safe  
- RMS amplitude of the mode  
- plus three part kinetic energies \(\sqrt{\mathrm{mean}(v^2)}\)

**Spatial graph modes** \((V, 9)\): per-joint energy of each part’s three modes, peak-normalised inside the part. These are the “spatial graph modes / eigenvectors” in the deck, used as a *spatial map*, not as a reconstructed fake trajectory.

### 2.3 Why this representation is the claim

FFT/CWT give real magnitude and phase on sliding windows. DMD eigenvalues are **complex**: \(\lambda=e^{(\gamma+i\omega)\Delta t}\) carries growth/decay and frequency of coherent motion, and the eigenvectors are spatially structured (which joints participate). That is the intended advantage over the spectral baselines, *independent of whether the classifier is ST-GCN*.

Literature used to justify fusion *without* committing to ST-GCN:

- Wang, Xu & Mu, **Neural Koopman Pooling**, CVPR 2023. Koopman pooling is a plug-in on an existing skeleton backbone; one-shot recognition matches class-wise linear maps \(K_c\) (via DMD) rather than embedding distance.
- Zhang et al., **Action recognition based on dynamic mode decomposition**, 2021. DMD eigenvalues / system matrix are compact, untrained empirical features; concatenating them with a learned encoder helps quasi-few-shot action recognition.
- Arbabi & Mezić, Hankel-DMD, *SIADS* 2017. Delay embedding is the finite-data estimator of Koopman spectral properties.

---

## 3. Revision history (classifier and fusion only)

Four trained revisions. CLI was `kdf_stgcn` for v1–v3 and `kdf_transformer` for v4 (`kdf_stgcn` remains an alias).

### 3.1 v1 — 21-channel ST-GCN fusion (`kdf_stgcn`, first trained run)

**Intent.** Literal reading of the deck: Kalman pose \(\to\) Hankel-DMD \(\to\) fuse pos/vel/acc *and reconstructed DMD trajectories* as extra joint channels \(\to\) ST-GCN, plus a tiny eigenvalue MLP at the classifier.

**Architecture.**

- Input to ST-GCN: \(C=21=9+3\times 4\) (xyz + vel + acc + four reconstructed mode xyz’s), \(T=30\), \(V=27\).
- Backbone: full ST-GCN (blocks 64-64-64-128-128-256-256), \(\approx 2.05\)M params.
- Eigenvalues: 8-D (\(\log|\lambda|\), \(\arg\lambda/\pi\) for 4 global modes) \(\to\) MLP \(\to\) concatenated at the linear head.
- Global DMD on 81-D pose, 4 modes, 4 delays.
- Dropout 0.2; no mixup.

**Failure mode.** This is not “ST-GCN + dynamics”. It is ST-GCN with a **seven-fold channel explosion**. Reconstructed modes have arbitrary sign/scale (DMD mode ambiguity), so the extra 12 channels look like a second, clip-specific copy of the pose. With 56 training clips the net memorises train/val.

**Last-draw result.** macro-F1 **0.382** [0.294, 0.458]; acc **0.386** [0.302, 0.478]; val−test gap **0.364**; 2,053,307 params; 80 epochs. Draw-mean F1 **0.428 \(\pm\) 0.043**.

**Versus ST-GCN (same test).** ST-GCN 0.459 F1 / 0.491 acc. McNemar: ST-GCN right & KDF wrong = 20, reverse = 8, \(p=0.036\). **ST-GCN significantly beats v1.** The “novel” fusion was worse than the backbone it was supposed to improve.

**Per-class (acc).** school 0.07, market 0.13, friend 0.20, thank you 0.20, hello 0.47. Collapse is *model*-side (ST-GCN itself gets hello 0.80).

A NumPy 2.x crash (`float()` on a length-1 `h@x` array) aborted the first cloud attempt; that was a scalar-Kalman bug, not an accuracy result.

---

### 3.2 v2 — 3-channel ST-GCN + identity-init FiLM (`kdf_stgcn`)

**Intent.** Keep ST-GCN comparable to the `stgcn` baseline (3 input channels). Inject dynamics as a *conditioner*, not as extra noisy channels. Switch to part-wise Hankel-DMD (`kdfv2` cache). Add mixup and label smoothing for 7-shot.

**Architecture.**

- ST-GCN input: Kalman-normalised xyz only (\(C=3\)).
- Spatial modes: \(1\times1\) conv from 9-D mode map \(\to\) 3 channels, **zero-initialised** residual (starts as plain ST-GCN).
- Eigenvalues: FiLM on the 256-D pooled embedding, last FiLM layer **zero-initialised** (starts as identity).
- Mixup \(\alpha=0.4\), label smoothing 0.1, dropout 0.35, weight decay \(5\times10^{-2}\), lr \(8\times10^{-4}\).
- 2,084,267 params.

**Hypothesis.** If Koopman features help, FiLM/mode residual can grow from identity. If they do not, the model remains ST-GCN.

**What actually happened.** Val still reached 0.75 on 8 clips; early stop at epoch 42. FiLM *can* suppress good pose features once it leaves identity. Hello fell from ST-GCN’s 0.80 to **0.27**. School went to **0/15**. Mixup 0.4 on batches of 16 with 56 clips is aggressive.

**Last-draw result.** macro-F1 **0.408** [0.319, 0.482]; acc **0.421** [0.334, 0.513]; gap **0.329**. Draw-mean F1 **0.447 \(\pm\) 0.038**.

**Delta vs v1.** Last-draw F1 \(+0.026\), acc \(+0.035\), gap \(-0.035\). McNemar vs ST-GCN became non-significant (\(p=0.185\)): the model is no longer *significantly worse* than ST-GCN, but it still does not beat it (point \(\Delta\) macro-acc still favours ST-GCN).

**Lesson.** Channel-count was part of the problem; **ST-GCN + GAP + a tiny val set** remains the problem. Identity-init FiLM is not a free lunch in 7-shot: it can unlearn pose.

---

### 3.3 v3 — compact joint/bone graph encoder + temporal transformer (`kdf_stgcn`)

**Intent.** Stop using a 2M ST-GCN. On this protocol, transformers on landmarks (`mp_transformer` 0.679, `cwt_transformer` 0.635) beat every GCN. Keep Koopman features; replace the backbone with a *small* graph encoder that **does not downsample time**, then a 2-layer transformer with attention pooling. Concatenate Koopman tokens instead of FiLM.

**Architecture.**

- Shared ST-GCN blocks (3 \(\to\) 64 \(\to\) 64 \(\to\) 128), **no temporal stride**, run on joint stream and bone stream.
- Part pooling (upper / left hand / right hand) \(\to\) 2-layer transformer (\(d=128\), 4 heads) + attention pool + mean pool.
- Concat [motion 128-D, eig MLP 64-D, mode MLP 64-D] \(\to\) classifier.
- Mixup 0.2, dropout 0.25, 3-part Koopman features unchanged (`kdfv2`).
- **639,120 params** (about 31% of v2).

**Last-draw result.** macro-F1 **0.578** [0.495, 0.652]; acc **0.588** [0.496, 0.674]; gap **0.162**. Draw-mean F1 **0.535 \(\pm\) 0.046**. Epochs 53.

**Delta vs v2.** Last-draw F1 \(+0.170\), acc \(+0.167\), gap \(-0.167\). This is the first revision that is **clearly a different operating point**, not a tweak.

**Versus graph baselines (McNemar).** Beats `stgcn` (\(p=0.019\)), `ctr_gcn` (\(p=0.035\)), `hwgat` (\(p=0.023\)). Tied with `td_gcn` (\(p=0.791\)). Still **loses to `mp_transformer`** (11 vs 0 disagreements, \(p=0.001\)).

**Per-class.** Hello recovered to 0.93 (the FiLM damage is gone). Friend 0.53, sit 0.82. School 0.13 and market 0.20 remain weak — those glosses are hard for almost every model, but v3 does not yet use the landmark sequence that `mp_transformer` uses (225-D pose+hands, including joints the 27-node graph drops).

**Lesson.** Capacity and temporal pooling dominate fusion details. A 639k time-preserving encoder with concat fusion is enough to beat ST-GCN. It is **not** enough to beat the best sequence model, because the input is still the reduced 27-joint graph, not the full MediaPipe pose+hands vector.

---

### 3.4 v4 — Koopman plug-in on `mp_transformer` (`kdf_transformer`)

**Intent.** Papers do not require ST-GCN. Wang et al. (CVPR 2023) treat Koopman pooling as a plug-in on the backbone that already works; Zhang et al. (2021) concatenate DMD features with a learned encoder specifically for few-shot. The best backbone on *this* split is `mp_transformer`. Attach Hankel-DMD to that model and add class-wise dynamical matching.

**Architecture.**

- **Backbone (identical hparams to `mp_transformer`):** MediaPipe pose+hands sequence \((T, 225)\), linear project to \(d=128\), sinusoidal positions, 3-layer pre-norm transformer (4 heads, FFN 256), dropout 0.2.
- **Zhang-style concat:** mean-pooled tokens (128-D) \(\oplus\) eig MLP (64-D) \(\oplus\) part-mode MLP (64-D) \(\to\) LayerNorm + linear over 8 classes.
- **Wang-style class-wise Koopman head:** each class is a rank-8 map \(K_c \approx U_c V_c^\top\) on transformer tokens. Score \(s_c = -\|h_{t+1}-K_c h_t\|_F^2\). Columns of \(U,V\) are \(\ell_2\)-normalised (stability prior, analogous to their eigenvalue normalisation). Logits \(=\) concat-head \(+\; \alpha\, s\) with learned scalar \(\alpha\) (init 0.5). Auxiliary CE on \(s_c\) with weight 0.3.
- Mixup 0.2, label smoothing 0.1 on the concat head.
- **459,669 params** (\(\approx\) 428k transformer + 32k Koopman). Same order as `mp_transformer` (427,912) and `cwt_transformer` (471,688).
- Feature cache still `kdfv2` (Koopman extractor unchanged from v2).

**Last-draw result.** macro-F1 **0.682** [0.594, 0.757]; acc **0.693** [0.603, 0.770]; Wilson on acc uses \(k=79/114\); val−test gap **0.057**; 79 epochs. Draw-mean F1 **0.649 \(\pm\) 0.030**; draw-mean acc **0.661 \(\pm\) 0.031**.

**Delta vs v3.** Last-draw F1 \(+0.104\), acc \(+0.105\), gap \(-0.105\), params \(639\mathrm{k}\to 460\mathrm{k}\). Draw-mean F1 \(0.535\to 0.649\).

**Versus `mp_transformer` (the ablation that matters).** Same transformer, plus Koopman concat + class-wise \(K_c\). Last-draw: KDF 0.682 F1 / 0.693 acc vs MP 0.679 / 0.684. McNemar **4 vs 5**, \(p=1.000\), \(\Delta\) macro-acc \(-0.008\) [−0.053, 0.041] (CI includes 0). **Statistically tied** on the locked test; point estimates slightly favour KDF. Gap 0.057 vs 0.066 (slightly less overfit).

**Versus graph / spectral models (McNemar, last draw).** Significantly beats `stgcn`, `ctr_gcn`, `td_gcn` (\(p=0.031\)), `hwgat`, `cwt_bilstm`. Bootstrap \(\Delta\) vs `fft_bilstm` excludes 0 but McNemar \(p=0.077\). Not significantly different from `cwt_transformer` (\(p=0.238\)). Beats `mp_bilstm` on paired bootstrap (CI excludes 0) with McNemar \(p=0.146\).

---

## 4. Headline numbers across revisions

Locked test \(n=114\). Last-draw columns are the canonical checkpoint (McNemar). \(\Delta\) is versus the immediately previous KDF revision.

| Rev | CLI | Classifier | Params | Last-draw macro-F1 | Last-draw acc | Draw-mean F1 | Val−test gap | Epochs |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v1 | `kdf_stgcn` | ST-GCN, 21-ch reconstructed DMD | 2,053,307 | 0.382 [0.294, 0.458] | 0.386 [0.302, 0.478] | 0.428 \(\pm\) 0.043 | 0.364 | 80 |
| v2 | `kdf_stgcn` | ST-GCN 3-ch + FiLM / mode residual | 2,084,267 | 0.408 [0.319, 0.482] | 0.421 [0.334, 0.513] | 0.447 \(\pm\) 0.038 | 0.329 | 42 |
| v3 | `kdf_stgcn` | Compact joint/bone GCN + 2-layer transformer, concat | 639,120 | 0.578 [0.495, 0.652] | 0.588 [0.496, 0.674] | 0.535 \(\pm\) 0.046 | 0.162 | 53 |
| v4 | `kdf_transformer` | `mp_transformer` + DMD concat + class-wise \(K_c\) | 459,669 | **0.682 [0.594, 0.757]** | **0.693 [0.603, 0.770]** | **0.649 \(\pm\) 0.030** | **0.057** | 79 |

| Step | \(\Delta\) last-draw F1 | \(\Delta\) last-draw acc | \(\Delta\) gap | Interpretation |
|---|---:|---:|---:|---|
| v1 \(\to\) v2 | +0.026 | +0.035 | −0.035 | Stops losing *significantly* to ST-GCN; still overfits; FiLM hurts hello/school |
| v2 \(\to\) v3 | **+0.170** | **+0.167** | **−0.167** | Backbone/capacity/temporal pooling, not a Koopman tweak |
| v3 \(\to\) v4 | **+0.104** | **+0.105** | **−0.105** | Same dynamical features on the protocol’s best sequence model + \(K_c\) matching |
| v1 \(\to\) v4 | **+0.300** | **+0.307** | **−0.307** | End-to-end: from worse-than-ST-GCN to tied with the best baseline |

Reference baselines on the **same** locked test (unchanged across KDF runs):

| Model | Params | Last-draw macro-F1 | Last-draw acc | Gap |
|---|---:|---:|---:|---:|
| `mp_transformer` | 427,912 | 0.679 [0.599, 0.750] | 0.684 [0.594, 0.762] | 0.066 |
| `cwt_transformer` | 471,688 | 0.635 [0.546, 0.708] | 0.640 [0.549, 0.723] | −0.015 |
| `mp_bilstm` | 2,571,272 | 0.627 [0.547, 0.689] | 0.640 [0.549, 0.723] | 0.110 |
| `fft_bilstm` | 2,939,912 | 0.618 [0.534, 0.692] | 0.623 [0.531, 0.706] | 0.127 |
| `td_gcn` | 1,446,290 | 0.574 [0.484, 0.652] | 0.605 [0.514, 0.690] | 0.145 |
| `stgcn` | 2,046,263 | 0.459 [0.382, 0.526] | 0.491 [0.401, 0.582] | 0.134 |
| `hwgat` | 1,206,792 | 0.466 [0.373, 0.545] | 0.482 [0.393, 0.573] | 0.143 |

v4 last-draw F1 is the highest number in the table (0.682 vs 0.679). That is a **0.3-point** difference on 8 classes / 114 clips and is **not** McNemar-significant. The honest paper claim is: *Koopman features on the MediaPipe transformer match the best baseline and significantly beat ST-GCN, CTR-GCN, TD-GCN, and HWGAT; they do not, on this split, significantly outperform the transformer without Koopman.*

---

## 5. Per-class accuracy (last draw)

Fractions are correct/test. Hospital test \(n=13\), sit \(n=11\), others \(n=15\).

| Gloss | v1 | v2 | v3 | v4 `kdf_transformer` | `mp_transformer` | `stgcn` | `td_gcn` |
|---|---:|---:|---:|---:|---:|---:|---:|
| friend | 0.20 (3/15) | 0.33 (5/15) | 0.53 (8/15) | **0.87 (13/15)** | 0.80 (12/15) | 0.07 | 0.67 |
| hello | 0.47 (7/15) | 0.27 (4/15) | 0.93 (14/15) | 0.93 (14/15) | 0.93 (14/15) | 0.80 | 0.73 |
| hospital | 0.85 (11/13) | 0.92 (12/13) | 0.92 (12/13) | **1.00 (13/13)** | 1.00 (13/13) | 1.00 | 1.00 |
| market | 0.13 (2/15) | 0.13 (2/15) | 0.20 (3/15) | **0.40 (6/15)** | 0.27 (4/15) | 0.07 | 0.13 |
| okay | 0.67 (10/15) | 0.80 (12/15) | 0.93 (14/15) | 0.87 (13/15) | 0.93 (14/15) | 0.93 | 1.00 |
| school | 0.07 (1/15) | 0.00 (0/15) | 0.13 (2/15) | **0.33 (5/15)** | 0.27 (4/15) | 0.20 | 0.13 |
| sit | 0.64 (7/11) | 0.73 (8/11) | 0.82 (9/11) | **1.00 (11/11)** | 1.00 (11/11) | 0.82 | 0.91 |
| thank you | 0.20 (3/15) | 0.33 (5/15) | 0.33 (5/15) | 0.27 (4/15) | **0.40 (6/15)** | 0.20 | 0.40 |

**Where Koopman+transformer helps relative to `mp_transformer` (last draw).** friend +1 clip, market +2, school +1; okay −1, thank you −2. Net McNemar 5 vs 4 is that trade. Market is the corpus-wide hard sign (every model is weak there); v4 is the *least* weak (0.40 vs 0.07–0.27). Thank you is the gloss v4 does *not* win — do not claim uniform per-class gains.

**Where the backbone change dominates.** hello 0.27 \(\to\) 0.93 from v2 \(\to\) v3 is recovery from FiLM, not a new dynamical feature. friend 0.53 \(\to\) 0.87 and sit 0.82 \(\to\) 1.00 from v3 \(\to\) v4 track switching from 27-joint GCN tokens to the 225-D pose+hands transformer.

---

## 6. Paired tests vs KDF, by revision

All tests on the same 114-clip locked set. “A right / B wrong” uses A = listed baseline, B = that KDF revision.

### 6.1 vs `stgcn` (the named fusion backbone in the original deck)

| KDF rev | ST-GCN right, KDF wrong | KDF right, ST-GCN wrong | McNemar \(p\) | \(\Delta\) macro-acc (ST-GCN − KDF) |
|---|---:|---:|---:|---|
| v1 | 20 | 8 | 0.036 | +0.109 [0.022, 0.194] — **ST-GCN better** |
| v2 | 18 | 10 | 0.185 | +0.071 [−0.010, 0.154] — n.s. |
| v3 | 4 | 15 | 0.019 | −0.090 [−0.156, −0.024] — **KDF better** |
| v4 | 2 | 25 | 0.000 | −0.198 [−0.266, −0.131] — **KDF better** |

The deck’s ST-GCN fusion story is only true from **v3** onward. v1 is a cautionary ablation: naively stacking DMD reconstructions into ST-GCN **hurts**.

### 6.2 vs `mp_transformer` (the protocol’s best model)

| KDF rev | MP right, KDF wrong | KDF right, MP wrong | McNemar \(p\) | \(\Delta\) macro-acc (MP − KDF) |
|---|---:|---:|---:|---|
| v1 | 36 | 2 | 0.000 | +0.298 [0.213, 0.387] — **MP better** |
| v2 | 31 | 1 | 0.000 | +0.260 [0.184, 0.340] — **MP better** |
| v3 | 11 | 0 | 0.001 | +0.099 [0.046, 0.155] — **MP better** |
| v4 | 4 | 5 | 1.000 | −0.008 [−0.053, 0.041] — **tied** |

v4 is the first revision that is not significantly worse than the best baseline. It is also **not** significantly better. Paper wording should be “matches the strongest baseline; Koopman features do not harm the transformer and yield a small, non-significant point-estimate gain.”

### 6.3 v4 vs the rest (for the results table)

| Baseline | McNemar \(p\) | Winner (if \(p<0.05\) or CI excludes 0) |
|---|---:|---|
| `cnn_bilstm` | 0.000 | KDF |
| `pgf_slr` | 0.000 | KDF |
| `stgcn` | 0.000 | KDF |
| `ctr_gcn` | 0.000 | KDF |
| `hwgat` | 0.000 | KDF |
| `td_gcn` | 0.031 | KDF |
| `cwt_bilstm` | 0.043 | KDF |
| `fft_bilstm` | 0.077 | CI favours KDF, McNemar n.s. |
| `mp_bilstm` | 0.146 | CI favours KDF, McNemar n.s. |
| `cwt_transformer` | 0.238 | n.s. |
| `mp_transformer` | 1.000 | n.s. (tied) |

---

## 7. Overfitting trajectory

Validation is 8 clips (1 per class). Val acc is quantised in steps of 0.125; several models report `best_val_acc = 0.75` (6/8) including every KDF revision. **Test gap is the quantity that changed.**

| Rev | Best val acc | Test acc | Gap | Reading |
|---|---:|---:|---:|---|
| v1 | 0.75 | 0.386 | 0.364 | Memorised 6/8 val clips; test near chance-to-weak |
| v2 | 0.75 | 0.421 | 0.329 | Same val, slightly less disastrous test |
| v3 | 0.75 | 0.588 | 0.162 | Val still 6/8; test in the TD-GCN band |
| v4 | 0.75 | 0.693 | **0.057** | Same val, test now with `mp_transformer` (gap 0.066) |

Draw-mean gaps: v2 0.355 \(\pm\) 0.114; v3 0.212 \(\pm\) 0.098; v4 0.157 \(\pm\) 0.100. Last-draw v4 gap (0.057) is the optimistic draw; the mean gap across draws is still \(\sim\)0.16. Report both.

---

## 8. Parameter count vs accuracy

```
params (M)     last-draw macro-F1
2.05  v1 ST-GCN 21-ch          0.382
2.08  v2 ST-GCN FiLM           0.408
1.45  td_gcn (baseline)        0.574
0.64  v3 compact GCN+T         0.578
0.47  cwt_transformer          0.635
0.43  mp_transformer           0.679
0.46  v4 kdf_transformer       0.682
```

Under 7-shot, **more parameters on a GCN made accuracy worse**. The useful operating region is 0.4–0.5M, the same band as the winning transformers. v4 adds only ~32k parameters on top of `mp_transformer` (~7.5%).

---

## 9. What each change was *for*, in one line

| Change | Scientific role | Empirical effect |
|---|---|---|
| Kalman + occlusion interp | Deck stage 2; stable DMD on MediaPipe dropouts | Necessary plumbing; not ablated alone |
| Part-wise Hankel-DMD (v2+) | Hand-centric Koopman; avoids 81-D rank starvation | Kept thereafter; not the source of the +0.30 F1 |
| Stop putting reconstructed modes in \(C=21\) | Remove DMD sign/scale ambiguity from GCN input | v1\(\to\)v2: small F1 gain, ends significant loss to ST-GCN |
| FiLM / zero-init residual | “Don’t hurt ST-GCN unless dynamics help” | Failed: hello 0.80\(\to\)0.27, school 0/15 |
| Drop full ST-GCN, keep time, concat fusion (v3) | Capacity + temporal pooling | **+0.17 F1**; first win vs ST-GCN |
| Backbone = `mp_transformer` (v4) | Koopman as plug-in on the best few-shot model | **+0.10 F1**; tie with best baseline |
| Class-wise low-rank \(K_c\) matching | Wang et al. one-shot dynamics matching | Coupled with v4; not isolated in a 2-way ablation |
| Mixup 0.4 \(\to\) 0.2 | 7-shot regularisation | 0.4 coincided with v2 collapse; 0.2 used in v3/v4 |
| CLI rename `kdf_stgcn` \(\to\) `kdf_transformer` | Name matches the model | No accuracy effect |

A 2-way ablation that is **still missing** (and should be flagged in the paper as future / supplementary): v4 transformer **without** eig/mode concat and **without** \(K_c\), trained with the same mixup/smoothing, versus full v4. `mp_transformer` is the closest proxy (same backbone, no mixup, no Koopman). The McNemar tie means we cannot yet claim a significant unique effect of the operator on this 8-word split.

---

## 10. Recommended paper narrative

**Do not** present v1 as the proposed model. Present it as a negative ablation: *fusing reconstructed Hankel-DMD trajectories as extra ST-GCN channels overfits 7-shot ISL and is significantly worse than ST-GCN alone.*

**Do** present the method as: *part-wise Hankel-DMD / Koopman spectrum (growth, frequency, spatial modes) as a plug-in dynamical descriptor, combined with a compact landmark transformer and class-wise Koopman matching*, citing Wang et al. (CVPR 2023) and Zhang et al. (2021).

**Results sentence that is supported.** On a locked 8-gloss, 7-shot, 114-clip ISL test, `kdf_transformer` (459,669 params) reaches last-draw macro-F1 0.682 [0.594, 0.757] and accuracy 0.693 [0.603, 0.770], significantly above ST-GCN, CTR-GCN, TD-GCN, and HWGAT, and not significantly different from the strongest baseline (`mp_transformer`, McNemar \(p=1.0\)). Mean \(\pm\) std over three training draws is 0.649 \(\pm\) 0.030 F1. The same Koopman features on a 2M 21-channel ST-GCN (v1) scored 0.382 F1 and lost to ST-GCN (\(p=0.036\)), so the operator is not a substitute for a data-efficient backbone.

**Limitations to state.** (1) Eight glosses, 7-shot, one corpus; hospital/sit test sets are short. (2) `mp_transformer` was not re-run with mixup, so the v4 vs MP comparison confounds Koopman features with regularisation. (3) Class-wise \(K_c\) and DMD concat were not ablated separately. (4) Draw-mean F1 (0.649) is below last-draw (0.682); last-draw is the paired McNemar snapshot, not an unbiased estimator of expected 7-shot performance.

---

## 11. Implementation pointers

| Item | Location |
|---|---|
| Koopman / Kalman / Hankel-DMD | `islr/models/kdf.py` (`kdf_joint_features`, cache `kdfv2`) |
| Classifier v4 | `KDFSTGCN` + `ClassKoopmanHead` in the same file |
| CLI | `kdf_transformer` in `islr/models/registry.py` (alias `kdf_stgcn`) |
| Data | `KDFSkeletonDataset` in `islr/fewshot/data.py` (pose+hands sequence + cached eig/modes) |
| Protocol | `islr/fewshot/protocol.py`, seed 42, `--train-shots 7 --test-per-class 15 --draws 3` |
| Cloud | `python scripts/run_pipeline_baselines.py --skip-clone --models kdf_transformer` |

---

## 12. Sources for the numbers

| KDF rev | Report file (local) | Comparison JSON |
|---|---|---|
| v1 | `baselines_report (2).md` | `baselines_comparison (1).json` |
| v2 | `baselines_report (3).md` | `baselines_comparison (2).json` |
| v3 | `baselines_report (4).md` | `baselines_comparison (3).json` |
| v4 | `baselines_report (5).md` | `baselines_comparison (4).json` |

Baseline rows in those files are the same canonical run (protocol seed 42). Only the KDF row changes.
