# Baseline Model Architectures for ISL Recognition — Detailed Reference

This expands the nine-model comparison table into full architecture descriptions, with the most depth given to the three models your proposal flags as the toughest to beat (CTR-GCN/TD-GCN, HWGAT, and PGF-SLR-style graph-Fourier attention), since those are the ones a reviewer will actually press you on. Each entry includes what the model computes, how the pieces fit together, and where to point a citation.

---

## Tier 1 — Sequence-model baselines (weak spatial structure)

### 1. CNN + BiLSTM (raw RGB frames)

**Pipeline:** Each frame `x_t` (H×W×3) passes through a 2D CNN backbone — historically a fine-tuned ImageNet network, in modern continuous sign-language recognition (CSLR) work usually a ResNet-style or lightweight backbone — producing a per-frame embedding `v_t ∈ R^d`. The sequence `{v_t}` then goes through a shallow 1D CNN (short-range temporal smoothing, kernel size 3–5) followed by a BiLSTM (typically 1–2 layers, hidden size 512–1024), which reads the sequence forward and backward and concatenates both directions per time step. A fully connected layer + softmax produces the class distribution for isolated-sign classification; for continuous/gloss-sequence recognition this is replaced with a CTC head over the BiLSTM outputs. This exact frame-CNN → 1D-CNN → BiLSTM → CTC/softmax scaffold is essentially the shared backbone used across most modern CSLR papers (VAC, SEN, CorrNet, MAM-FSD) — it's less a single model than the default recipe the field converges to.

**Why it's the weakest baseline here:** the CNN has to relearn "where the hands are" from raw pixels every time, so it inherits all the usual RGB failure modes — background clutter, lighting, camera angle, clothing contrast with skin tone — and needs substantially more labeled data to generalize than a landmark-based model does. On a few-thousand-clip ISL dataset this is the model most likely to overfit to nuisance variation rather than sign identity.

**References**
- Hu, Zhou, Wang, Ge & Chen (2022). *Self-Emphasizing Network for Continuous Sign Language Recognition* — canonical CNN→1D-CNN→BiLSTM→CTC backbone. [arXiv:2211.17081](https://arxiv.org/pdf/2211.17081)
- Hu, Zhou, Zhou, Wang & Li (2023). *Continuous Sign Language Recognition with Correlation Network*. [arXiv:2303.03202](https://arxiv.org/pdf/2303.03202)
- Al-Qurishi, Khalid & Souissi (2021). Attention-based Arabic SLR with CNN-BiLSTM, 85.6% signer-independent accuracy — representative of the CNN+BiLSTM accuracy ceiling on a modest dataset. Cited via [ResearchGate summary, Continuous Chinese sign language recognition with CNN-LSTM survey](https://www.researchgate.net/publication/318610001_Continuous_Chinese_sign_language_recognition_with_CNN-LSTM)
- Real-Time Lightweight Sign Language Recognition, CNN-BiLSTM with attention. [thesai.org, Vol 16 No 4](https://thesai.org/Downloads/Volume16No4/Paper_52-Real_Time_Lightweight_Sign_Language_Recognition.pdf)
- Dynamic Kannada Sign Language Recognition on Resource Constrained Devices — CNN+BiLSTM fusion of sensor + skeletal data. *Scientific Reports* (2026). [nature.com/articles/s41598-026-40181-7](https://www.nature.com/articles/s41598-026-40181-7)

---

### 2. MediaPipe + BiLSTM (flat landmark vector)

**Pipeline:** MediaPipe Holistic runs three separate models per frame — BlazePose for body pose, a palm-detector + hand-landmark model for 21 keypoints per hand, and a face-mesh model for 468 facial points — and concatenates whatever subset you keep (commonly pose + both hands, sometimes a reduced face subset) into a single flat vector per frame, typically 2D or 3D coordinates. The per-frame vectors are stacked into a `T × D` sequence and fed directly into a BiLSTM (again usually 1–2 layers), followed by dense layers and softmax. There is no explicit encoding of which joints are anatomically connected — the network has to learn joint relationships implicitly from co-occurrence in the flattened vector, which is exactly the "no explicit spatial structure" limitation your table notes.

**Why it's a good sanity check:** it isolates the temporal-modeling question from the spatial-modeling question — if this baseline already gets you most of the way to your GCN accuracy, that's a strong signal the anatomical graph isn't buying you much on your specific dataset, which is worth knowing before you build anything fancier.

**References**
- Goyal (2023). *Indian Sign Language Recognition Using MediaPipe Holistic*. [arXiv:2304.10256](https://arxiv.org/pdf/2304.10256)
- Rawat, Kumar, Tamta & Kumar (2025). *A Comprehensive Approach to ISL Recognition: Leveraging LSTM and MediaPipe Holistic*. EAI Trans. AI & Robotics. [publications.eai.eu](https://publications.eai.eu/index.php/airo/article/view/8693)
- Anonymous authors (2022). *An Integrated MediaPipe-Optimized GRU Model for Indian Sign Language Recognition (MOPGRU)*. *Scientific Reports*. [nature.com/articles/s41598-022-15998-7](https://www.nature.com/articles/s41598-022-15998-7)
- *MediaPipe's Landmarks with RNN for Dynamic Sign Language Recognition* — GRU/LSTM/BiLSTM comparison on DSL10. [ResearchGate](https://www.researchgate.net/publication/364279614_MediaPipe's_Landmarks_with_RNN_for_Dynamic_Sign_Language_Recognition)

---

### 3. MediaPipe + Transformer (self-attention over landmark sequence)

**Pipeline:** Same MediaPipe landmark front end as above, but the temporal encoder is a self-attention stack instead of a recurrent one — each per-frame landmark vector is linearly projected to `d_model`, given a sinusoidal or learned positional encoding over frame index, and passed through several standard transformer encoder blocks (multi-head self-attention + feed-forward, pre/post-LayerNorm, residual connections). A pooled representation (CLS token, mean pool, or final-frame token) feeds a classification head.

**Why it's more data-hungry than BiLSTM here:** self-attention has weaker built-in temporal-locality bias than a recurrent network, so it typically needs either a larger corpus or explicit pretraining to match BiLSTM sample efficiency at a few-thousand-clip scale. The two concrete recent examples below both had to work around this: TSLFormer keeps the model deliberately lightweight and validates on a large (36k-clip) corpus rather than a small one, and SignBart gets away with a tiny parameter count (≈750K) specifically to stay trainable on a small dataset.

**References**
- Vaswani et al. (2017). *Attention Is All You Need* — base transformer architecture.
- Alp Karaoğlu et al. (2025). *TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks*, ~90% on AUTSL (36k clips) using only MediaPipe landmarks. [arXiv:2505.07890](https://arxiv.org/pdf/2505.07890)
- Zhu et al. (2023). *Spatial-Temporal Graph Transformer for Skeleton-Based Sign Language Recognition (STGT)* — treats the skeleton as a fully-connected graph with graph positional embedding + graph multi-head self-attention. [Springer, LNCS](https://link.springer.com/chapter/10.1007/978-981-99-1645-0_12)
- (2025). *SignBart — skeleton-sequence isolated SLR via a BART encoder-decoder*, x/y coordinate streams encoded separately with cross-attention, 96.04% on LSA-64 with <750K parameters. [arXiv:2506.21592](https://arxiv.org/html/2506.21592)

---

## Tier 2 — Graph-based baselines (explicit anatomical structure)

### 4. ST-GCN (fixed skeleton graph)

**Pipeline:** ST-GCN represents each frame's skeleton as a graph `G=(V,E)` — joints as nodes, bones as edges — and stacks frames so that each joint also connects to itself in the previous and next frame (a temporal edge). A layer alternates a **spatial graph convolution** — aggregating each joint's features from its 1-hop neighbors, split into (per Yan et al.'s "spatial configuration partitioning") the root joint, centripetal neighbors (closer to the skeleton's center of gravity), and centrifugal neighbors (farther from it), each with its own learned weight matrix — with a standard **temporal convolution** (1D conv along the frame axis, same weights for every joint). A learnable per-edge importance mask scales the fixed adjacency matrix so the network can down-weight anatomically-connected-but-behaviorally-irrelevant edges. Nine such ST-GCN blocks are stacked in the original architecture, followed by global average pooling and a softmax classifier.

**Why it's the standard graph baseline, not a contribution:** it's the founding architecture of the entire skeleton-GCN literature — everything in row 5 below is explicitly a response to its two acknowledged limitations: the spatial receptive field is stuck at 1-hop neighbors, and the edge weights (beyond the importance mask) are fixed rather than learned per-input.

**References**
- Yan, Xiong & Lin (2018). *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition*. AAAI 2018. [arXiv:1801.07455](https://arxiv.org/abs/1801.07455) · [code](https://github.com/yysijie/st-gcn)

---

### 5. CTR-GCN / TD-GCN — the toughest baseline family (deep dive)

Your table calls this "likely the toughest baseline to beat," and it's worth understanding exactly why: this family replaces ST-GCN's single fixed adjacency matrix with a **learned, input-dependent topology**, and it does so at two different granularities that compose.

**CTR-GC (the base idea).** Chen et al.'s Channel-wise Topology Refinement Graph Convolution starts from the observation that different feature channels (think: different "types" of learned joint features) often want *different* joint-to-joint relationships — e.g., one channel might encode "which joints move together," another "which joints are anatomically adjacent." Instead of learning one adjacency matrix per layer, CTR-GC learns a single **shared topology** as a generic prior, then **refines it per channel** using a lightweight correlation-modeling function over pairwise channel-specific joint features (a small MLP applied to the difference between each pair of joints' channel-projected features, producing a per-channel correction to the shared topology). This adds very few extra parameters relative to a fully independent per-channel adjacency matrix, and the paper shows — via a unified reformulation of graph convolution — that this refinement relaxes some of the strict structural constraints that limit ST-GCN-style convolutions, giving it more representational capacity per parameter. Combined with standard multi-scale temporal convolution and a joint+bone+motion multi-stream fusion (a pattern inherited from 2s-AGCN), the resulting CTR-GCN network notably outperformed prior state of the art on NTU RGB+D, NTU RGB+D 120, and NW-UCLA.

**TD-GCN (the temporal extension).** Liu et al. observed that CTR-GC's refinement is channel-specific but still **frame-invariant** — the same adjacency matrix (per channel) is reused across every timestep, which limits how well the model can represent topology that genuinely changes over the course of a gesture (e.g., the effective "joint coupling" during the approach phase of a sign versus its hold phase). TD-GCN's fix is to make the adjacency matrix **both channel-dependent and temporal-dependent**: it computes distinct adjacency matrices per frame, conditioned on that frame's channel-projected features, so the graph topology itself evolves through the gesture rather than being fixed once per channel. This is specifically validated on **gesture** benchmarks (SHREC'17 Track, DHG-14/28) rather than full-body action datasets — a closer match to sign language's hand-centric, fine-grained motion than NTU RGB+D's whole-body actions — and it reports state-of-the-art results there, built directly on top of the CTR-GCN codebase.

**Why this specific family is dangerous to your Koopman claim:** it is, structurally, already doing something adjacent to what you're proposing — extracting a time-varying, data-driven description of how the joints are coupled — except it learns that coupling implicitly and discriminatively (whatever adjacency helps classification), whereas Koopman/DMD extracts it explicitly and with a physical interpretation (eigenvalues as frequency/decay, eigenvectors as coherent motion patterns). If TD-GCN already saturates accuracy on your dataset, your paper's contribution has to lean harder on interpretability and data efficiency rather than raw accuracy, since a channel-and-time-adaptive learned topology is a strong implicit competitor to an explicit dynamical decomposition. It's also worth knowing that this "learn a better adjacency matrix" direction kept extending after TD-GCN — most recently into **DSTSA-GCN** (Group Channel-wise + Group Temporal-wise graph convolution, plus multi-scale temporal convolution, again benchmarked on SHREC'17/DHG-14/28), which is the current frontier of exactly this family. Your proposal is right not to try to out-innovate this line with yet another adjacency-learning trick.

**References**
- Chen, Zhang, Yuan, Li, Deng & Hu (2021). *Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition*. ICCV 2021. [arXiv:2107.12213](https://arxiv.org/abs/2107.12213) · [code](https://github.com/Uason-Chen/CTR-GCN)
- Liu, Wang, Wang, Gao & Liu (2024). *Temporal Decoupling Graph Convolutional Network for Skeleton-Based Gesture Recognition (TD-GCN)*. IEEE Transactions on Multimedia. [IEEE Xplore, doi:10.1109/TMM.2023.3271811](https://ieeexplore.ieee.org/document/10113233/) · [code](https://github.com/liujf69/TD-GCN-Gesture)
- Cui et al. (2025). *DSTSA-GCN: Advancing Skeleton-Based Gesture Recognition with Semantic-Aware Spatio-Temporal Topology Modeling*. [arXiv:2501.12086](https://arxiv.org/abs/2501.12086)
- Shi, Zhang, Cheng & Lu (2019). *Two-Stream Adaptive Graph Convolutional Networks (2s-AGCN)* — origin of the joint+bone multi-stream fusion pattern CTR-GCN inherits.

---

### 6. HWGAT — your most directly relevant baseline (deep dive)

This is the baseline your proposal correctly flags as the one apples-to-apples comparison, because it was built and benchmarked specifically on Indian Sign Language.

**Dataset it was introduced with.** Patra et al. paired the model with a new large-scale isolated-ISL dataset (**FDMSE-ISL**): 2,002 commonly used words signed by 20 deaf adult signers, 40,033 videos total, with a train/val/test split constructed to have **no signer overlap** — a meaningfully harder and more realistic split than random per-clip splitting, since it forces the model to generalize across signing style rather than memorize a specific signer's execution.

**Architecture.** Where ST-GCN and the CTR-GCN family process spatial and temporal structure in alternating, separate steps, HWGAT (extended in a follow-up paper into "HWGATE," the Hierarchical Windowed Graph Attention Transformer Encoder) builds one **unified spatio-temporal graph** and aggregates both dimensions at once. The pipeline: (1) extract 27 keypoints per frame (10 per hand + 7 upper-body pose points) as 2D (x,y) coordinates via a pose estimator; (2) construct a spatio-temporal graph by connecting each frame's spatial keypoint graph to the corresponding keypoints in adjacent frames via temporal edges, so the graph spans both space and time simultaneously rather than being processed as separate spatial and temporal stages; (3) partition this graph into **spatial windows** grouped by combinations of body parts (e.g., left hand, right hand, upper body), directly borrowing the windowed-attention idea from the Swin Transformer's shifted local windows, so attention is computed locally within anatomically meaningful groups rather than globally over all keypoints at once — this is what keeps the attention computation tractable given a much denser graph than ST-GCN's single-frame skeleton; (4) apply positional encoding together with **Fourier feature mapping** (projecting raw coordinates through sinusoidal basis functions before the network sees them) to help the attention layers represent the high-frequency spatial detail that plain coordinate inputs struggle to capture, following Tancik et al.'s finding that raw low-dimensional coordinates are hard for networks to fit precisely without this kind of frequency-based feature expansion; (5) partition the temporal axis into smaller blocks and apply three stacked **part-attention** layers over the windowed graph; (6) average-pool and pass through a fully connected layer to predict the gloss.

**Why it's already a real benchmark, not just a plausible baseline:** the authors pretrained HWGAT on FDMSE-ISL and then fine-tuned it on four *other* published sign-language datasets — INCLUDE, LSA64, AUTSL, and WLASL — reporting accuracy gains of **+1.10, +0.46, +0.78, and +6.84 percentage points respectively** over the prior best keypoint-based models on each of those datasets. That cross-dataset transfer result is exactly the kind of number a reviewer will expect your Koopman-augmented model to be measured against if you evaluate on INCLUDE or a similar corpus — it's not a from-scratch baseline you're inventing, it's a documented state-of-the-art keypoint model with a public implementation.

**References**
- Patra, Maitra, Tiwari & Roy (2024/2025). *Hierarchical Windowed Graph Attention Network and a Large Scale Dataset for Isolated Indian Sign Language Recognition*. [arXiv:2407.14224](https://arxiv.org/pdf/2407.14224)
- Patra et al. (2025). *Hierarchical Windowed Graph Attention Transformer Encoder and a Large Scale Dataset for Indian Sign Language Recognition*. Pattern Analysis and Applications (Springer). [link.springer.com/article/10.1007/s10044-025-01529-3](https://link.springer.com/article/10.1007/s10044-025-01529-3) · [demo code](https://github.com/suvajit-patra/sl-hwgat-demo)
- Liu et al. (2021). *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows* — source of the windowed-attention design HWGAT borrows. ICCV 2021.
- Tancik et al. (2020). *Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains*. NeurIPS 2020 — source of the Fourier feature mapping used in HWGAT's input encoding.

---

## Tier 3 — Spectral / frequency-domain baselines

### 7. FFT + kinematic features + BiLSTM (the Libras recipe)

**Pipeline:** Rego et al. augment raw joint positions with explicit kinematic derivatives — per-joint velocity and acceleration computed by finite-differencing the position sequence — then apply an **adaptive sliding-window FFT** to each joint's trajectory, extracting the dominant frequency's magnitude and phase within each window. These spectral features are concatenated with the geometric (position) and kinematic (velocity/acceleration) features into one combined per-window feature vector, which feeds a BiLSTM, then fully connected layers with ReLU activations, then a softmax over the sign classes.

**Concrete numbers worth knowing:** evaluated on MINDS-Libras (20 Brazilian Sign Language classes, 10-fold cross-validation), this combined feature set reached **92% accuracy/F1**, versus 84% for a plain LSTM, 89% for plain BiLSTM, and 89% for plain CNN baselines run in the same paper. That ~3-point gain over bare BiLSTM is a useful, concrete number: it's roughly the size of the improvement plain spectral feature fusion buys you on a comparably small isolated-sign task, which gives you a rough lower bound your DMD-feature-fusion ablation (§2.3.1 in your proposal) should be judged against — if Hankel-DMD features can't clear a similar or larger margin over the same BiLSTM backbone, that's a red flag worth catching early.

**References**
- Rego, [et al.] (2025). *Brazilian Sign Language Recognition Using Deep Learning Based on Fast Fourier Transform and Kinematic Features*. IEEE Access. [doi:10.1109/ACCESS.2025.3637779](https://ieeexplore.ieee.org/document/11270928/)

---

### 8. CWT (continuous wavelet transform) + BiLSTM/Transformer

**What it changes relative to plain FFT.** The FFT (or a fixed-window STFT) assumes the signal is locally stationary within each analysis window and gives every frequency the same time resolution — problematic for signing motion, which starts, moves, and stops rather than oscillating steadily. The CWT instead convolves the signal with scaled and translated copies of a "mother wavelet" (commonly Morlet or Mexican-hat), producing a 2D time-scale representation (a **scalogram**) whose window width automatically shrinks at high frequencies and widens at low frequencies — giving good time localization for fast transients and good frequency localization for slow, sustained motion, rather than a single fixed trade-off. This is the standard signal-processing argument for why CWT features should already beat plain-FFT features on transient, non-stationary motion, and it's exactly why your proposal treats CWT (not FFT) as the harder, more appropriate baseline to clear.

**Pipeline as it would apply here:** per joint (or per selected channel), compute a CWT scalogram over a sliding window; either (a) flatten or pool the scalogram into a compact feature vector per window and feed a sequence of these into a BiLSTM (directly parallel to the FFT-feature pipeline above, with CWT coefficients substituted for FFT magnitude/phase), or (b) treat each window's scalogram as a small image and feed it through a lightweight CNN or patch embedding into a Transformer, echoing how CWT scalograms are used as CNN inputs in other gesture-recognition settings. There is no sign-language-specific published architecture combining CWT with a BiLSTM/Transformer backbone (which is exactly why your proposal correctly calls this "standard signal-processing tool, not sign-language-specific" rather than citing a canonical paper) — the architectural novelty here would be entirely in adapting a well-understood general tool to this domain, not in inventing new spectral math.

**References**
- Daubechies (1990). *The Wavelet Transform, Time-Frequency Localization and Signal Analysis*. IEEE Transactions on Information Theory — foundational CWT theory.
- Mallat (2008). *A Wavelet Tour of Signal Processing*, 3rd ed. — standard reference text.
- Ronao & Cho (2021-area work). *Human Activity Recognition Using Continuous Wavelet Transform and Convolutional Neural Networks* — CWT scalogram + CNN for IMU-based HAR, closest general-domain analogue. [arXiv:2106.12666](https://arxiv.org/pdf/2106.12666)
- (2026). *Real-Time Hand Gesture Recognition for IoT Devices Using FMCW mmWave Radar and Continuous Wavelet Transform* — CWT (Morlet) + lightweight CNN, 99.87% on a 5-gesture radar task; demonstrates the CWT-for-gesture pattern outside sign language specifically. [doi:10.3390/electronics15020250](https://doi.org/10.3390/electronics15020250)
- Garg, Ghosh & Pradhan (2024). *GestFormer: Multiscale Wavelet Pooling Transformer Network for Dynamic Hand Gesture Recognition*. CVPR Workshops 2024 — closest published wavelet+transformer gesture architecture, worth reading even though it's not CWT-scalogram-based in the same way.
- (2026). *Dual-Stream BiLSTM–Transformer Architecture for Real-Time Two-Handed Dynamic Sign Language Gesture Recognition* — a current BiLSTM+Transformer two-handed sign backbone a CWT front end could plausibly be paired with. [doi:10.3390/app16062912](https://doi.org/10.3390/app16062912)

---

### 9. Graph-Fourier attention (PGF-SLR-style) — closest published work to your framing (deep dive)

Your proposal calls this the closest existing work to "graph + spectral," and it's worth understanding precisely, because it's the paper most likely to come up in a reviewer's "isn't this already done?" question.

**What problem it targets.** Wei, Hu & Ma's PGF-SLR is built for **continuous** sign language recognition (recognizing a sequence of glosses from an unsegmented signing video), not isolated-sign classification — a different task setting than your other eight baselines, which matters when you cite it. Its motivation is twofold: RGB-input methods carry heavy compute cost and background sensitivity (the same complaint as row 1 above), and — more specifically — existing skeleton methods struggle to model the fact that different body parts move with **nonlinear, mutually asynchronous** dynamics during a sign (e.g., a handshape can hold steady while the arm is still moving to position, or facial expression can lead or lag manual articulation).

**Architecture.** PGF-SLR treats **each body part at each time step** as a separate graph node — not each individual joint, and not each full frame — so the graph is indexed jointly by (body part, time), rather than the usual (joint) or (joint, time) indexing used by ST-GCN/CTR-GCN. The edges of this graph are not spatial proximity or learned correlation in the CTR-GCN sense; instead, **frequency-domain attention between parts** defines the edges, forming what the authors call a "part-level Fourier fully connected graph" — every part-timestep node is potentially connected to every other, with edge strength coming from a Fourier/frequency-domain attention computation rather than a spatial-domain one. A graph Fourier learning module then jointly captures spatial (cross-part) and temporal dependencies simultaneously in the frequency domain, rather than factorizing them into separate spatial-then-temporal stages the way ST-GCN and most GCN variants do. On top of this, an **adaptive frequency enhancement** step selectively amplifies the frequency components that carry discriminative motion information, and a **dual-branch** design adds an auxiliary action-prediction branch that assists the main gloss-recognition branch during training (a form of auxiliary-task regularization). The paper reports relative improvements in the 2–4% range over prior methods on the continuous-SLR benchmarks it evaluates.

**Why this is the paper to differentiate against most carefully.** PGF-SLR is doing spectral analysis over an anatomical structure — which sounds, at a glance, close to what Hankel-DMD on landmark graphs does — but the mechanism is fundamentally different in a way you should state explicitly in your related-work section: PGF-SLR's "frequency domain" is used to build **attention weights** (a learned soft-edge weighting between graph nodes, still trained purely for classification), whereas your Koopman/DMD approach extracts an actual **linear dynamical operator** whose eigenvalues carry a physical decay-rate-and-frequency interpretation per mode, independent of any downstream classification objective. PGF-SLR's frequency features are a discriminative attention mechanism; DMD's frequency features are a generative/reconstructive decomposition of the trajectory itself, which is what lets you do the mode-perturbation augmentation (§3.2) and phonological mode-probing (§2.3.3) that PGF-SLR's architecture has no analogue for. That distinction — attention-weighting-in-frequency-space versus operator-eigendecomposition-in-frequency-space — is the sentence you want ready for a reviewer who's read this paper.

**References**
- Wei, Hu & Ma (2025). *Part-Wise Graph Fourier Learning for Skeleton-Based Continuous Sign Language Recognition (PGF-SLR)*. Journal of Imaging, 11(8):286. [doi:10.3390/jimaging11080286](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12387829/)

---

## How these baselines line up against the proposal

The three deep-dive models above are the ones worth running as actual comparison points before you commit compute to the full Koopman pipeline: CTR-GCN/TD-GCN and HWGAT because they're the accuracy ceiling to beat, and PGF-SLR because it's the paper a reviewer will hand you if your related-work section doesn't pre-empt the comparison. The others (ST-GCN, CNN+BiLSTM, both MediaPipe+sequence-model variants, and FFT+kinematic) are cheap to implement and mainly useful for the comparison table and for sanity-checking that your Kalman-filtered preprocessing (§3.1) and Hankel-DMD feature fusion (§2.1–2.3.1) are each independently pulling their weight before you stack them on top of a strong backbone.