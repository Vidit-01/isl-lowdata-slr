"""Signer-independent low-data protocols with nested shot subsets.

Guarantees (checked by `smoke_test`):
  1. The test set depends only on (clip table, seed, test parameters). It is the same
     for every shot count K, every protocol that shares the vocabulary, and every model.
  2. Test identities (signer > session > clip, see `store.identity`) never appear in
     training.
  3. Training subsets are nested: the K=2 clips of a word contain its K=1 clip, and
     so on. Each word's pool is ordered once per (seed, word) by interleaving its
     identities round-robin, so small K already spans several signers.

Protocols
  uniform       every word gets K training clips
  scarce        target words get K clips, the other ("rich") words keep all of their
                pool (or `rich_shots`); this is the "few words with 8 videos vs many
                words with more data" experiment generalised over K
  cross_source  train on `train_sources`, test on every clip of `test_sources`
  full          all pool clips of every word (upper bound; pre-training)
`n_words` limits the vocabulary for any protocol (the target words are always kept,
the others are added in a fixed per-seed order), which gives the vocabulary sweep.
"""
from __future__ import annotations

import hashlib
import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

LEGACY8 = ("eat", "go", "hello", "help", "no", "please", "water", "yes")  # IIPD/ISL_DATASET_8WORDS
TARGET_PRESETS = {"legacy8": LEGACY8}
PROTOCOLS = ("uniform", "scarce", "cross_source", "full")


@dataclass
class SplitConfig:
    protocol: str = "scarce"
    shots: int | None = 4  # None = all pool clips
    seed: int = 0
    targets: list[str] | str = "legacy8"
    rich_shots: int | None = None
    n_words: int | None = None
    words: list[str] | None = None
    min_clips: int = 3  # a word needs this many clips to be used at all
    test_frac: float = 0.25
    min_test: int = 2
    max_test: int | None = 20
    max_test_frac: float = 0.5
    train_sources: list[str] | None = None
    test_sources: list[str] | None = None

    def target_list(self) -> list[str]:
        t = self.targets
        if isinstance(t, str):
            t = TARGET_PRESETS.get(t, [w.strip() for w in t.split(",") if w.strip()])
        return sorted({str(w).strip().lower() for w in t})

    def tag(self) -> str:
        """Short name of the protocol variant (without K and seed)."""
        parts = [self.protocol]
        if self.protocol in ("scarce",):
            tg = self.targets if isinstance(self.targets, str) and self.targets in TARGET_PRESETS else \
                f"t{len(self.target_list())}"
            parts.append(tg)
            if self.rich_shots is not None:
                parts.append(f"rich{self.rich_shots}")
        if self.protocol == "cross_source":
            parts.append("-".join(self.train_sources or []) + "_to_" + "-".join(self.test_sources or []))
        if self.n_words:
            parts.append(f"V{self.n_words}")
        return "_".join(parts)


@dataclass
class Split:
    cfg: SplitConfig
    train_idx: np.ndarray
    test_idx: np.ndarray
    words: list[str]  # label order
    targets: list[str]  # target words present in the label space
    info: dict = field(default_factory=dict)

    @property
    def label_of(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.words)}


def _fp(keys) -> str:
    return hashlib.sha1("\n".join(sorted(map(str, keys))).encode()).hexdigest()[:12]


def _rng(seed: int, *salt) -> np.random.Generator:
    return np.random.default_rng([int(seed)] + [zlib.crc32(str(s).encode()) for s in salt])


def eligible_words(df: pd.DataFrame, min_clips: int) -> list[str]:
    c = df.groupby("word").size()
    return sorted(c[c >= min_clips].index)


def fixed_test(df: pd.DataFrame, seed: int, test_frac: float = 0.25, min_test: int = 2,
               max_test: int | None = 20, max_test_frac: float = 0.5) -> set[str]:
    """Identities held out for testing under `seed`.

    Identities are visited in a seeded random order; one is moved to the test side if
    it adds clips to a word that is still below its test target, and no word it
    touches would end up with more than `max_test_frac` of its clips in test. Big
    identities (an ISL500 signer covers most words) make the result coarse, which is
    the price of signer independence.
    """
    counts = df.groupby("word").size()
    target = {w: int(min(max(min_test, round(test_frac * n)), max_test or n)) for w, n in counts.items()}
    cap = {w: max(target[w], int(np.floor(max_test_frac * n))) for w, n in counts.items()}
    per_id = df.groupby(["identity", "word"]).size()
    ids = sorted(df["identity"].unique())
    order = _rng(seed, "test").permutation(len(ids))
    have = dict.fromkeys(counts.index, 0)
    test: set[str] = set()
    for i in order:
        ident = ids[i]
        wc = per_id.loc[ident]
        helps = any(have[w] < target[w] for w in wc.index)
        fits = all(have[w] + int(n) <= cap[w] for w, n in wc.items())
        if helps and fits:
            test.add(ident)
            for w, n in wc.items():
                have[w] += int(n)
    return test


def pool_order(pool: pd.DataFrame, seed: int, word: str) -> list[int]:
    """Fixed order of a word's pool clips: identities shuffled, then round-robin."""
    rng = _rng(seed, "pool", word)
    groups = {}
    for ident, g in pool.groupby("identity", sort=True):
        idx = g.sort_values("key").index.to_numpy()
        groups[ident] = list(idx[rng.permutation(len(idx))])
    names = sorted(groups)
    names = [names[i] for i in rng.permutation(len(names))]
    out = []
    while any(groups[n] for n in names):
        for n in names:
            if groups[n]:
                out.append(int(groups[n].pop(0)))
    return out


def make_split(df: pd.DataFrame, cfg: SplitConfig) -> Split:
    if cfg.protocol not in PROTOCOLS:
        raise ValueError(f"protocol must be one of {PROTOCOLS}")
    data = df
    if cfg.words:
        data = data[data["word"].isin(set(cfg.words))]
    targets_all = cfg.target_list()

    if cfg.protocol == "cross_source":
        if not cfg.train_sources or not cfg.test_sources:
            raise ValueError("cross_source needs train_sources and test_sources")
        tr = data[data["source"].isin(cfg.train_sources)]
        te = data[data["source"].isin(cfg.test_sources)]
        vocab = sorted(set(eligible_words(tr, 1)) & set(te["word"]))
        pool, test_rows = tr[tr["word"].isin(vocab)], te[te["word"].isin(vocab)]
    else:
        vocab = eligible_words(data, cfg.min_clips)
        data = data[data["word"].isin(vocab)]
        # the test set depends on the eligible clip table and the seed only
        test_ids = fixed_test(data, cfg.seed, cfg.test_frac, cfg.min_test, cfg.max_test, cfg.max_test_frac)
        is_test = data["identity"].isin(test_ids)
        pool, test_rows = data[~is_test], data[is_test]

    if cfg.n_words:
        keep = [w for w in vocab if w in targets_all]
        rest = [w for w in vocab if w not in targets_all]
        rest = [rest[i] for i in _rng(cfg.seed, "vocab").permutation(len(rest))]
        vocab = sorted(keep + rest[: max(0, cfg.n_words - len(keep))])

    train, eff = [], {}
    for w in vocab:
        order = pool_order(pool[pool["word"] == w], cfg.seed, w)
        if cfg.protocol == "scarce":
            k = cfg.shots if w in targets_all else cfg.rich_shots
        elif cfg.protocol == "full":
            k = None
        else:
            k = cfg.shots
        take = order if k is None else order[:k]
        eff[w] = len(take)
        train += take
    words = sorted(w for w in vocab if eff.get(w, 0) > 0)
    test_rows = test_rows[test_rows["word"].isin(words)]
    train_idx = np.asarray(sorted(train), dtype=np.int64)
    test_idx = test_rows.index.to_numpy(dtype=np.int64)

    leak = sorted(set(df.loc[train_idx, "identity"]) & set(df.loc[test_idx, "identity"]))
    if cfg.protocol != "cross_source" and leak:
        raise AssertionError(f"identity leakage between train and test: {leak[:5]}")
    targets = [w for w in targets_all if w in words]
    short = {w: n for w, n in eff.items() if cfg.shots and w in (targets if cfg.protocol == "scarce" else words)
             and n < cfg.shots}
    test_counts = test_rows.groupby("word").size().reindex(words, fill_value=0)
    info = {
        "config": asdict(cfg),
        "tag": cfg.tag(),
        "n_words": len(words),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_test_target": int(test_rows["word"].isin(targets).sum()),
        "targets": targets,
        "missing_targets": [w for w in targets_all if w not in words],
        "words_without_test": [w for w in words if test_counts[w] == 0],
        "shots_short": short,  # words with fewer pool clips than K
        "train_per_word": {w: int(eff[w]) for w in words},
        "test_per_word": {w: int(n) for w, n in test_counts.items()},
        "test_fingerprint": _fp(df.loc[test_idx, "key"]),
        "train_fingerprint": _fp(df.loc[train_idx, "key"]),
        "n_test_identities": int(df.loc[test_idx, "identity"].nunique()),
        "n_train_identities": int(df.loc[train_idx, "identity"].nunique()),
    }
    return Split(cfg, train_idx, test_idx, words, targets, info)


def split_config_from(d: dict) -> SplitConfig:
    known = SplitConfig.__dataclass_fields__
    return SplitConfig(**{k: v for k, v in d.items() if k in known})
