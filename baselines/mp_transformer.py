"""MediaPipe landmarks + Transformer encoder (arch.md §3)."""
from __future__ import annotations

from .heads import TransformerClassifier


class MPTransformer(TransformerClassifier):
    """Self-attention over per-frame landmark vectors with sinusoidal positions."""
