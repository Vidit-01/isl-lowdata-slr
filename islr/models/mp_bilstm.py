"""MediaPipe landmarks + BiLSTM (arch.md §2). Flat joint vector, no graph."""
from __future__ import annotations

from .heads import BiLSTMClassifier


class MPBiLSTM(BiLSTMClassifier):
    """Thin alias so the registry can name the architecture explicitly."""
