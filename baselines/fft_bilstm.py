"""FFT + kinematic features + BiLSTM (arch.md §7, Libras recipe)."""
from __future__ import annotations

from .heads import BiLSTMClassifier


class FFTBiLSTM(BiLSTMClassifier):
    """Sequence model over sliding-window FFT magnitude/phase + pos/vel/acc."""
