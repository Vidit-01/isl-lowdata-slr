"""CWT scalogram features + BiLSTM or Transformer (arch.md §8)."""
from __future__ import annotations

from .heads import BiLSTMClassifier, TransformerClassifier


class CWTBiLSTM(BiLSTMClassifier):
    """Band-pooled Morlet CWT + kinematics, then BiLSTM."""


class CWTTransformer(TransformerClassifier):
    """Same CWT front-end, self-attention temporal encoder."""
