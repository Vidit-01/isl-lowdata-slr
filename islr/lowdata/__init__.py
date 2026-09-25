"""Low-data study: landmark stores, data sources, signer-independent shot protocols,
fixed-budget training (single GPU, one-job-per-GPU sweeps, or DDP), and reports.

Entry point: ``python lowdata.py <command>`` from the repository root.
Nothing in this package imports torch at module import time except the training
modules, so extraction can run in a MediaPipe-only environment.
"""
