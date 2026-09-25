"""Launcher for the low-data study (islr/lowdata). Run from the repository root.

    python lowdata.py smoke                      # every model + protocols + DDP, one command
    python lowdata.py sources list               # INCLUDE zips on Zenodo
    python lowdata.py data --grid islr/lowdata/configs/isl40_scarce.json --dest stores   # one member's data
    python lowdata.py sources isl40 --root ISL_DATASET_40WORDS --store outputs/stores/isl40
    python lowdata.py run --stores outputs/stores/isl40 --model partformer --protocol scarce --shots 4
    python lowdata.py sweep --grid islr/lowdata/configs/scarce_words.json --out sweeps/scarce
    python lowdata.py report --out sweeps/scarce
    torchrun --standalone --nproc_per_node=2 lowdata.py run --stores ... --model kdf_transformer

`extract` and `sources` need only MediaPipe + OpenCV + pandas (no torch).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

COMMANDS = {
    "extract": "islr.lowdata.extract",
    "sources": "islr.lowdata.sources",
    "data": "islr.lowdata.prepare",
    "run": "islr.lowdata.run",
    "sweep": "islr.lowdata.sweep",
    "report": "islr.lowdata.report",
    "smoke": "islr.lowdata.smoke_test",
    "inspect": "islr.lowdata.inspect_data",
}


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        print("commands:", ", ".join(COMMANDS))
        return 2
    import importlib

    mod = importlib.import_module(COMMANDS[sys.argv[1]])
    return mod.main(sys.argv[2:]) or 0


if __name__ == "__main__":
    sys.exit(main())
