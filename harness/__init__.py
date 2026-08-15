"""Validation harness for the Biohub cell-tracking contest.

Importing this package must NOT require `tracksdata` — it cannot be installed on
Kaggle (numpy pin), and Kaggle is where the work runs. `harness.scorer`, which wraps
the organisers' code, is the only module that needs it and is imported lazily by
`Harness` when the official path is asked for. See `harness/purescore.py`.
"""

from . import purescore
from .harness import Harness, Result, Verdict, gate, load_geff
from .purescore import DEFAULT_SCALE
from .submission import build_submission, validate_submission
from .tracks import Tracks, read_estimated_nodes, read_geff, read_scale

__all__ = [
    "Harness", "Result", "Verdict", "gate", "load_geff",
    "Tracks", "read_geff", "read_estimated_nodes", "read_scale",
    "DEFAULT_SCALE", "purescore",
    "build_submission", "validate_submission",
]
