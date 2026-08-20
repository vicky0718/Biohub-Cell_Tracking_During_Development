"""Validation harness for the Biohub cell-tracking contest.

Importing this package must NOT require `tracksdata` — it cannot be installed on
Kaggle (numpy pin), and Kaggle is where the work runs. `harness.scorer`, which wraps
the organisers' code, is the only module that needs it and is imported lazily by
`Harness` when the official path is asked for. See `harness/purescore.py`.
"""

from . import csvout, purescore
from .csvout import check_graph, write_submission
from .harness import Harness, Result, Verdict, gate, load_geff
from .purescore import DEFAULT_SCALE
from .tracks import Tracks, read_estimated_nodes, read_geff, read_scale

# `submission` needs polars. Unlike every other module here it is not needed to PRODUCE
# a submission, only to pretty-print and validate one — and the scored rerun runs with
# internet off, so anything missing there stays missing. Degrade instead of taking the
# whole package down; `csvout` above is the stdlib-only path.
try:
    from .submission import build_submission, validate_submission
except ImportError as _e:  # pragma: no cover
    _POLARS_ERR = _e

    def build_submission(*a, **k):
        raise ImportError(f"harness.submission needs polars ({_POLARS_ERR}); "
                          "use harness.csvout.write_submission instead")

    def validate_submission(*a, **k):
        raise ImportError(f"harness.submission needs polars ({_POLARS_ERR}); "
                          "harness.csvout.check_graph covers the per-graph checks")

__all__ = [
    "Harness", "Result", "Verdict", "gate", "load_geff",
    "Tracks", "read_geff", "read_estimated_nodes", "read_scale",
    "DEFAULT_SCALE", "purescore",
    "build_submission", "validate_submission",
    "csvout", "write_submission", "check_graph",
]
