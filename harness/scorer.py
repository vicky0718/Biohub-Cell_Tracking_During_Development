"""Thin adapter onto the OFFICIAL scorer.

Every metric number this project reports comes through here, so that a local
measurement is the same computation the leaderboard runs. Nothing in this file
reimplements a metric — it only locates the official code and reads the two
pieces of dataset metadata the scorer needs (voxel scale, node budget).

Point `CELLMOT_REPO` at a clone of
https://github.com/royerlab/kaggle-cell-tracking-competition
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # z, y, x µm/px

_CANDIDATES = [
    os.environ.get("CELLMOT_REPO"),
    "/kaggle/working/kaggle-cell-tracking-competition",
    "./kaggle-cell-tracking-competition",
    "../kaggle-cell-tracking-competition",
]


def _locate_official_scorer() -> Path:
    for cand in _CANDIDATES:
        if not cand:
            continue
        p = Path(cand)
        if (p / "src" / "tracking_cellmot").is_dir():
            return p
    raise ImportError(
        "Could not find the official scorer. Clone it and set CELLMOT_REPO:\n"
        "  git clone https://github.com/royerlab/kaggle-cell-tracking-competition\n"
        "  export CELLMOT_REPO=$PWD/kaggle-cell-tracking-competition\n"
        "We deliberately do not vendor or reimplement the metric — a local score has to "
        "be the leaderboard's score, or measuring it is worse than useless."
    )


CELLMOT_REPO = _locate_official_scorer()
if str(CELLMOT_REPO / "src") not in sys.path:
    sys.path.insert(0, str(CELLMOT_REPO / "src"))

from tracking_cellmot.metrics import (  # noqa: E402
    ADJUSTMENT_ALPHA,
    SCORE_DIVISION_WEIGHT,
    evaluate,
    node_recall,
    per_sample_metrics,
    summarise,
)


def read_scale(zarr_path: Path | str) -> tuple[float, float, float]:
    """(z, y, x) µm/px from OME-Zarr metadata; DEFAULT_SCALE if unavailable.

    Reads metadata only — never loads pixels.
    """
    try:
        import zarr

        attrs = dict(zarr.open_group(str(zarr_path), mode="r").attrs)
        if "multiscales" in attrs:
            tr = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
            if tr.get("type") == "scale":
                return tuple(tr["scale"][-3:])
    except Exception:
        pass
    return DEFAULT_SCALE


def read_estimated_nodes(geff_path: Path | str) -> float:
    """The scorer's node budget (`estimated_number_of_nodes`), or NaN if absent."""
    try:
        from geff import GeffMetadata

        meta = GeffMetadata.read(str(geff_path))
        val = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(val) if val is not None else float("nan")
    except Exception:
        return float("nan")


__all__ = [
    "ADJUSTMENT_ALPHA", "SCORE_DIVISION_WEIGHT", "CELLMOT_REPO", "DEFAULT_SCALE",
    "evaluate", "node_recall", "per_sample_metrics", "summarise",
    "read_scale", "read_estimated_nodes",
]
