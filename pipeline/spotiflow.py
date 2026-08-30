"""Run a Spotiflow 3D spot detector over a competition volume.

`notes/45`: the score is `J_adj = J · (1 − 0.1·(N_pred − N_total)/N_total)`, we sit at a
multiplier of 1.0012 against a ceiling of 1.1, and predictions that match no ground-truth
node are excluded from the edge term rather than penalised. So the prize goes to whoever
detects **the annotated cells specifically** and as little else as possible.

`altervation/biohub-r35-spotiflow` (MIT) does that with Spotiflow — a stereographic-flow
spot detector from the Weigert lab — pretrained on `synth_3d` and fine-tuned on this
competition's own GEFF spots. Its own docstring records zero-shot precision at ~0.002
before that fine-tune, which is the whole argument for the fine-tune mattering more than
the architecture.

This module is a thin loader and per-frame driver written against Spotiflow's public API.
It is **not** a copy of r35's `spotiflow_detect.py`, but the call convention below was
learned from reading it, and two of its hard-won details are reproduced deliberately
because getting either wrong is silent rather than loud:

* **Axis order.** Hub-pretrained `synth_3d` returns `(z, x, y)`; a local fine-tune trained
  on GEFF labels already returns `(z, y, x)`. Remapping the latter swaps Y and X again and
  produces detections that look reasonable and match nothing.
* **Scores must be 1-D.** If `details.prob` arrives shaped `(N, 1)`, `argsort` stays 2-D,
  `pts[order]` becomes `(N, 1, 3)`, and row iteration breaks downstream rather than here.

    Portions of the approach derive from altervation/biohub-r35-spotiflow, MIT licence.
"""
from __future__ import annotations

import numpy as np


def load(model_dir, device: str | None = None):
    """Load a Spotiflow fine-tune from a local folder.

    `from_folder` is required for a local fine-tune; `from_pretrained` resolves Hub names
    and would silently fetch a different model — with internet off it simply fails, which
    is the better outcome of the two.
    """
    from spotiflow.model import Spotiflow

    kw = {"device": device} if device is not None else {}
    try:
        return Spotiflow.from_folder(str(model_dir), **kw)
    except TypeError:
        return Spotiflow.from_folder(str(model_dir))


def _points_zyx(points, remap_zxy: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.size == 0:
        return np.zeros((0, 3), np.float32)
    if pts.ndim == 2 and pts.shape[0] == 3 and pts.shape[1] != 3:
        pts = pts.T
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"expected (N, 3+) spots, got {pts.shape}")
    if not remap_zxy:
        return np.ascontiguousarray(pts[:, :3])
    return np.ascontiguousarray(np.stack([pts[:, 0], pts[:, 2], pts[:, 1]], axis=1))


def _frame_scores(details, n: int) -> np.ndarray:
    """Peak probability if Spotiflow gave one, else ones. Always 1-D of length `n`."""
    if details is not None:
        for attr in ("prob", "intens"):
            raw = getattr(details, attr, None)
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float32).reshape(-1)
            if arr.shape[0] == n:
                return arr
    return np.ones(n, np.float32)


def detect_volume(model, volume, prob_thresh: float | None = 0.3,
                  remap_zxy: bool = False, per_frame_cap: int | None = None,
                  total_cap: int | None = None):
    """Detect spots in every frame of `volume` ``(T, Z, Y, X)``.

    Returns ``(t, zyx, scores)`` with `zyx` in **voxel index space** — the space the
    ground truth uses — so no rescaling is applied here.

    `per_frame_cap` and `total_cap` keep the highest-scoring detections. Both default to
    off: r35's own docstring warns that capping to 6–10 per frame gives **zero** recall,
    so a cap is a deliberate budget decision made by the caller, never a default.
    """
    kw = {}
    if prob_thresh is not None:
        kw["prob_thresh"] = float(prob_thresh)

    ts, zs, ss = [], [], []
    for t in range(int(volume.shape[0])):
        frame = np.asarray(volume[t], dtype=np.float32)
        points, details = model.predict(frame, **kw)
        pts = _points_zyx(points, remap_zxy)
        if len(pts) == 0:
            continue
        sc = _frame_scores(details, len(pts))
        order = np.argsort(-sc.reshape(-1))
        pts, sc = pts[order], sc.reshape(-1)[order]
        if per_frame_cap is not None and len(pts) > per_frame_cap:
            pts, sc = pts[:per_frame_cap], sc[:per_frame_cap]
        # Clip into the frame: Spotiflow can place a sub-voxel peak just outside.
        hi = np.asarray(frame.shape, dtype=np.float32) - 1.0
        pts = np.clip(pts, 0.0, hi)
        ts.append(np.full(len(pts), t, np.int64))
        zs.append(pts.astype(float))
        ss.append(sc.astype(float))

    if not ts:
        return np.zeros(0, np.int64), np.zeros((0, 3), float), np.zeros(0, float)
    t_all = np.concatenate(ts)
    z_all = np.concatenate(zs)
    s_all = np.concatenate(ss)

    if total_cap is not None and len(t_all) > total_cap:
        # A GLOBAL trim, not per frame: the budget term counts total nodes, and frames
        # differ enormously in how many annotated cells they contain.
        keep = np.argsort(-s_all)[:total_cap]
        keep.sort()
        t_all, z_all, s_all = t_all[keep], z_all[keep], s_all[keep]
    return t_all, z_all, s_all


def node_budget(cells_per_frame: float, n_frames: int, margin: float = 1.26,
                lo: int = 50, per_frame_max: int = 7) -> int:
    """Total node budget from an estimated ANNOTATION density.

    `notes/45`: the edge term ignores predictions matching no ground-truth node, so the
    target is the annotated count, not the true cell count. On this competition's sparse
    embryo the annotation runs 0.5–1.7 cells per frame against ~240 real ones, which is
    why a budget in the low hundreds is not the absurdity it looks like.

    The shape (margin, floor, per-frame ceiling) follows r35's `predict_total_node_budget`.
    """
    cpf = max(0.0, float(cells_per_frame))
    n_frames = max(1, int(n_frames))
    return int(np.clip(round(cpf * n_frames * margin), lo, n_frames * per_frame_max))
