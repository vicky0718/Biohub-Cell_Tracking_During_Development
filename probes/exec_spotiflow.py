"""Prove the Spotiflow driver without spotiflow installed.

`spotiflow` is a compiled cp312 wheel that only exists inside r35's dataset on Kaggle, so
the only thing testable here is the part that has actually gone wrong in this project
before: **shape and axis handling around a model call**, not the model call itself.

Three of this session's failures were exactly that — `peaks_from_prob` returning a tuple,
`detect_frame_dog` returning a tuple, and an `(N,1)` score array turning `pts[order]` into
`(N,1,3)`. A fake model reproduces each of those shapes deliberately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.spotiflow import _frame_scores, _points_zyx, detect_volume, node_budget

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


class Details:
    def __init__(self, prob):
        self.prob = prob


class FakeModel:
    """Returns whatever shapes we tell it to, which is the point."""

    def __init__(self, pts_per_frame, prob=None, transposed=False):
        self.pts_per_frame, self.prob, self.transposed = pts_per_frame, prob, transposed
        self.seen_kwargs = None

    def predict(self, frame, **kw):
        self.seen_kwargs = kw
        pts = np.asarray(self.pts_per_frame, dtype=np.float32)
        if self.transposed:
            pts = pts.T
        p = self.prob
        return pts, (Details(p) if p is not None else None)


def main() -> int:
    print("=" * 78)
    print("axis handling — the failure that produces plausible-looking nothing")
    print("=" * 78)
    pts = np.array([[1.0, 2.0, 3.0]], np.float32)
    check("a local fine-tune is left alone (already zyx)",
          np.array_equal(_points_zyx(pts, False), pts))
    check("hub pretrained is remapped z,x,y -> z,y,x",
          np.array_equal(_points_zyx(pts, True), np.array([[1.0, 3.0, 2.0]], np.float32)),
          "remapping a local fine-tune would swap Y/X and match nothing")
    check("a (3, N) array is transposed to (N, 3)",
          _points_zyx(np.zeros((3, 5), np.float32), False).shape == (5, 3))
    check("empty input gives (0, 3)", _points_zyx(np.zeros((0,)), False).shape == (0, 3))
    try:
        _points_zyx(np.zeros((4, 2), np.float32), False)
        check("a (N, 2) array raises", False, "it did not")
    except ValueError:
        check("a (N, 2) array raises", True)

    print()
    print("=" * 78)
    print("scores must come back 1-D")
    print("=" * 78)
    s = _frame_scores(Details(np.zeros((4, 1), np.float32)), 4)
    check("an (N,1) prob array is flattened", s.ndim == 1 and len(s) == 4,
          "if this stays 2-D, pts[order] becomes (N,1,3) and breaks far downstream")
    check("a mismatched-length prob is ignored",
          np.array_equal(_frame_scores(Details(np.zeros(9, np.float32)), 4), np.ones(4)))
    check("no details gives ones", np.array_equal(_frame_scores(None, 3), np.ones(3)))

    print()
    print("=" * 78)
    print("detect_volume")
    print("=" * 78)
    vol = np.zeros((5, 8, 16, 16), np.float32)
    three = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 1.0, 2.0]], np.float32)
    m = FakeModel(three, prob=np.array([0.2, 0.9, 0.5], np.float32))
    t, zyx, sc = detect_volume(m, vol, prob_thresh=0.3)
    check("one row per detection per frame", len(t) == 15 and zyx.shape == (15, 3))
    check("prob_thresh reaches the model", m.seen_kwargs == {"prob_thresh": 0.3})
    check("frames are labelled 0..T-1", sorted(set(t.tolist())) == [0, 1, 2, 3, 4])
    check("sorted by score within a frame",
          np.allclose(sc[:3], [0.9, 0.5, 0.2]),
          "the caps keep a PREFIX, so the order is what makes them keep the best")

    t2, z2, s2 = detect_volume(m, vol, per_frame_cap=1)
    check("per_frame_cap keeps one per frame", len(t2) == 5)
    check("and keeps the HIGHEST-scoring one", np.allclose(s2, 0.9))

    t3, z3, s3 = detect_volume(m, vol, total_cap=4)
    check("total_cap trims globally", len(t3) == 4, f"{len(t3)} kept")
    # Five frames each scoring [0.9, 0.5, 0.2], so the best four GLOBALLY are four 0.9s —
    # not one per frame. That distinction is the whole difference between total_cap and
    # per_frame_cap, and writing the expectation wrong the first time is what made it
    # worth asserting.
    check("total_cap keeps the best 4 overall", np.allclose(sorted(s3), [0.9, 0.9, 0.9, 0.9]))
    check("total_cap is NOT one-per-frame", len(set(t3.tolist())) < 5,
          "if it kept one per frame this would be 5 distinct frames")
    check("and leaves them in frame order", np.all(np.diff(t3) >= 0),
          "a graph builder that assumes time order gets nonsense otherwise")

    out = detect_volume(FakeModel(np.zeros((0, 3), np.float32)), vol)
    check("a model that finds nothing returns empty, not a crash",
          out[0].shape == (0,) and out[1].shape == (0, 3))

    big = FakeModel(np.array([[99.0, 99.0, 99.0]], np.float32))
    _, zb, _ = detect_volume(big, vol)
    check("out-of-frame peaks are clipped inside",
          zb.max(axis=0).tolist() == [7.0, 15.0, 15.0],
          "sub-voxel peaks can land outside; an unclipped index matches nothing")

    tt = FakeModel(three, prob=np.array([0.2, 0.9, 0.5], np.float32), transposed=True)
    check("a (3, N) model output still works",
          len(detect_volume(tt, vol)[0]) == 15)

    print()
    print("=" * 78)
    print("node_budget — the annotation count, not the cell count")
    print("=" * 78)
    check("0.5 cells/frame over 100 frames -> the floor", node_budget(0.5, 100) == 63,
          f"got {node_budget(0.5, 100)}")
    check("a dense estimate is capped at 7/frame", node_budget(50.0, 100) == 700)
    check("a tiny estimate cannot go below the floor", node_budget(0.0, 100) == 50)
    check("it scales with frame count", node_budget(2.0, 50) == 126,
          f"got {node_budget(2.0, 50)}")
    check("the budget is far below N_total (~24,000) — that is the whole point",
          node_budget(1.7, 100) < 1000)

    print()
    print("=" * 78)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("the driver handles every shape that has bitten this project")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
