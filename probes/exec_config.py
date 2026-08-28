"""Prove the two config levers `notes/39`'s audit found untested.

Both are changes to code that is already in every submission, so the load-bearing test is
not "does the new thing work" but **"is the old behaviour byte-identical when the new knob
is off"**. A `close_gaps` that quietly differs at `max_gap=1` would re-tune the whole repair
chain underneath a sweep that claims to be measuring one variable.

  * `close_gaps(max_gap=2)` — the 0.927 notebook runs `GAP_CLOSE_MAX_GAP = 2`; we have only
    ever bridged one-frame holes.
  * `prune_short_tracks` — it runs `OUTPUT_MIN_TRACK_LEN = 6` with short-track filtering on;
    we have never pruned anything at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.repair import close_gaps, prune_short_tracks

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    S = (1.0, 1.0, 1.0)

    print("=" * 78)
    print("close_gaps(max_gap=1) must be UNCHANGED — everything scored so far used it")
    print("=" * 78)
    rng = np.random.default_rng(0)
    for trial in range(30):
        n = rng.integers(20, 60)
        t = rng.integers(0, 12, n).astype(np.int64)
        zyx = rng.random((n, 3)) * 30
        m = rng.integers(0, max(1, n // 2))
        e = np.stack([rng.integers(0, n, m), rng.integers(0, n, m)], 1).astype(np.int64)
        e = e[e[:, 0] != e[:, 1]]
        a = close_gaps(t, zyx, e, scale=S, max_um=9.0)
        b = close_gaps(t, zyx, e, scale=S, max_um=9.0, max_gap=1)
        same = (np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1])
                and np.array_equal(a[2], b[2]))
        if not same:
            check(f"default == max_gap=1 on random graph {trial}", False)
            break
    else:
        check("default == max_gap=1 on 30 random graphs", True,
              "the default path is untouched, so every prior measurement still stands")

    print()
    print("=" * 78)
    print("close_gaps(max_gap=2) bridges a two-frame hole")
    print("=" * 78)
    # A tail at t=0 and a head at t=3: a two-frame hole needing TWO inserted nodes.
    t = np.array([0, 3], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    e = np.zeros((0, 2), np.int64)

    o1 = close_gaps(t, zyx, e, scale=S, max_um=9.0, max_added_frac=1.0, max_gap=1)
    check("max_gap=1 leaves a two-frame hole alone", len(o1[0]) == 2 and len(o1[2]) == 0)

    o2 = close_gaps(t, zyx, e, scale=S, max_um=9.0, max_added_frac=1.0, max_gap=2)
    check("max_gap=2 inserts TWO nodes", len(o2[0]) == 4, f"{len(o2[0])} nodes")
    check("at the missing frames", sorted(o2[0].tolist()) == [0, 1, 2, 3],
          f"{sorted(o2[0].tolist())}")
    check("interpolated evenly along the span",
          np.allclose(sorted(o2[1][:, 2]), [0.0, 1.0, 2.0, 3.0]),
          f"{sorted(np.round(o2[1][:, 2], 3).tolist())}")
    check("chained head-to-tail with no branch",
          len(o2[2]) == 3
          and len(set(o2[2][:, 0].tolist())) == 3
          and len(set(o2[2][:, 1].tolist())) == 3,
          f"edges {o2[2].tolist()}")

    print()
    print("=" * 78)
    print("ranking and budget")
    print("=" * 78)
    # One tail at t=0. A one-frame bridge to t=2 and a two-frame bridge to t=3, the
    # two-frame one geometrically CLOSER. Gap must outrank distance.
    t = np.array([0, 2, 3], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 6.0], [0.0, 0.0, 1.0]])
    o = close_gaps(t, zyx, np.zeros((0, 2), np.int64), scale=S, max_um=9.0,
                   max_added_frac=1.0, max_gap=2)
    inserted_t = sorted(o[0][3:].tolist()) if len(o[0]) > 3 else []
    check("a one-frame bridge outranks a closer two-frame one",
          len(o[0]) == 4 and inserted_t == [1],
          f"inserted at frames {inserted_t} — [1] is the 1-frame bridge, [1,2] the 2-frame")

    # A two-frame gap costs two nodes and must respect the budget.
    t = np.array([0, 3], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    tight = close_gaps(t, zyx, np.zeros((0, 2), np.int64), scale=S, max_um=9.0,
                       max_added_abs=1, max_added_frac=1.0, max_gap=2)
    check("a two-node bridge is refused when only one node of budget remains",
          len(tight[0]) == 2, f"{len(tight[0])} nodes with budget 1")

    print()
    print("=" * 78)
    print("prune_short_tracks")
    print("=" * 78)
    # a 5-frame chain, a 2-frame fragment, and a 2-frame component containing a fork
    t = np.array([0, 1, 2, 3, 4, 0, 1, 0, 1, 1], np.int64)
    zyx = np.zeros((10, 3)); zyx[:, 0] = np.arange(10)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [5, 6], [7, 8], [7, 9]], np.int64)

    keep = prune_short_tracks(t, zyx, e, min_frames=5, keep_division_components=True)
    check("the long chain survives", len(keep[0]) == 8, f"{len(keep[0])} nodes")
    check("the short NON-division fragment is dropped",
          5 not in set(np.flatnonzero(np.isin(np.arange(10), [])).tolist()) and len(keep[2]) == 6,
          f"{len(keep[2])} edges left of 7")
    drop = prune_short_tracks(t, zyx, e, min_frames=5, keep_division_components=False)
    check("with keep_division off the fork component goes too", len(drop[0]) == 5,
          f"{len(drop[0])} nodes")

    check("min_frames<=1 is a no-op",
          len(prune_short_tracks(t, zyx, e, min_frames=1)[0]) == 10)
    check("edges are remapped, never left dangling",
          keep[2].max() < len(keep[0]) if len(keep[2]) else True,
          "a stale index here is silent and catastrophic")

    # Span is FRAMES, not node count: eight nodes in one frame is not an eight-frame track.
    t2 = np.array([0] * 8, np.int64)
    z2 = np.zeros((8, 3)); z2[:, 0] = np.arange(8)
    e2 = np.zeros((0, 2), np.int64)
    check("eight nodes in ONE frame is a 1-frame track, not an 8-frame one",
          len(prune_short_tracks(t2, z2, e2, min_frames=5)[0]) == 0,
          "counting nodes instead of frames would keep exactly the clutter this removes")

    print()
    print("=" * 78)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("both config levers behave, and the old path is untouched")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
