"""Tests for `pipeline/divisions.py`.

The two constraints these exist to protect are not stylistic. `notes/25` §1 reads them off
the official `division_metrics.py`: a fork whose sister already has a parent is
`malformed`, which is an **automatic false positive** — strictly worse than not forking at
all — and an edge that does not span exactly one frame is silently dropped. Everything
else here is about the sweep meaning what it says: `cap=0` has to be an exact no-op or the
probe has no control, and the cap has to bind per frame or the sweep axis is fictional.

Pure numpy; scipy only if present (both `_pairs_within` paths are exercised).

    python tests/test_divisions.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.divisions import DivisionParams, insert_divisions, _pairs_within  # noqa: E402

ISO = (1.0, 1.0, 1.0)   # µm/voxel — keeps every distance in the tests readable as-is
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def chain(n_tracks: int, T: int, spacing: float = 40.0, drift: float = 0.5):
    """`n_tracks` straight, well-separated tracks over `T` frames.

    Spacing is far outside any radius under test, so a bare chain must yield no
    insertions at all — anything found in it is a false positive by construction.
    """
    t, zyx, edges = [], [], []
    for k in range(n_tracks):
        prev = None
        for f in range(T):
            i = len(t)
            t.append(f)
            zyx.append([0.0, k * spacing, f * drift])
            if prev is not None:
                edges.append([prev, i])
            prev = i
    return (np.array(t, np.int64), np.array(zyx, float),
            np.array(edges, np.int64).reshape(-1, 2))


def add_free_node(t, zyx, edges, frame, pos):
    """Append an unlinked detection (in-degree 0, out-degree 0) at `frame`."""
    t = np.append(t, np.int64(frame))
    zyx = np.vstack([zyx, np.asarray(pos, float)[None, :]])
    return t, zyx, edges, len(t) - 1


def main() -> int:
    print("=" * 60)
    print("insert_divisions")
    print("=" * 60)

    # -- the control -------------------------------------------------------------
    t, zyx, e = chain(6, 5)
    out = insert_divisions(t, zyx, e, scale=ISO, frame_frac_cap=0.0)
    check("cap=0 is an exact no-op", out.shape == (0, 2),
          f"returned {out.shape}, the sweep's control must add nothing")

    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    check("well-separated tracks yield nothing even at cap=1.0", len(out) == 0,
          f"{len(out)} edge(s) proposed at 40 µm spacing with a 4.5 µm radius")

    # -- it finds a real one -----------------------------------------------------
    # Track 0 continues to (0, 0, 1.0) at frame 1. Put a free node 2 µm from the parent
    # on the other side, so parent is close to the daughters' midpoint.
    t, zyx, e = chain(3, 3)
    t, zyx, e, free = add_free_node(t, zyx, e, 1, [0.0, 2.0, 0.5])
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    hit = [(int(a), int(b)) for a, b in out]
    check("a plausible sister is proposed", (0, free) in hit,
          f"proposed {hit}, wanted (0, {free})")

    # -- the two hard constraints ------------------------------------------------
    t, zyx, e = chain(3, 3)
    # A node at frame 1 that is close by but ALREADY has a parent: never proposable.
    claimed = int(np.flatnonzero((t == 1))[1])
    zyx[claimed] = [0.0, 2.0, 0.5]
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    check("a sister that already has a parent is never used",
          all(int(b) != claimed for _, b in out),
          f"proposed {[(int(a), int(b)) for a, b in out]}; node {claimed} has in-degree 1 "
          "and would make the fork `malformed` -> automatic FP")

    # Two free nodes near the same parent: it may fork once, never twice.
    t, zyx, e = chain(3, 3)
    t, zyx, e, f1 = add_free_node(t, zyx, e, 1, [0.0, 2.0, 0.5])
    t, zyx, e, f2 = add_free_node(t, zyx, e, 1, [0.0, -2.0, 0.5])
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    srcs = [int(a) for a, _ in out]
    check("a parent is never given two extra children",
          len(srcs) == len(set(srcs)),
          f"sources {srcs} — a second insertion would make a 3-fork")

    # Two parents competing for one free node, both genuinely eligible for it: exactly
    # one must win. Two 2-frame tracks 4 µm apart, with the free node midway.
    t, zyx, e = chain(2, 2, spacing=4.0, drift=1.0)
    t, zyx, e, contested = add_free_node(t, zyx, e, 1, [0.0, 2.0, 1.0])
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=6.0, sister_max_um=8.0,
                           frame_frac_cap=1.0)
    tgts = [int(b) for _, b in out]
    check("a sister is claimed by at most one parent",
          tgts.count(contested) == 1 and len(tgts) == len(set(tgts)),
          f"targets {tgts}: both parents are eligible for node {contested}, "
          "a repeat would be a merge and neither would be a proposal at all")

    # -- every proposal spans exactly one frame ----------------------------------
    rng = np.random.default_rng(0)
    t, zyx, e = chain(30, 6, spacing=3.0)
    for f in range(6):
        t, zyx, e, _ = add_free_node(t, zyx, e, f, rng.normal(0, 2.0, 3))
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    dt = t[out[:, 1]] - t[out[:, 0]] if len(out) else np.array([], np.int64)
    check("every proposed edge spans exactly t -> t+1",
          len(out) > 0 and bool((dt == 1).all()),
          f"{len(out)} proposed, dt values {sorted(set(dt.tolist()))}")

    # -- the cap binds, per frame ------------------------------------------------
    t, zyx, e = chain(40, 5, spacing=2.5)
    for f in range(5):
        for _ in range(20):
            t, zyx, e, _ = add_free_node(t, zyx, e, f, rng.normal(0, 6.0, 3))
    counts = {}
    for cap in (0.002, 0.02, 0.2):
        out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                               frame_frac_cap=cap)
        counts[cap] = len(out)
        if len(out):
            per_frame = np.bincount(t[out[:, 0]])
            allowed = np.bincount(t)
            over = [(int(f), int(c)) for f, c in enumerate(per_frame)
                    if c > int(cap * allowed[f])]
            check(f"cap={cap} is respected in every frame", not over,
                  f"frames over budget: {over}" if over else f"{len(out)} inserted")
    check("more cap yields at least as many insertions",
          counts[0.002] <= counts[0.02] <= counts[0.2],
          f"counts {counts} — the sweep axis has to be monotone to be readable")

    # -- radii actually gate -----------------------------------------------------
    t, zyx, e = chain(3, 3)
    t, zyx, e, far = add_free_node(t, zyx, e, 1, [0.0, 5.5, 0.5])
    tight = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                             frame_frac_cap=1.0)
    loose = insert_divisions(t, zyx, e, scale=ISO, max_um=6.0, sister_max_um=6.8,
                             frame_frac_cap=1.0)
    check("max_um gates the parent->sister distance",
          len(tight) == 0 and any(int(b) == far for _, b in loose),
          f"5.5 µm away: {len(tight)} at max_um=4.5, {len(loose)} at 6.0")

    t, zyx, e = chain(3, 3)
    # 4 µm from the parent but 7.9 µm from the existing daughter: passes gate 1, fails 2.
    t, zyx, e, odd = add_free_node(t, zyx, e, 1, [0.0, -4.0, 8.0])
    zyx[0] = [0.0, 0.0, 4.5]
    narrow = insert_divisions(t, zyx, e, scale=ISO, max_um=6.0, sister_max_um=4.0,
                              frame_frac_cap=1.0)
    wide = insert_divisions(t, zyx, e, scale=ISO, max_um=6.0, sister_max_um=12.0,
                            frame_frac_cap=1.0)
    check("sister_max_um gates the daughter->daughter distance",
          all(int(b) != odd for _, b in narrow) and any(int(b) == odd for _, b in wide),
          f"{len(narrow)} at sister_max_um=4.0, {len(wide)} at 12.0")

    # -- ranking prefers the parent-between-daughters geometry -------------------
    # The existing daughter has to be displaced off the parent's axis or there is no
    # "side" to be opposite to and the test means nothing (it caught exactly that).
    t, zyx, e = chain(2, 3, spacing=60.0)
    zyx[0] = [0.0, 0.0, 0.0]        # parent
    zyx[1] = [0.0, 1.5, 1.0]        # its existing daughter, displaced +1.5 in y
    # Opposite the daughter: the parent lands almost exactly on the daughters' midpoint,
    # which is what a real division looks like. Beyond the daughter: a tail-follower, the
    # same shape as an ordinary tracking error. Both pass both radius gates.
    t, zyx, e, opposite = add_free_node(t, zyx, e, 1, [0.0, -1.5, 1.0])
    t, zyx, e, beyond = add_free_node(t, zyx, e, 1, [0.0, 3.0, 1.0])
    out = insert_divisions(t, zyx, e, scale=ISO, max_um=4.5, sister_max_um=6.8,
                           frame_frac_cap=1.0)
    first = [int(b) for a, b in out if int(a) == 0]
    check("the parent-between-daughters candidate is ranked first",
          first[:1] == [opposite],
          f"chose {first[:1]}, wanted [{opposite}] (midpoint on the parent) over "
          f"[{beyond}] (tail-follower)")

    # -- determinism -------------------------------------------------------------
    t, zyx, e = chain(30, 6, spacing=3.0)
    for f in range(6):
        t, zyx, e, _ = add_free_node(t, zyx, e, f, rng.normal(0, 2.0, 3))
    a = insert_divisions(t, zyx, e, scale=ISO, frame_frac_cap=0.5)
    b = insert_divisions(t, zyx, e, scale=ISO, frame_frac_cap=0.5)
    check("the same input gives the same output", np.array_equal(a, b),
          f"{len(a)} vs {len(b)} edges")

    # -- scale is applied, not ignored -------------------------------------------
    # Same voxel geometry, anisotropic scale: the µm distance grows 4x on y, so a
    # candidate inside the radius in voxels falls outside it in µm.
    t, zyx, e = chain(3, 3, spacing=40.0)
    t, zyx, e, near = add_free_node(t, zyx, e, 1, [0.0, 3.0, 0.5])
    iso = insert_divisions(t, zyx, e, scale=(1.0, 1.0, 1.0), max_um=4.5,
                           sister_max_um=6.8, frame_frac_cap=1.0)
    aniso = insert_divisions(t, zyx, e, scale=(1.0, 4.0, 1.0), max_um=4.5,
                             sister_max_um=6.8, frame_frac_cap=1.0)
    check("scale converts voxels to µm before the radii are applied",
          len(iso) == 1 and len(aniso) == 0,
          f"{len(iso)} at 1 µm/px vs {len(aniso)} at 4 µm/px on y")

    # -- degenerate inputs -------------------------------------------------------
    empty = np.zeros((0, 2), np.int64)
    check("an empty graph is handled",
          insert_divisions(np.zeros(0, np.int64), np.zeros((0, 3)), empty,
                           scale=ISO, frame_frac_cap=1.0).shape == (0, 2))
    t, zyx, _ = chain(4, 3)
    check("a graph with no edges is handled",
          insert_divisions(t, zyx, empty, scale=ISO, frame_frac_cap=1.0).shape == (0, 2),
          "no out-degree-1 nodes means nothing to fork")

    # -- DivisionParams is a faithful carrier for a sweep cell -------------------
    p = DivisionParams()
    check("DivisionParams defaults are the public notebook's published constants",
          (p.max_um, p.sister_max_um, p.frame_frac_cap) == (4.5, 6.8, 0.008),
          f"{p!r}")
    t, zyx, e = chain(30, 6, spacing=3.0)
    for f in range(6):
        t, zyx, e, _ = add_free_node(t, zyx, e, f, rng.normal(0, 2.0, 3))
    p = DivisionParams(max_um=6.0, sister_max_um=8.0, frame_frac_cap=0.25)
    check("passing DivisionParams through matches passing the fields directly",
          np.array_equal(
              insert_divisions(t, zyx, e, scale=ISO, max_um=p.max_um,
                               sister_max_um=p.sister_max_um,
                               frame_frac_cap=p.frame_frac_cap),
              insert_divisions(t, zyx, e, scale=ISO, max_um=6.0, sister_max_um=8.0,
                               frame_frac_cap=0.25)))

    # -- both neighbour paths agree ----------------------------------------------
    a_pts = rng.normal(0, 5, (40, 3))
    b_pts = rng.normal(0, 5, (35, 3))
    kd = _pairs_within(a_pts, b_pts, 3.0)
    d = np.linalg.norm(a_pts[:, None, :] - b_pts[None, :, :], axis=2)
    i, j = np.nonzero(d <= 3.0)
    brute = set(zip(i.tolist(), j.tolist()))
    check("the KD-tree and brute-force neighbour paths agree",
          set(map(tuple, kd.tolist())) == brute,
          f"{len(kd)} vs {len(brute)} pairs")

    print()
    print("=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all division-insertion tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
