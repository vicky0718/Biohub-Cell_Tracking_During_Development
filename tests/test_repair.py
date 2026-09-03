"""Tests for `pipeline/repair.py`.

Two of these repairs change the node set, and getting an index remap wrong is silent —
the graph stays well-formed and simply describes different cells. Those are the checks
that matter most here. The rest guard invariants the scorer punishes: an edge that does
not span exactly `t -> t+1` is dropped, a second parent is a merge, and a node pushed
outside its own 7 µm match radius is a detection thrown away by a repair meant to help.

Pure numpy; scipy only if present.

    python tests/test_repair.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.repair import _components  # noqa: E402
from pipeline.repair import (  # noqa: E402
    cap_edge_length, close_gaps, linefit_smooth, prune_isolated, rank_budget_prune,
    single_parent_repair,
)

ISO = (1.0, 1.0, 1.0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def track(start, T, vel=(0.0, 1.0, 0.0), t0=0):
    """One straight track: returns (t, zyx) with T frames from t0."""
    start = np.asarray(start, float)
    vel = np.asarray(vel, float)
    t = np.arange(t0, t0 + T, dtype=np.int64)
    zyx = start[None, :] + vel[None, :] * np.arange(T)[:, None]
    return t, zyx


def build(tracks):
    """Concatenate `[(t, zyx), ...]` into one graph, chaining each track's own nodes."""
    ts, zs, es, off = [], [], [], 0
    for t, zyx in tracks:
        ts.append(t)
        zs.append(zyx)
        es += [(off + i, off + i + 1) for i in range(len(t) - 1)]
        off += len(t)
    return (np.concatenate(ts), np.vstack(zs),
            np.asarray(es, np.int64).reshape(-1, 2))


def edge_set(t, zyx, edges):
    """Edges as coordinate pairs, so they survive an index remap and can be compared."""
    return {(tuple(np.round(zyx[a], 6)), tuple(np.round(zyx[b], 6))) for a, b in edges}



def rank_budget_section() -> None:
    """`rank_budget_prune` — notes/51's third selection rule, cutting AFTER linking."""
    print("=" * 62)
    print("rank_budget_prune")
    print("=" * 62)

    # A long tight track and a short erratic one, disjoint.
    t = np.array([0, 1, 2, 3, 4, 5, 0, 1], np.int64)
    zyx = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 0, 3], [0, 0, 4], [0, 0, 5],
                    [0, 50, 0], [0, 90, 0]], float)
    e = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [6, 7]], np.int64)

    nt, nz, ne = rank_budget_prune(t, zyx, e, n_target=6, mode="geometry", scale=ISO)
    check("the long tight track survives the budget", len(nt) == 6, f"{len(nt)} nodes")
    check("the erratic track is the one dropped", nz[:, 1].max() == 0)
    check("edges are remapped with the nodes", len(ne) == 5 and ne.max() < len(nt),
          f"{len(ne)} edges, max index {ne.max() if len(ne) else -1}")

    r = rank_budget_prune(t, zyx, e, n_target=100, mode="geometry", scale=ISO)
    check("already under budget is an exact no-op",
          len(r[0]) == len(t) and len(r[2]) == len(e))

    # isolated mode: an edgeless node is pure budget cost and cannot ever be a TP edge.
    it = np.array([0, 1, 0], np.int64)
    iz = np.array([[0, 0, 0], [0, 0, 1], [0, 9, 9]], float)
    ie = np.array([[0, 1]], np.int64)
    r = rank_budget_prune(it, iz, ie, n_target=float("nan"), mode="isolated", scale=ISO)
    check("isolated mode drops only the edgeless node", len(r[0]) == 2, f"{len(r[0])} left")
    check("isolated mode keeps the real edge intact",
          len(r[2]) == 1 and r[2][0].tolist() == [0, 1])

    # A fork must outrank a longer plain track when the budget cannot hold both.
    ft = np.array([0, 1, 1, 0, 1, 2, 3], np.int64)
    fz = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0],
                   [0, 20, 0], [0, 20, 1], [0, 20, 2], [0, 20, 3]], float)
    fe = np.array([[0, 1], [0, 2], [3, 4], [4, 5], [5, 6]], np.int64)
    r = rank_budget_prune(ft, fz, fe, n_target=3, mode="geometry",
                          keep_division_components=True, scale=ISO)
    check("a fork component outranks a longer plain track",
          any(np.allclose(q, [0, 0, 0]) for q in r[1]))

    # An empty prediction scores edge_J 0; an over-budget one only shrinks the multiplier
    # (1 - 0.1*ratio, floored at 0). So when even the best single track busts the budget it
    # is still kept -- the invariant is "under target, OR exactly one track survives".
    for target, n_tracks in ((2, 1), (6, 1), (7, 1), (8, 2)):
        r = rank_budget_prune(t, zyx, e, n_target=target, mode="geometry", scale=ISO)
        comps = len(np.unique(_components(r[0], r[2], len(r[0])))) if len(r[0]) else 0
        check(f"target {target}: under budget or a single kept track",
              len(r[0]) <= target or comps == 1, f"{len(r[0])} nodes in {comps} track(s)")
        check(f"target {target} keeps {n_tracks} track(s)", comps == n_tracks, f"got {comps}")

    # Equal spans: geometry must break the tie on tightness, length must not see it.
    qt = np.array([0, 1, 2, 0, 1, 2], np.int64)
    qz = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 2],
                   [0, 50, 0], [0, 90, 0], [0, 50, 0]], float)
    qe = np.array([[0, 1], [1, 2], [3, 4], [4, 5]], np.int64)
    g = rank_budget_prune(qt, qz, qe, n_target=3, mode="geometry", scale=ISO)
    check("geometry breaks an equal-span tie on tightness", g[1][:, 1].max() == 0)

    r = rank_budget_prune(np.zeros(0, np.int64), np.zeros((0, 3)),
                          np.zeros((0, 2), np.int64), n_target=5, scale=ISO)
    check("an empty graph is handled", len(r[0]) == 0)
    print()


def main() -> int:
    rng = np.random.default_rng(0)

    print("=" * 62)
    print("prune_isolated")
    print("=" * 62)
    t, zyx, e = build([track([0, 0, 0], 4), track([0, 50, 0], 4)])
    before = edge_set(t, zyx, e)
    # Three nodes touched by nothing, interleaved so a naive remap would shift indices.
    t2 = np.concatenate([t, [1, 2, 3]])
    zyx2 = np.vstack([zyx, rng.normal(0, 100, (3, 3))])
    nt, nz, ne = prune_isolated(t2, zyx2, e)
    check("isolated nodes are dropped", len(nt) == len(t),
          f"{len(t2)} -> {len(nt)}, expected {len(t)}")
    check("no edge is lost", len(ne) == len(e), f"{len(e)} -> {len(ne)}")
    check("the remap preserves which cells each edge joins",
          edge_set(nt, nz, ne) == before,
          "an index remap that silently rewires edges is the failure mode here")

    # The remap must survive isolated nodes appearing BEFORE used ones.
    t3 = np.concatenate([[5], t])
    zyx3 = np.vstack([[[9, 9, 9]], zyx])
    nt, nz, ne = prune_isolated(t3, zyx3, e + 1)
    check("the remap is correct when the isolated node comes first",
          edge_set(nt, nz, ne) == before)
    check("an all-connected graph is returned unchanged",
          len(prune_isolated(t, zyx, e)[0]) == len(t))
    check("an empty graph is handled",
          len(prune_isolated(np.zeros(0, np.int64), np.zeros((0, 3)),
                             np.zeros((0, 2), np.int64))[0]) == 0)

    print()
    print("=" * 62)
    print("cap_edge_length")
    print("=" * 62)
    t, zyx, e = build([track([0, 0, 0], 3, vel=(0, 2.0, 0))])
    t = np.append(t, np.int64(1))
    zyx = np.vstack([zyx, [[0.0, 40.0, 0.0]]])
    e = np.vstack([e, [[0, 3]]])                 # a 40 µm link
    _, _, ne = cap_edge_length(t, zyx, e, scale=ISO, max_um=14.0)
    check("an over-length edge is dropped", len(ne) == len(e) - 1,
          f"{len(e)} -> {len(ne)}")
    check("in-range edges survive untouched",
          all(tuple(r) != (0, 3) for r in ne.tolist()))
    _, _, ne = cap_edge_length(t, zyx, e, scale=ISO, max_um=100.0)
    check("a generous cap drops nothing", len(ne) == len(e))

    print()
    print("=" * 62)
    print("single_parent_repair")
    print("=" * 62)
    # Two parents at t=0 both pointing at one node at t=1; the nearer one must win.
    t = np.array([0, 0, 1], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 1.0, 0.0]])
    e = np.array([[1, 2], [0, 2]], np.int64)     # far one listed FIRST
    _, _, ne = single_parent_repair(t, zyx, e, scale=ISO)
    check("only one incoming edge survives", len(ne) == 1, f"{ne.tolist()}")
    check("the NEAREST parent is kept, not the lowest row id",
          ne.tolist() == [[0, 2]],
          f"got {ne.tolist()}; row 0 is the 5 µm-further parent, the scorer would keep it")
    t, zyx, e = build([track([0, 0, 0], 4), track([0, 50, 0], 4)])
    check("a clean graph is untouched",
          np.array_equal(single_parent_repair(t, zyx, e, scale=ISO)[2], e))

    print()
    print("=" * 62)
    print("linefit_smooth")
    print("=" * 62)
    # A perfectly straight track with one node knocked sideways: smoothing must pull it
    # back, and must not move the nodes that were already on the line very far.
    t, zyx, e = build([track([0, 0, 0], 7, vel=(0.0, 1.0, 0.0))])
    truth = zyx.copy()
    zyx = zyx.copy()
    zyx[3] += np.array([0.0, 0.0, 2.5])
    _, sm, ne = linefit_smooth(t, zyx, e, window=2, weight=0.76, scale=ISO)
    before_err = np.linalg.norm(zyx[3] - truth[3])
    after_err = np.linalg.norm(sm[3] - truth[3])
    check("a knocked-out node is pulled back toward the line",
          after_err < before_err,
          f"{before_err:.3f} µm -> {after_err:.3f} µm off truth")
    check("edges are untouched", np.array_equal(ne, e))
    check("node count is unchanged", len(sm) == len(zyx))

    # A fork: the parent must NOT be dragged toward either daughter.
    t = np.array([0, 1, 2, 2], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.0, 2.0, 3.0], [0.0, 2.0, -3.0]])
    e = np.array([[0, 1], [1, 2], [1, 3]], np.int64)
    _, sm, _ = linefit_smooth(t, zyx, e, window=2, weight=0.76, scale=ISO)
    check("a line is not fitted across a division",
          np.allclose(sm[1], zyx[1]),
          f"parent moved {np.linalg.norm(sm[1]-zyx[1]):.4f} µm; fitting through a fork "
          "drags it toward one daughter")

    # A merge: two parents into one child is not a chain either.
    t = np.array([0, 0, 1, 2], np.int64)
    zyx = np.array([[0.0, 0.0, 0.0], [0.0, 6.0, 0.0],
                    [0.0, 3.0, 0.0], [0.0, 3.0, 1.0]])
    e = np.array([[0, 2], [1, 2], [2, 3]], np.int64)
    _, sm, _ = linefit_smooth(t, zyx, e, window=2, weight=0.76, scale=ISO)
    check("a line is not fitted across a merge",
          np.allclose(sm[2], zyx[2]),
          f"child moved {np.linalg.norm(sm[2]-zyx[2]):.4f} µm")

    # The shift bound must actually bind.
    t, zyx, e = build([track([0, 0, 0], 7, vel=(0.0, 1.0, 0.0))])
    zyx = zyx.copy()
    zyx[3] += np.array([0.0, 0.0, 60.0])
    _, sm, _ = linefit_smooth(t, zyx, e, window=2, weight=1.0, scale=ISO,
                              max_shift_um=3.2)
    moved = np.linalg.norm(sm - zyx, axis=1)
    check("no node moves further than max_shift_um", moved.max() <= 3.2 + 1e-9,
          f"max move {moved.max():.4f} µm against a 3.2 µm bound")

    check("weight=0 is an exact no-op",
          np.array_equal(linefit_smooth(t, zyx, e, weight=0.0, scale=ISO)[1], zyx))

    # Scale must be applied: the same voxel move is a different µm move on an
    # anisotropic grid, so the bound has to bite differently.
    t, zyx, e = build([track([0, 0, 0], 7, vel=(0.0, 1.0, 0.0))])
    zyx = zyx.copy()
    zyx[3] += np.array([0.0, 0.0, 8.0])
    _, a, _ = linefit_smooth(t, zyx, e, weight=1.0, scale=(1.0, 1.0, 1.0),
                             max_shift_um=3.2)
    _, b, _ = linefit_smooth(t, zyx, e, weight=1.0, scale=(1.0, 1.0, 8.0),
                             max_shift_um=3.2)
    check("scale is applied before the shift bound",
          not np.allclose(a[3], b[3]),
          "identical voxel geometry must clamp differently at 1 vs 8 µm/px on x")

    print()
    print("=" * 62)
    print("close_gaps")
    print("=" * 62)
    # A track that stops at t=1 and one that starts at t=3, 2 µm apart: one hole.
    a_t, a_z = track([0, 0, 0], 2, vel=(0.0, 1.0, 0.0), t0=0)
    b_t, b_z = track([0, 3.0, 0], 2, vel=(0.0, 1.0, 0.0), t0=3)
    t, zyx, e = build([(a_t, a_z), (b_t, b_z)])
    n0 = len(t)
    nt, nz, ne = close_gaps(t, zyx, e, scale=ISO, max_um=5.75, max_added_frac=1.0)
    check("a one-frame hole is bridged", len(nt) == n0 + 1, f"{n0} -> {len(nt)}")
    check("two edges are recovered for one node", len(ne) == len(e) + 2,
          f"{len(e)} -> {len(ne)}")
    if len(nt) == n0 + 1:
        check("the inserted node sits at the missing frame", int(nt[-1]) == 2,
              f"t={int(nt[-1])}, wanted 2")
    dt = nt[ne[:, 1]] - nt[ne[:, 0]]
    check("every edge still spans exactly t -> t+1", bool((dt == 1).all()),
          f"spans present: {sorted(set(dt.tolist()))}")
    ind = np.bincount(ne[:, 1], minlength=len(nt))
    check("no node gains a second parent", int(ind.max(initial=0)) <= 1,
          f"max in-degree {int(ind.max(initial=0))}")

    # Too far apart: nothing should be bridged.
    b_t, b_z = track([0, 40.0, 0], 2, vel=(0.0, 1.0, 0.0), t0=3)
    t, zyx, e = build([(a_t, a_z), (b_t, b_z)])
    check("a hole wider than max_um is left alone",
          len(close_gaps(t, zyx, e, scale=ISO, max_um=5.75, max_added_frac=1.0)[0]) == len(t))

    # Budget must bind, and two tails must not both claim one head.
    tracks = []
    for k in range(12):
        tracks.append(track([0, k * 10.0, 0], 2, t0=0))
        tracks.append(track([0.5, k * 10.0, 0], 2, t0=3))
    t, zyx, e = build(tracks)
    n0 = len(t)
    nt, _, ne = close_gaps(t, zyx, e, scale=ISO, max_um=5.75,
                           max_added_frac=1.0, max_added_abs=3)
    check("max_added_abs binds", len(nt) - n0 <= 3, f"added {len(nt)-n0}")
    ind = np.bincount(ne[:, 1], minlength=len(nt))
    check("greedy assignment never double-claims a head",
          int(ind.max(initial=0)) <= 1, f"max in-degree {int(ind.max(initial=0))}")
    check("a zero budget is an exact no-op",
          len(close_gaps(t, zyx, e, scale=ISO, max_added_frac=0.0,
                         max_added_abs=0)[0]) == n0)

    print()
    print("=" * 62)
    print("composition — the order the ablation applies them in")
    print("=" * 62)
    t, zyx, e = build([track([0, 0, 0], 6), track([0, 40, 0], 6)])
    t = np.concatenate([t, [2, 4]])
    zyx = np.vstack([zyx, rng.normal(0, 80, (2, 3))])
    g = (t, zyx, e)
    for fn in (lambda g: cap_edge_length(*g, scale=ISO),
               lambda g: single_parent_repair(*g, scale=ISO),
               lambda g: close_gaps(*g, scale=ISO),
               lambda g: linefit_smooth(*g, scale=ISO),
               lambda g: prune_isolated(*g)):
        g = fn(g)
    ct, cz, ce = g
    dt = ct[ce[:, 1]] - ct[ce[:, 0]]
    check("the full chain keeps every edge spanning t -> t+1",
          len(ce) > 0 and bool((dt == 1).all()),
          f"{len(ce)} edges, spans {sorted(set(dt.tolist()))}")
    check("the full chain leaves no node with two parents",
          int(np.bincount(ce[:, 1], minlength=len(ct)).max(initial=0)) <= 1)
    check("the full chain leaves no isolated node",
          len(ct) == len(np.unique(ce)), f"{len(ct)} nodes, {len(np.unique(ce))} used")
    check("every edge index is in range",
          ce.size == 0 or (ce.min() >= 0 and ce.max() < len(ct)))

    print()
    rank_budget_section()

    print("=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all repair tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
