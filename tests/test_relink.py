"""Tests for `pipeline/relink.py`.

Three invariants matter more than the rest, and each has a reason with a number behind it:

* **Forks survive.** The pack emits ~54 divisions per 24 datasets and the division term is
  worth 0.1 of the 1.1 maximum. A solver that re-derives whole frames collapses every fork
  to one child — trading a term we score 0.000 on for one we would score worse on.
* **The change budget binds.** `notes/27` §1 measured that node-rewiring repairs buy
  mislinks and pay in detection failures, ~4:1 where they work and inverting when pushed.
  `max_change_frac` is what stops a bad cost function from destroying a working graph.
* **The graph stays well-formed.** Edges spanning anything but `t → t+1` are dropped by the
  scorer, and a second parent is a merge; either would be silent.

    python tests/test_relink.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.relink import RelinkParams, motion_relink  # noqa: E402

ISO = (1.0, 1.0, 1.0)
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


class G:
    """Explicit graph builder — every test states its own geometry."""

    def __init__(self):
        self.t, self.zyx, self.e = [], [], []

    def n(self, frame, y, x=0.0, z=0.0):
        self.t.append(int(frame))
        self.zyx.append([z, float(y), float(x)])
        return len(self.t) - 1

    def link(self, a, b):
        self.e.append((a, b))
        return self

    def out(self):
        return (np.asarray(self.t, np.int64), np.asarray(self.zyx, float),
                np.asarray(self.e, np.int64).reshape(-1, 2))


def main() -> int:
    print("=" * 66)
    print("motion_relink")
    print("=" * 66)

    # -- it fixes a crossed assignment ------------------------------------------
    # Two cells moving straight down their own lanes, but the graph has them CROSSED.
    # Velocity continuity should uncross them; plain distance alone would not, because
    # the crossed pairing is no further in raw distance.
    g = G()
    a0, b0 = g.n(0, 0.0), g.n(0, 10.0)
    a1, b1 = g.n(1, 2.0), g.n(1, 12.0)
    a2, b2 = g.n(2, 4.0), g.n(2, 14.0)
    g.link(a0, a1).link(b0, b1)
    g.link(a1, b2).link(b1, a2)          # <- crossed at t=1
    t, zyx, e = g.out()
    out = motion_relink(t, zyx, e, scale=ISO, tight_um=6.0, relaxed_um=12.0,
                        velocity_weight=0.9, max_change_frac=1.0)
    got = set(map(tuple, out.tolist()))
    check("a crossed assignment is uncrossed by velocity continuity",
          (a1, a2) in got and (b1, b2) in got,
          f"got {sorted(got)}; wanted ({a1},{a2}) and ({b1},{b2})")

    # -- forks are protected ----------------------------------------------------
    g = G()
    p = g.n(0, 0.0)
    c1, c2 = g.n(1, -2.0), g.n(1, 2.0)
    g.link(p, c1).link(p, c2)
    d1, d2 = g.n(2, -3.0), g.n(2, 3.0)
    g.link(c1, d1).link(c2, d2)
    t, zyx, e = g.out()
    out = motion_relink(t, zyx, e, scale=ISO, tight_um=6.0, relaxed_um=12.0,
                        max_change_frac=1.0)
    got = set(map(tuple, out.tolist()))
    check("both branches of a fork survive", (p, c1) in got and (p, c2) in got,
          f"got {sorted(got)}; a solver that re-derives frames collapses this to one child")
    outd = np.bincount(out[:, 0], minlength=len(t))
    check("the fork is still a fork", int(outd[p]) == 2, f"out-degree {int(outd[p])}")

    # -- well-formedness --------------------------------------------------------
    rng = np.random.default_rng(0)
    g = G()
    prev = {}
    for f in range(8):
        for k in range(25):
            i = g.n(f, k * 4.0 + rng.normal(0, 0.6), rng.normal(0, 0.6))
            if k in prev:
                g.link(prev[k], i)
            prev[k] = i
    t, zyx, e = g.out()
    out = motion_relink(t, zyx, e, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                        max_change_frac=1.0)
    dt = t[out[:, 1]] - t[out[:, 0]]
    check("every relinked edge spans exactly t -> t+1", bool((dt == 1).all()),
          f"spans {sorted(set(dt.tolist()))}")
    ind = np.bincount(out[:, 1], minlength=len(t))
    check("no node gains a second parent", int(ind.max(initial=0)) <= 1,
          f"max in-degree {int(ind.max(initial=0))}")
    outd = np.bincount(out[:, 0], minlength=len(t))
    check("no non-fork source gains a second child", int(outd.max(initial=0)) <= 1,
          f"max out-degree {int(outd.max(initial=0))}")
    check("indices stay in range", out.min() >= 0 and out.max() < len(t))
    check("the node set is untouched", len(t) == len(zyx))

    # -- the change budget binds ------------------------------------------------
    # A deliberately terrible cost function (bonus on a random probability table) would
    # rewrite everything; max_change_frac must stop it.
    fake = np.array([[i, j, rng.random()] for i in range(len(t)) for j in range(len(t))
                     if t[j] == t[i] + 1 and abs(zyx[i][1] - zyx[j][1]) < 9.0],
                    dtype=float)
    base = set(map(tuple, e.tolist()))
    for frac in (0.0, 0.1, 1.0):
        o = motion_relink(t, zyx, e, cand=fake, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                          learned_bonus=50.0, max_change_frac=frac)
        n_changed = len(set(map(tuple, o.tolist())) - base)
        if frac == 0.0:
            check("max_change_frac=0 is an exact no-op", np.array_equal(o, e),
                  f"{n_changed} edges changed")
        else:
            cap = int(frac * len(e)) + len(np.unique(t)) + 1   # per-frame budget, summed
            check(f"max_change_frac={frac} bounds the rewrite", n_changed <= cap,
                  f"{n_changed} changed against a ~{cap} bound")

    # -- the learned probability is actually used -------------------------------
    # Two equally-plausible targets; the probability table breaks the tie. If the bonus
    # were ignored, the pick could not follow the table when the table is flipped.
    picks = []
    for favour in (0, 1):
        g = G()
        s = g.n(0, 0.0)
        x1, x2 = g.n(1, 3.0), g.n(1, -3.0)
        g.link(s, x1)
        tt, zz, ee = g.out()
        cand = np.array([[s, x1, 1.0 - favour], [s, x2, float(favour)]], float)
        o = motion_relink(tt, zz, ee, cand=cand, scale=ISO, tight_um=6.0, relaxed_um=9.0,
                          velocity_weight=0.0, learned_bonus=5.0, max_change_frac=1.0)
        picks.append(int(o[0, 1]))
    check("the learned probability decides a geometric tie",
          picks[0] != picks[1], f"picked {picks} for the two probability tables")

    # -- no candidate table is a supported mode ---------------------------------
    o = motion_relink(t, zyx, e, cand=None, scale=ISO, max_change_frac=1.0)
    check("running without probabilities works (geometry + velocity only)",
          len(o) > 0 and bool((t[o[:, 1]] - t[o[:, 0]] == 1).all()),
          f"{len(o)} edges — this is the arm that makes the learned term measurable")

    # -- the greedy fallback agrees on structure --------------------------------
    hung = motion_relink(t, zyx, e, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                         max_change_frac=1.0, max_frame_nodes=10_000)
    gred = motion_relink(t, zyx, e, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                         max_change_frac=1.0, max_frame_nodes=1)
    for lbl, o in (("hungarian", hung), ("greedy", gred)):
        ind = np.bincount(o[:, 1], minlength=len(t))
        check(f"{lbl} produces a well-formed graph",
              int(ind.max(initial=0)) <= 1 and bool((t[o[:, 1]] - t[o[:, 0]] == 1).all()))
    check("the two solvers agree on most edges",
          len(set(map(tuple, hung.tolist())) & set(map(tuple, gred.tolist())))
          >= 0.8 * len(hung),
          f"{len(set(map(tuple, hung.tolist())) & set(map(tuple, gred.tolist())))}/{len(hung)} shared")

    # -- radii gate -------------------------------------------------------------
    g = G()
    s = g.n(0, 0.0)
    far = g.n(1, 20.0)
    g.link(s, far)
    tt, zz, ee = g.out()
    o = motion_relink(tt, zz, ee, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                      max_change_frac=1.0)
    check("a pair beyond relaxed_um leaves the existing edge alone",
          np.array_equal(o, ee),
          "nothing is assignable, so the input must come back unchanged")

    # -- scale is applied -------------------------------------------------------
    g = G()
    s = g.n(0, 0.0)
    a, b = g.n(1, 3.0), g.n(1, -3.0)
    g.link(s, a)
    tt, zz, ee = g.out()
    iso = motion_relink(tt, zz, ee, scale=(1.0, 1.0, 1.0), tight_um=4.0, relaxed_um=5.0,
                        max_change_frac=1.0)
    aniso = motion_relink(tt, zz, ee, scale=(1.0, 4.0, 1.0), tight_um=4.0, relaxed_um=5.0,
                          max_change_frac=1.0)
    check("scale converts voxels to µm before the radii apply",
          len(iso) == 1 and np.array_equal(aniso, ee),
          f"iso {len(iso)} edge(s); at 4 µm/px on y everything is out of range")

    # -- degenerate inputs ------------------------------------------------------
    empty = np.zeros((0, 2), np.int64)
    check("an empty graph is handled",
          motion_relink(np.zeros(0, np.int64), np.zeros((0, 3)), empty,
                        scale=ISO).shape == (0, 2))
    check("a graph with no edges is handled",
          motion_relink(t, zyx, empty, scale=ISO).shape == (0, 2))

    # -- params carrier ---------------------------------------------------------
    p = RelinkParams()
    check("RelinkParams defaults are the public notebook's constants",
          (p.tight_um, p.relaxed_um, p.velocity_weight, p.learned_bonus,
           p.max_frame_nodes) == (5.9, 9.8, 0.50, 0.78, 3200), f"{p!r}")
    check("passing params through matches passing fields directly",
          np.array_equal(
              motion_relink(t, zyx, e, scale=ISO, tight_um=p.tight_um,
                            relaxed_um=p.relaxed_um, velocity_weight=p.velocity_weight,
                            learned_bonus=p.learned_bonus, max_change_frac=0.2),
              motion_relink(t, zyx, e, scale=ISO, tight_um=5.9, relaxed_um=9.8,
                            velocity_weight=0.5, learned_bonus=0.78,
                            max_change_frac=0.2)))

    # -- determinism ------------------------------------------------------------
    check("the same input gives the same output",
          np.array_equal(motion_relink(t, zyx, e, scale=ISO, max_change_frac=0.3),
                         motion_relink(t, zyx, e, scale=ISO, max_change_frac=0.3)))

    print()
    print("=" * 66)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all relink tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
