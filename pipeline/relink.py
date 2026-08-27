"""Motion relink: re-solve frame-to-frame assignment using velocity and the model's own
edge probabilities.

`notes/26` split the edge loss and `fn_mislink` came out largest — **473 edges, 3.42 % of
all GT edges**, where both endpoints are detected and the graph joins the wrong pair.
`notes/27` then swept pure geometry to exhaustion against it and repaired **12.5 %**.
`notes/28` confirmed on the leaderboard (0.867 → 0.880) that graph repair transfers, which
is what makes the rest worth building.

What geometry does not have is the information the pack already computed and discarded.
`predict_video` returns candidate edges as `(src, tgt, prob, dist)` — a learned probability
for **every** candidate — and the ILP consumes them into a global solution and drops the
rest. This module puts that probability back into a local assignment.

## Three costs, one assignment

For a candidate pair `(a at t, b at t+1)`:

* **distance** — plain centroid separation in µm;
* **velocity continuity** — where `a` would be at `t+1` if it kept the velocity it arrived
  with, `2·pos[a] − pos[prev(a)]`. A cell that has been moving does not stop;
* **the learned probability** — subtracted as a bonus, so a pair the model likes is cheaper.

```
cost = (1 − vw)·d + vw·dv − bonus·prob        rejected outright if d > relaxed_um
```

## Two things it must not break, both learned the hard way

* **Forks are protected, never re-derived.** The pack emits ~54 divisions per 24 datasets
  and the division term is worth 0.1 of the 1.1 maximum. An assignment solver re-deriving
  the whole frame would collapse every fork to a single child, trading a term we are bad at
  for one we are worse at. Every edge out of a forking node is held fixed.
* **A bound on how much it may change.** `notes/27` §1 measured that node-moving repairs
  buy mislinks and pay in *detection failures*, about 4:1 where they work and inverting
  when pushed — `linefit_smooth`'s `max_shift_um` is what keeps it on the right side.
  Relink gets the same kind of governor: `max_change_frac` caps the share of existing edges
  it may rewrite per frame, so a bad cost function degrades gracefully instead of
  destroying a working graph.

Hungarian for small frames, greedy for large ones. `notes/04` §7 measured optimal
assignment beating greedy by only **+0.0068** with ground-truth nodes fed in, so the
fallback is cheap insurance rather than a compromise.
"""

from __future__ import annotations

import numpy as np

__all__ = ["motion_relink", "RelinkParams"]


class RelinkParams:
    """One sweep cell. Defaults are the public notebook's published constants.

    They are a *starting point*, not a target: `notes/27` found its smoothing constants
    were already near-optimal for our graphs, and its gap radius was at the optimum, so
    they are worth trying first and worth sweeping anyway.
    """

    __slots__ = ("tight_um", "relaxed_um", "velocity_weight", "learned_bonus",
                 "max_frame_nodes", "max_change_frac")

    def __init__(self, tight_um: float = 5.9, relaxed_um: float = 9.8,
                 velocity_weight: float = 0.50, learned_bonus: float = 0.78,
                 max_frame_nodes: int = 3200, max_change_frac: float = 0.10) -> None:
        self.tight_um = float(tight_um)
        self.relaxed_um = float(relaxed_um)
        self.velocity_weight = float(velocity_weight)
        self.learned_bonus = float(learned_bonus)
        self.max_frame_nodes = int(max_frame_nodes)
        self.max_change_frac = float(max_change_frac)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"RelinkParams(tight={self.tight_um}, relaxed={self.relaxed_um}, "
                f"vw={self.velocity_weight}, bonus={self.learned_bonus}, "
                f"max_change_frac={self.max_change_frac})")


def motion_relink(
    t: np.ndarray,
    zyx: np.ndarray,
    edges: np.ndarray,
    cand: np.ndarray | None = None,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    tight_um: float = 5.9,
    relaxed_um: float = 9.8,
    velocity_weight: float = 0.50,
    learned_bonus: float = 0.78,
    max_frame_nodes: int = 3200,
    max_change_frac: float = 0.10,
) -> np.ndarray:
    """Return a new edge array with non-fork links re-assigned.

    `cand` is the pack's candidate edge table as `(K, 3+)`: columns
    `(source, target, prob, ...)`. Pass `None` to run on geometry and velocity alone —
    which is what makes the learned term's contribution measurable as a delta rather than
    assumed.

    Node positions and the node set are untouched; only the edge array changes. Edges out
    of a forking node are copied through unchanged.
    """
    t = np.asarray(t, np.int64)
    zyx = np.asarray(zyx, float)
    edges = np.asarray(edges, np.int64).reshape(-1, 2)
    n = len(t)
    if n == 0 or len(edges) == 0 or max_change_frac <= 0:
        return edges

    pos = zyx * np.asarray(scale, float)[None, :]
    out_deg = np.bincount(edges[:, 0], minlength=n)

    # Forks are held fixed. Re-deriving them would collapse a division to a single child,
    # and the division term is worth 0.1 of the 1.1 maximum.
    protected = out_deg[edges[:, 0]] >= 2
    keep = edges[protected]
    movable = edges[~protected]
    if len(movable) == 0:
        return edges

    cs, ct, cp = _cand_arrays(cand)

    # Where each node came from, so velocity is available. Built from the ORIGINAL graph:
    # a node's arrival direction is evidence regardless of what happens downstream.
    prev = np.full(n, -1, np.int64)
    prev[edges[:, 1]] = edges[:, 0]

    by_frame: dict[int, np.ndarray] = {}
    order = np.argsort(t, kind="stable")
    frames, starts = np.unique(t[order], return_index=True)
    bounds = np.append(starts, len(order))
    for i, f in enumerate(frames):
        by_frame[int(f)] = np.sort(order[bounds[i]:bounds[i + 1]])

    # Targets held by a fork are off the table; a fork's child cannot be stolen.
    fork_targets = set(keep[:, 1].tolist()) if len(keep) else set()
    # Forking sources are off the table too — their edges are all in `keep`.
    forking = out_deg >= 2

    ef = t[movable[:, 0]]
    new_edges: list[np.ndarray] = [keep] if len(keep) else []
    cand_frame = t[cs] if len(cs) else np.zeros(0, np.int64)

    for f in np.unique(ef):
        f = int(f)
        here = movable[ef == f]
        nxt = by_frame.get(f + 1)
        if nxt is None:
            new_edges.append(here)
            continue

        # EVERY non-forking node in the frame is a candidate source, and every unclaimed
        # node in the next frame a candidate target — not just the ones already linked.
        # Restricting targets to existing link targets (the first version of this) lets
        # relink permute edges but never attach to an unclaimed node, which is exactly
        # what the 336 `source_busy` mislinks in notes/26 need it to do.
        srcs = by_frame[f][~forking[by_frame[f]]]
        tgts = nxt if not fork_targets else np.array(
            [x for x in nxt.tolist() if x not in fork_targets], np.int64)
        if len(srcs) == 0 or len(tgts) == 0:
            new_edges.append(here)
            continue

        fsel = cand_frame == f if len(cs) else None
        pm = (_prob_matrix(cs[fsel], ct[fsel], cp[fsel], srcs, tgts)
              if fsel is not None and fsel.any() else None)
        # Hungarian is O(n^3); past this it is not worth the wall time for the +0.0068
        # notes/04 §7 measured optimal assignment to be worth over greedy.
        greedy = len(srcs) > max_frame_nodes or len(tgts) > max_frame_nodes
        new_edges.append(_solve(here, srcs, tgts, pos, prev, pm, tight_um,
                                relaxed_um, velocity_weight, learned_bonus,
                                max_change_frac, greedy=greedy))

    out = np.vstack([e for e in new_edges if len(e)]) if new_edges else edges
    return np.asarray(out, np.int64).reshape(-1, 2)


def _cand_arrays(cand):
    """Split the pack's candidate table into `(source, target, prob)` arrays."""
    if cand is None or len(cand) == 0:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, float),)
    cand = np.asarray(cand)
    if cand.ndim != 2 or cand.shape[1] < 3:
        return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, float),)
    return (cand[:, 0].astype(np.int64), cand[:, 1].astype(np.int64),
            cand[:, 2].astype(float))


def _prob_matrix(cs, ct, cp, srcs, tgts):
    """Scatter candidate probabilities into a dense `(len(srcs), len(tgts))` matrix.

    Vectorised deliberately. A frame here holds 500-1,000 nodes, so a nested Python loop
    over the pair grid is ~1e6 dict lookups per frame and over 1e9 across a sweep — which
    is what the first version did and why it could not have been run.
    `srcs` and `tgts` must be sorted; `searchsorted` is what makes this O(K log n).
    """
    m = np.zeros((len(srcs), len(tgts)), float)
    if len(cs) == 0:
        return m
    si = np.searchsorted(srcs, cs)
    ti = np.searchsorted(tgts, ct)
    ok = (si < len(srcs)) & (ti < len(tgts))
    if not ok.any():
        return m
    si_c = np.clip(si, 0, len(srcs) - 1)
    ti_c = np.clip(ti, 0, len(tgts) - 1)
    ok &= (srcs[si_c] == cs) & (tgts[ti_c] == ct)
    if ok.any():
        m[si[ok], ti[ok]] = cp[ok]
    return m


def _cost_matrix(srcs, tgts, pos, prev, prob, tight_um, relaxed_um, vw, bonus):
    """Cost per (source, target) pair; `inf` where the pair is not permitted."""
    d = np.linalg.norm(pos[tgts][None, :, :] - pos[srcs][:, None, :], axis=2)

    # Velocity continuity: where each source would land keeping its arrival velocity.
    # Sources with no predecessor have no velocity evidence, so they fall back to plain
    # distance rather than to an invented zero-velocity prediction.
    has_prev = prev[srcs] >= 0
    pred = pos[srcs].copy()
    if has_prev.any():
        p = prev[srcs][has_prev]
        pred[has_prev] = 2.0 * pos[srcs][has_prev] - pos[p]
    dv = np.linalg.norm(pos[tgts][None, :, :] - pred[:, None, :], axis=2)
    dv[~has_prev] = d[~has_prev]

    cost = (1.0 - vw) * d + vw * dv
    if prob is not None:
        cost = cost - bonus * prob
    # A pair inside the tight radius is preferred over one merely inside the relaxed one,
    # independent of how the learned bonus happens to score it.
    cost = np.where(d <= tight_um, cost, cost + 1.0)
    cost[d > relaxed_um] = np.inf
    return cost


def _solve(here, srcs, tgts, pos, prev, prob, tight_um, relaxed_um, vw, bonus,
           max_change_frac, greedy):
    """Apply the cheapest permitted re-assignments on top of one frame's existing edges.

    This **edits** the existing edge set rather than replacing it with the assignment.
    Replacing it silently deletes every source the solver left unassigned — with 500-node
    frames and a 9.8 µm radius the solver assigns only a fraction, and an earlier version
    of this function returned 3,039 edges from an input of 24,500 before that was caught.

    Both structural invariants are maintained while changes are applied: moving a source
    onto a target evicts whoever held that target (otherwise a merge) and drops the
    source's old link (otherwise a second child).
    """
    cost = _cost_matrix(srcs, tgts, pos, prev, prob, tight_um, relaxed_um, vw, bonus)
    if not np.isfinite(cost).any():
        return here

    pairs = _assign(cost, greedy)
    if not pairs:
        return here

    existing = {int(a): int(b) for a, b in here.tolist()}
    proposals = [(float(cost[i, j]), int(srcs[i]), int(tgts[j])) for i, j in pairs]
    changes = [(c, s, d) for c, s, d in proposals if existing.get(s) != d]
    if not changes:
        return here
    changes.sort(key=lambda csd: (csd[0], csd[1], csd[2]))

    # The governor. notes/27 §1: a repair that rewires nodes buys mislinks and pays in
    # detection failures, ~4:1 where it works and inverting when pushed. Capping the share
    # rewritten per frame means a bad cost function degrades instead of destroying.
    budget = int(max_change_frac * len(here))
    if budget <= 0:
        return here

    out = dict(existing)
    owner = {d: s for s, d in existing.items()}
    spent = 0
    for _, s, d in changes:
        if spent >= budget:
            break
        held_by = owner.get(d)
        if held_by == s:
            continue
        if held_by is not None:            # evict: two parents would be a merge
            out.pop(held_by, None)
            owner.pop(d, None)
        old = out.get(s)
        if old is not None:                # drop the old link: two children would fork
            owner.pop(old, None)
        out[s] = d
        owner[d] = s
        spent += 1

    return np.asarray(sorted(out.items()), np.int64).reshape(-1, 2)


def _assign(cost, greedy):
    """`[(i, j), ...]` one-to-one pairs of finite cost."""
    finite = np.isfinite(cost)
    if not finite.any():
        return []
    if not greedy:
        try:
            from scipy.optimize import linear_sum_assignment

            big = float(np.nanmax(cost[finite])) + 1e6
            dense = np.where(finite, cost, big)
            ri, ci = linear_sum_assignment(dense)
            return [(int(i), int(j)) for i, j in zip(ri, ci) if finite[i, j]]
        except ImportError:
            pass
    idx = np.argsort(cost[finite], kind="stable")
    rows, cols = np.nonzero(finite)
    used_r, used_c, out = set(), set(), []
    for k in idx:
        i, j = int(rows[k]), int(cols[k])
        if i in used_r or j in used_c:
            continue
        used_r.add(i)
        used_c.add(j)
        out.append((i, j))
    return out
