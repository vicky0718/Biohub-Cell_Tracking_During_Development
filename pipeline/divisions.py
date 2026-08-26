"""Speculative division insertion on an already-linked graph.

`notes/25` §1 is the reason this exists. The pack's ILP output scores **0.0000 of the
0.1000** the division term is worth — 37 forks over 24 datasets, TP=0, FP=0, FN=27 — while
edge Jaccard is at 0.9293 and the node-budget multiplier at 1.0012. Divisions are the only
term with real room left.

The arithmetic that makes insertion worth trying, read off the official
`division_metrics.py` rather than guessed:

* `FN = D - TP` by construction and `summarise` micro-averages, so
  ``division_jaccard = TP / (FP + D)`` with ``D`` fixed by the ground truth. A true
  positive never raises the denominator; only a *chargeable* false positive does.
* A fork is chargeable only if it matched a GT node with out-degree >= 1, sat inside a GT
  division window and failed the topology test, or is structurally invalid. Ground truth
  is annotated at 1-in-8 to 1-in-167, so most forks land on unmatched nodes and cost
  nothing at all.
* The same link is also an *edge*, and there `notes/04` §10's break-even applies:
  ``p > J/(1+J)``, about 48 % at J=0.93. But one edge is worth ~1/5,000 of a dataset's edge
  Jaccard while one division is worth ~1/27 of the pooled division Jaccard, so the division
  side dominates by roughly three orders of magnitude per event.

**None of that says insertion wins** — it says the loss is bounded and the upside is the
largest unclaimed term, which is what makes a sweep worth running. `claude_div_probe`
measures it.

## Two hard constraints, from the FP rules rather than from taste

* **The sister must have in-degree 0.** Attaching a cell that already has a parent makes
  the fork `malformed` in `_branch_component_evidence`, which is an automatic false
  positive — the worst possible trade.
* **The sister must be at exactly `t+1`.** Edges that do not span one frame are dropped by
  the scorer, so such a fork would add cost with no chance of credit.

## The geometry

A dividing cell sits between its two daughters. So a candidate is ranked by how far the
parent is from the **midpoint of the two daughters** — small for a real division, large
when a nearby unrelated cell is being roped in. Two gates gate it first: the parent must be
within `max_um` of the new daughter, and the two daughters within `sister_max_um` of each
other.
"""

from __future__ import annotations

import numpy as np

__all__ = ["insert_divisions", "DivisionParams"]


class DivisionParams:
    """Container for the three knobs, so a sweep can name a cell in one object.

    Defaults are the public notebook's published constants (`SAFE_DIV_MAX_UM`,
    `SAFE_DIV_SISTER_MAX_UM`, `SAFE_DIV_FRAME_FRAC_CAP`), used as a *starting point for the
    sweep* — they are the only externally-attested operating point, not a target.
    """

    __slots__ = ("max_um", "sister_max_um", "frame_frac_cap")

    def __init__(self, max_um: float = 4.5, sister_max_um: float = 6.8,
                 frame_frac_cap: float = 0.008) -> None:
        self.max_um = float(max_um)
        self.sister_max_um = float(sister_max_um)
        self.frame_frac_cap = float(frame_frac_cap)

    def __repr__(self) -> str:  # pragma: no cover - display only
        return (f"DivisionParams(max_um={self.max_um}, "
                f"sister_max_um={self.sister_max_um}, "
                f"frame_frac_cap={self.frame_frac_cap})")


def insert_divisions(
    t: np.ndarray,
    zyx: np.ndarray,
    edges: np.ndarray,
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625),
    max_um: float = 4.5,
    sister_max_um: float = 6.8,
    frame_frac_cap: float = 0.008,
) -> np.ndarray:
    """Propose extra `(parent, sister)` edges that turn linked nodes into forks.

    Returns only the **new** edges, `(K, 2)` int64, so a caller can score
    `edges` and `vstack([edges, new])` against each other without rebuilding anything.
    `frame_frac_cap = 0` returns an empty `(0, 2)` array, which is the sweep's control.

    `zyx` is in voxel units and `scale` converts it to µm — the same convention as the
    metric's 7 µm matching radius, so the radii here mean what they say.

    Every returned edge satisfies, by construction:

    * source has out-degree exactly 1 in `edges` (it becomes a 2-fork, never a 3-fork);
    * target has in-degree 0 in `edges` and appears at most once in the result (no merge,
      so never `malformed`);
    * `t[target] == t[source] + 1`.
    """
    t = np.asarray(t, dtype=np.int64)
    zyx = np.asarray(zyx, dtype=float)
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    n = len(t)
    if n == 0 or frame_frac_cap <= 0 or len(edges) == 0:
        return np.zeros((0, 2), np.int64)

    pos = zyx * np.asarray(scale, dtype=float)[None, :]

    out_deg = np.bincount(edges[:, 0], minlength=n)
    in_deg = np.bincount(edges[:, 1], minlength=n)
    # The one existing child of each out-degree-1 node. Built by scattering rather than
    # searching: several million edges make a per-node lookup the wrong shape of loop.
    child = np.full(n, -1, np.int64)
    single = out_deg[edges[:, 0]] == 1
    child[edges[single, 0]] = edges[single, 1]

    order = np.argsort(t, kind="stable")
    frames, starts = np.unique(t[order], return_index=True)
    bounds = np.append(starts, len(order))
    at_frame = {int(f): order[bounds[i]:bounds[i + 1]]
                for i, f in enumerate(frames)}

    new: list[tuple[int, int]] = []
    for f in frames:
        f = int(f)
        here, nxt = at_frame.get(f), at_frame.get(f + 1)
        if nxt is None or here is None:
            continue
        cap = int(frame_frac_cap * len(here))
        if cap <= 0:
            continue

        parents = here[(out_deg[here] == 1) & (child[here] >= 0)]
        sisters = nxt[in_deg[nxt] == 0]
        if len(parents) == 0 or len(sisters) == 0:
            continue

        pairs = _pairs_within(pos[parents], pos[sisters], max_um)
        if len(pairs) == 0:
            continue
        pi, si = pairs[:, 0], pairs[:, 1]
        p_idx, s_idx = parents[pi], sisters[si]
        c_idx = child[p_idx]

        # Gate 2: the two daughters must be plausible siblings, not a daughter and a
        # stranger that merely happens to sit near the parent.
        sib = np.linalg.norm(pos[c_idx] - pos[s_idx], axis=1)
        keep = sib <= sister_max_um
        if not keep.any():
            continue
        p_idx, s_idx, c_idx = p_idx[keep], s_idx[keep], c_idx[keep]

        # Rank: a real parent sits between its daughters, so score by the parent's
        # distance to their midpoint. Sorting by (cost, source, target) keeps the greedy
        # assignment deterministic when costs tie.
        cost = np.linalg.norm(pos[p_idx] - 0.5 * (pos[c_idx] + pos[s_idx]), axis=1)
        rank = np.lexsort((s_idx, p_idx, cost))

        used_p: set[int] = set()
        used_s: set[int] = set()
        taken = 0
        for k in rank:
            if taken >= cap:
                break
            p, s = int(p_idx[k]), int(s_idx[k])
            if p in used_p or s in used_s:
                continue
            used_p.add(p)
            used_s.add(s)
            new.append((p, s))
            taken += 1

    if not new:
        return np.zeros((0, 2), np.int64)
    return np.asarray(new, np.int64)


def _pairs_within(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """`(K, 2)` index pairs `(i, j)` with `|a[i] - b[j]| <= radius`.

    A KD-tree when scipy is importable, brute force otherwise. The brute-force path is
    not a nicety: it keeps this module usable where the pack's wheel set is not installed,
    and it is what the unit tests exercise for the small cases.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), np.int64)
    try:
        from scipy.spatial import cKDTree

        ta, tb = cKDTree(a), cKDTree(b)
        lists = ta.query_ball_tree(tb, r=radius)
        pairs = [(i, j) for i, js in enumerate(lists) for j in js]
        return (np.asarray(pairs, np.int64) if pairs
                else np.zeros((0, 2), np.int64))
    except ImportError:
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        i, j = np.nonzero(d <= radius)
        return np.stack([i, j], axis=1).astype(np.int64)
