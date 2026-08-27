"""Graph repairs that need only `(t, zyx, edges)` — no image, no GPU, no model.

The pack's manifest says it emits the *"ILP candidate graph before notebook-level graph
repair"*, and the public notebook's config names eight repair stages. Its `licenseName`
is `None`, so none of its code is usable; what is usable is the observation that its
experiment tag reads `candidate_23_...`. **Their 0.046 over us is 23 iterations of many
small repairs, not one big idea** — which is consistent with `claude_div_probe` finding
their division machinery worth ~nothing (+0.0015) when we built it.

Five of those stages need nothing but the graph, which is what makes them worth doing
first: they run against `claude_div_probe`'s cached post-ILP graphs in seconds, with no
prediction pass, so the ablation is iterable in minutes instead of half an hour.

Each function takes and returns `(t, zyx, edges)` and never mutates its input. Two of
them change the node set, so they return remapped edges rather than expecting the caller
to fix up indices — getting that wrong is silent and catastrophic.

## What each one is aimed at

* `prune_isolated` — **the multiplier.** On the budget-stratified 24 the pack's node-budget
  multiplier is **0.9892**, i.e. below 1: it is net *over* budget and paying for it. A node
  with no edges at all contributes nothing to edge Jaccard by construction, so dropping it
  is the one repair here whose sign cannot be negative. (`notes/25` §2 called this term a
  mirage. That was measured on the confounded subset; on the honest one it is not.)
* `cap_edge_length` — edge FP. A link longer than a cell can travel in one frame is wrong.
* `single_parent_repair` — merges. The scorer collapses them anyway, keeping the lowest
  edge id; doing it here means we choose which one survives instead of the row order.
* `linefit_smooth` — node *positions*. The metric matches at 7 µm, so jitter costs matched
  endpoints. Bounded: node recall is already 0.995, so there is at most ~0.005 here.
* `close_gaps` — `fn_gap` in `pipeline/anatomy.py`. Bridges a one-frame hole by inserting
  a node. Costs budget, so it is capped.
"""

from __future__ import annotations

import numpy as np

__all__ = ["prune_isolated", "cap_edge_length", "single_parent_repair",
           "linefit_smooth", "close_gaps"]


def _as_arrays(t, zyx, edges):
    return (np.asarray(t, np.int64),
            np.asarray(zyx, float),
            np.asarray(edges, np.int64).reshape(-1, 2))


def prune_isolated(t, zyx, edges):
    """Drop nodes with no incident edge, remapping the edge indices.

    Cannot lose an edge: a dropped node had none. So this moves `total_node_ratio` down
    and leaves edge Jaccard exactly where it was — the only repair here that is
    monotone by construction rather than by measurement.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    n = len(t)
    if n == 0:
        return t, zyx, edges

    used = np.zeros(n, bool)
    if len(edges):
        used[edges[:, 0]] = True
        used[edges[:, 1]] = True
    if used.all():
        return t, zyx, edges

    keep = np.flatnonzero(used)
    remap = np.full(n, -1, np.int64)
    remap[keep] = np.arange(len(keep))
    new_edges = remap[edges] if len(edges) else edges
    return t[keep], zyx[keep], new_edges


def cap_edge_length(t, zyx, edges, scale=(1.625, 0.40625, 0.40625), max_um=14.0):
    """Drop edges whose endpoints are further apart than a cell can move in one frame.

    Nodes are left alone — pair with `prune_isolated` if the orphans should go too.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    if len(edges) == 0:
        return t, zyx, edges
    pos = zyx * np.asarray(scale, float)[None, :]
    d = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    return t, zyx, edges[d <= max_um]


def single_parent_repair(t, zyx, edges, scale=(1.625, 0.40625, 0.40625)):
    """Keep one incoming edge per node — the geometrically closest parent.

    A cell has exactly one predecessor; two incoming edges is a merge, which is not
    biology. The scorer resolves merges by keeping the **lowest edge id**, i.e. whatever
    order the rows happened to be in. Choosing the nearest parent instead is a strictly
    better-informed tie-break, and it is the only reason to do this before submission
    rather than letting the scorer do it.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    if len(edges) == 0:
        return t, zyx, edges
    pos = zyx * np.asarray(scale, float)[None, :]
    d = np.linalg.norm(pos[edges[:, 1]] - pos[edges[:, 0]], axis=1)
    # Sort by (target, distance) and keep the first row of each target group. Stable, so
    # exact distance ties fall back to row order, matching the scorer's own tie-break.
    order = np.lexsort((d, edges[:, 1]))
    tgt_sorted = edges[order, 1]
    first = np.ones(len(order), bool)
    first[1:] = tgt_sorted[1:] != tgt_sorted[:-1]
    return t, zyx, edges[np.sort(order[first])]


def linefit_smooth(t, zyx, edges, window=2, weight=0.76,
                   scale=(1.625, 0.40625, 0.40625), max_shift_um=3.2):
    """Pull each node toward a local straight-line fit of its own track.

    Cells move smoothly over a frame, so the jitter in a detected centroid is largely
    noise. The metric matches at 7 µm, so removing jitter can only help — but node recall
    is already 0.995, which caps the gain near +0.005. It is included because it is
    nearly free, not because it is expected to be large.

    Chains are followed through out-degree-1 / in-degree-1 nodes only. At a division the
    track ends for this purpose: fitting a line across a fork would drag the parent toward
    one daughter, which is exactly wrong.

    `max_shift_um` bounds the move so a bad fit cannot push a node out of its own match
    radius — the failure this repair exists to prevent.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    n = len(t)
    if n == 0 or len(edges) == 0 or weight <= 0:
        return t, zyx, edges

    s = np.asarray(scale, float)
    out_deg = np.bincount(edges[:, 0], minlength=n)
    in_deg = np.bincount(edges[:, 1], minlength=n)

    nxt = np.full(n, -1, np.int64)
    simple = out_deg[edges[:, 0]] == 1
    nxt[edges[simple, 0]] = edges[simple, 1]
    # A step belongs to a chain only if the target has exactly one parent too. Otherwise
    # this is two tracks merging, and fitting one line across them averages both.
    has_next = np.flatnonzero(nxt >= 0)
    nxt[has_next[in_deg[nxt[has_next]] != 1]] = -1

    prev = np.full(n, -1, np.int64)
    fwd = np.flatnonzero(nxt >= 0)
    prev[nxt[fwd]] = fwd

    # Gather each node's window along the chain, vectorised. A per-node Python loop with
    # polyfit would run ~700k times across one ablation; this is the same fit in O(n*W).
    # `valid` marks real neighbours, so a node near a track end is not silently padded
    # with copies of itself — that would weight its own position into its own fit.
    w = 2 * window + 1
    idx = np.tile(np.arange(n, dtype=np.int64)[:, None], (1, w))
    valid = np.zeros((n, w), bool)
    valid[:, window] = True
    for direction, step in ((-1, prev), (1, nxt)):
        j = np.arange(n, dtype=np.int64)
        ok = np.ones(n, bool)
        for k in range(1, window + 1):
            nxt_j = step[j]
            ok &= nxt_j >= 0
            j = np.where(ok, nxt_j, j)
            idx[:, window + direction * k] = j
            valid[:, window + direction * k] = ok

    cnt = valid.sum(1)
    m = valid.astype(float)
    x = t[idx].astype(float)
    y = zyx[idx]                                     # (n, w, 3)
    mean_x = (x * m).sum(1) / cnt
    mean_y = (y * m[:, :, None]).sum(1) / cnt[:, None]
    dx = (x - mean_x[:, None]) * m
    dy = (y - mean_y[:, None, :]) * m[:, :, None]
    var_x = (dx * dx).sum(1)
    slope = np.zeros_like(mean_y)
    fit_ok = (cnt >= 3) & (var_x > 0)
    slope[fit_ok] = (dx[:, :, None] * dy).sum(1)[fit_ok] / var_x[fit_ok, None]
    target = mean_y + slope * (t.astype(float) - mean_x)[:, None]

    delta = (target - zyx) * weight
    delta[~fit_ok] = 0.0
    # Bound the move so a bad fit cannot push a node out of its own 7 µm match radius —
    # the failure this repair exists to prevent, not one to introduce.
    shift_um = np.linalg.norm(delta * s[None, :], axis=1)
    over = shift_um > max_shift_um
    if over.any():
        delta[over] *= (max_shift_um / shift_um[over])[:, None]
    return t, zyx + delta, edges


def close_gaps(t, zyx, edges, scale=(1.625, 0.40625, 0.40625), max_um=5.75,
               max_added_frac=0.038, max_added_abs=1650, accept=None):
    """Bridge one-frame holes by inserting a node at the midpoint.

    A track ending at `t` and another starting at `t+2` within `max_um` is one missed
    detection, not two tracks. Inserting a node at `t+1` recovers **two** edges for one
    node of budget, which is why the trade can pay at all.

    Candidates are ranked by distance and assigned greedily, each endpoint used once, so
    no inserted node ever gains a second parent — that would be a merge, and worse, it
    would make any fork through it `malformed` under the division metric's FP rules.

    Capped two ways because the node budget is a two-sided term: `max_added_frac` of the
    current node count and `max_added_abs` outright.

    `accept(t_mid, zyx_mid) -> bool[K]` optionally vetoes candidates by their proposed
    midpoint (voxel coordinates) before any endpoint is consumed. Filtering here rather
    than after assignment matters: a rejected pair must not burn a tail or a head that a
    surviving pair could have used. `pipeline.deepcenter.FrameScorer.accept` supplies the
    learned version — the whole point being that this function otherwise invents nodes
    without ever looking at the image (`notes/27` §1, `notes/33` §1).
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    n = len(t)
    if n == 0:
        return t, zyx, edges

    out_deg = np.bincount(edges[:, 0], minlength=n) if len(edges) else np.zeros(n, int)
    in_deg = np.bincount(edges[:, 1], minlength=n) if len(edges) else np.zeros(n, int)
    pos = zyx * np.asarray(scale, float)[None, :]

    budget = min(int(max_added_frac * n), int(max_added_abs))
    if budget <= 0:
        return t, zyx, edges

    tails = np.flatnonzero(out_deg == 0)     # a track stops here
    heads = np.flatnonzero(in_deg == 0)      # a track starts here
    if len(tails) == 0 or len(heads) == 0:
        return t, zyx, edges

    by_frame_heads: dict[int, np.ndarray] = {}
    for f in np.unique(t[heads]):
        by_frame_heads[int(f)] = heads[t[heads] == f]

    cands: list[tuple[float, int, int]] = []
    for f in np.unique(t[tails]):
        f = int(f)
        h = by_frame_heads.get(f + 2)
        if h is None:
            continue
        tl = tails[t[tails] == f]
        pairs = _pairs_within(pos[tl], pos[h], max_um)
        for a, b in pairs:
            i, j = int(tl[a]), int(h[b])
            cands.append((float(np.linalg.norm(pos[j] - pos[i])), i, j))
    if not cands:
        return t, zyx, edges

    cands.sort()
    if accept is not None:
        # Score every surviving candidate's midpoint in ONE batch. Per-candidate calls
        # would re-run a heatmap per point in the worst frame ordering; the veto is only
        # cheap if it is asked in bulk.
        ci = np.asarray([c[1] for c in cands], np.int64)
        cj = np.asarray([c[2] for c in cands], np.int64)
        keep = np.asarray(accept(t[ci] + 1, 0.5 * (zyx[ci] + zyx[cj])), bool)
        if keep.shape != (len(cands),):
            raise ValueError(
                f"accept returned {keep.shape}, expected ({len(cands)},) — a veto that "
                "silently returns the wrong length would filter the wrong candidates")
        cands = [c for c, k in zip(cands, keep) if k]
        if not cands:
            return t, zyx, edges

    used_t: set[int] = set()
    used_h: set[int] = set()
    new_pos, new_t, new_edges = [], [], []
    for _, i, j in cands:
        if len(new_t) >= budget:
            break
        if i in used_t or j in used_h:
            continue
        used_t.add(i)
        used_h.add(j)
        k = n + len(new_t)
        new_t.append(t[i] + 1)
        new_pos.append(0.5 * (zyx[i] + zyx[j]))
        new_edges.append((i, k))
        new_edges.append((k, j))
    if not new_t:
        return t, zyx, edges

    return (np.concatenate([t, np.asarray(new_t, np.int64)]),
            np.vstack([zyx, np.asarray(new_pos, float)]),
            np.vstack([edges, np.asarray(new_edges, np.int64)]) if len(edges)
            else np.asarray(new_edges, np.int64))


def _pairs_within(a: np.ndarray, b: np.ndarray, radius: float) -> np.ndarray:
    """`(K, 2)` index pairs within `radius`. Shared shape with `pipeline/divisions.py`."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((0, 2), np.int64)
    try:
        from scipy.spatial import cKDTree

        lists = cKDTree(a).query_ball_tree(cKDTree(b), r=radius)
        pairs = [(i, j) for i, js in enumerate(lists) for j in js]
        return np.asarray(pairs, np.int64) if pairs else np.zeros((0, 2), np.int64)
    except ImportError:
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        i, j = np.nonzero(d <= radius)
        return np.stack([i, j], axis=1).astype(np.int64)
