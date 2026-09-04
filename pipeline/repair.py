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
           "linefit_smooth", "close_gaps", "prune_short_tracks", "rank_budget_prune"]


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
                   scale=(1.625, 0.40625, 0.40625), max_shift_um=3.2,
                   static_um=0.0):
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

    `static_um` treats a slow-enough chain as **stationary** and falls back to the window
    mean instead of the line. `notes/59` measured **8.4% of ground-truth links at exactly
    0.0 µm displacement** (10,772 of 128,883) — frozen frames, because the volumes are
    crops of one master acquisition, plus annotations interpolated between labelled
    frames. Where the truth is static our detections still jitter, a line fit through that
    jitter has a spurious slope, and smoothing then drags the node *along* it, away from
    the fixed position it should sit at. Zeroing the slope pulls toward the window mean,
    which is the right estimator for a static point. Default 0.0 leaves behaviour exactly
    as it was.
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
    if static_um > 0:
        # `slope` is voxels per frame; scale it to µm per frame to compare against a
        # physical threshold. Below it the chain is not moving and the slope is fitted
        # noise (notes/59), so fall back to the window mean by zeroing it.
        speed_um = np.linalg.norm(slope * s[None, :], axis=1)
        slope[speed_um < static_um] = 0.0
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
               max_added_frac=0.038, max_added_abs=1650, accept=None, max_gap=1):
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

    `max_gap` is the largest hole in FRAMES that may be bridged. 1 (the default) is the
    original behaviour: tail at `f`, head at `f+2`, one node inserted. 2 also joins a head
    at `f+3` with two interpolated nodes. `notes/39`'s audit found the 0.927 public notebook
    runs `GAP_CLOSE_MAX_GAP = 2` where we have only ever done 1.

    Candidates are ranked by `(gap, distance)`, not distance alone, so a one-frame bridge
    always outranks a two-frame one over the same span. A wider hole is strictly more
    speculative for the same geometry, and this keeps `max_gap=1` byte-identical.
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

    cands: list[tuple[int, float, int, int]] = []
    for f in np.unique(t[tails]):
        f = int(f)
        tl = tails[t[tails] == f]
        for gap in range(1, int(max_gap) + 1):
            h = by_frame_heads.get(f + gap + 1)
            if h is None:
                continue
            # The radius scales with the hole: a cell crossing two frames may legitimately
            # travel twice as far. Using a fixed max_um would make wide gaps unreachable
            # rather than merely rarer, which is not the trade being tested.
            pairs = _pairs_within(pos[tl], pos[h], max_um * gap)
            for a, b in pairs:
                i, j = int(tl[a]), int(h[b])
                cands.append((gap, float(np.linalg.norm(pos[j] - pos[i])), i, j))
    if not cands:
        return t, zyx, edges

    cands.sort()
    if accept is not None:
        # Score every surviving candidate's midpoint in ONE batch. Per-candidate calls
        # would re-run a heatmap per point in the worst frame ordering; the veto is only
        # cheap if it is asked in bulk.
        ci = np.asarray([c[2] for c in cands], np.int64)
        cj = np.asarray([c[3] for c in cands], np.int64)
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
    for gap, _, i, j in cands:
        if len(new_t) + gap > budget:
            continue          # a wider gap may not fit where a narrower one still would
        if i in used_t or j in used_h:
            continue
        used_t.add(i)
        used_h.add(j)
        # Interpolate `gap` nodes evenly between the endpoints and chain them.
        prev = i
        for step in range(1, gap + 1):
            k = n + len(new_t)
            frac = step / (gap + 1.0)
            new_t.append(t[i] + step)
            new_pos.append((1.0 - frac) * zyx[i] + frac * zyx[j])
            new_edges.append((prev, k))
            prev = k
        new_edges.append((prev, j))
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


def _components(t, edges, n):
    """Union-find over `edges`, returning a root label per node."""
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    return np.array([find(i) for i in range(n)], np.int64)


def rank_budget_prune(t, zyx, edges, n_target, scale=(1.625, 0.40625, 0.40625),
                      mode="geometry", keep_division_components=True):
    """Drop whole tracks, worst-first, until the node count is within `n_target`.

    `notes/51`: the two thinning rules this project has tried both cut at the DETECTION
    stage — `pool_kernel_um` (`notes/46`, a spatial NMS radius) and `det_threshold`
    (`notes/48`/`49`, a confidence cut). Both destroyed ground-truth cells as fast as
    anything else, because a detector-stage cut cannot know which detections will end up in
    a good track.

    This is the third rule, and r35's `linker.py` is where it comes from — its `TrackConfig`
    carries `max_pred_nodes` beside `rank_tracks_by_geometry`, commented *"Pivot H — drop
    short false tracks to cut |V̂|/φ penalty"* and *"R11 — rank tracks by link geometry
    (tight long tracks) under budget"*. Cutting AFTER linking is different in kind: dropping
    a junk track removes its nodes (a budget gain) **and** its false-positive edges (a
    Jaccard gain), where a detector-stage cut removes nodes that a good track needed.

    `n_target` is the dataset's own `estimated_number_of_nodes`, optionally scaled. That is
    the half `notes/04` §9 flagged and this project never built: *"the two datasets that are
    the leaderboard have node budgets 11× apart, 64 vs 698 cells per frame. A detector with
    one global threshold cannot serve both."* We still ship one global `DET_THRESHOLD`.

    The metric being exploited (`harness/purescore.py::per_sample`, confirmed on the forum
    by two independent readings of the released evaluator in thread 739018):
    ``adj = max(0, edge_J * (1 - 0.1 * (n_pred - n_total) / n_total))`` — a floor at zero and
    **no ceiling**, so predicting under budget multiplies the score above 1.

    `mode`:
      * ``"geometry"`` — r35's rule. Rank by frame span, then by tightness (median step
        length, smaller is better). Long smooth tracks survive; short erratic ones go.
      * ``"length"`` — span alone, the ablation that says whether tightness carries anything.
      * ``"isolated"`` — drop only nodes in no edge at all. The free half: such a node cannot
        contribute a TP edge, so it is pure budget cost. Ignores `n_target`.

    Ties break on span so the result does not depend on node ordering. Components holding a
    fork are kept under `keep_division_components`, matching `prune_short_tracks` — a
    division is worth a tenth of the metric and its component is often short by nature.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    n = len(t)
    if n == 0:
        return t, zyx, edges

    roots = _components(t, edges, n)

    if mode == "isolated":
        deg = np.zeros(n, np.int64)
        if len(edges):
            deg = np.bincount(edges[:, :2].ravel(), minlength=n)
        keep = deg > 0
    else:
        if not (n_target == n_target) or n_target <= 0 or n <= n_target:
            return t, zyx, edges
        has_fork = np.zeros(n, bool)
        if len(edges):
            out_deg = np.bincount(edges[:, 0], minlength=n)
            for r in np.unique(roots[out_deg >= 2]):
                has_fork[roots == r] = True

        um = np.asarray(zyx, float) * np.asarray(scale, float)
        order = np.argsort(t, kind="stable")
        stats = {}
        for r in np.unique(roots):
            sel = np.flatnonzero(roots == r)
            sel = sel[np.argsort(t[sel], kind="stable")]
            span = int(t[sel].max() - t[sel].min()) + 1
            if len(sel) > 1:
                step = float(np.median(np.linalg.norm(np.diff(um[sel], axis=0), axis=1)))
            else:
                step = float("inf")      # a singleton has no geometry to vouch for it
            stats[int(r)] = (span, step, len(sel), bool(has_fork[sel[0]]))

        # Best first. Forks first of all, then long, then tight.
        def key(r):
            span, step, _, fork = stats[r]
            if mode == "length":
                return (not (fork and keep_division_components), -span)
            return (not (fork and keep_division_components), -span, step)

        keep = np.zeros(n, bool)
        used = 0
        for r in sorted(stats, key=key):
            size = stats[r][2]
            if used + size > n_target and used > 0:
                continue          # skip this track, a shorter one may still fit
            keep[roots == r] = True
            used += size

    if keep.all():
        return t, zyx, edges
    remap = np.full(n, -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    e = edges[keep[edges[:, 0]] & keep[edges[:, 1]]] if len(edges) else edges
    return t[keep], zyx[keep], (remap[e] if len(e) else np.zeros((0, 2), np.int64))


def prune_short_tracks(t, zyx, edges, min_frames: int = 6,
                       keep_division_components: bool = True):
    """Drop connected components spanning fewer than `min_frames` frames.

    `notes/39`'s audit: the 0.927 public notebook runs `OUTPUT_MIN_TRACK_LEN = 6` with
    `OUTPUT_FILTER_SHORT_TRACKS = 1`; we have never pruned anything. A two-node fragment
    is almost always a detection artifact, and it costs twice — the spurious edge misses,
    and both nodes eat the node budget, which `notes/35` §3 measured as worth ~0.01 of
    score on its own.

    `keep_division_components` mirrors their `OUTPUT_KEEP_DIVISION_COMPONENTS = 1`: a short
    component containing a fork is kept, because a division is the one structure whose
    shortness is expected rather than suspicious, and `division_jaccard` is a tenth of the
    metric.

    Span is measured in FRAMES, not nodes: a component with 8 nodes all in one frame is not
    an 8-frame track. Counting nodes instead would keep exactly the dense-region clutter
    this is meant to remove.
    """
    t, zyx, edges = _as_arrays(t, zyx, edges)
    n = len(t)
    if n == 0 or min_frames <= 1:
        return t, zyx, edges

    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for u, v in edges:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    roots = np.array([find(i) for i in range(n)], np.int64)

    has_fork = np.zeros(n, bool)
    if len(edges):
        out_deg = np.bincount(edges[:, 0], minlength=n)
        for r in np.unique(roots[out_deg >= 2]):
            has_fork[roots == r] = True

    keep = np.ones(n, bool)
    for r in np.unique(roots):
        sel = roots == r
        span = int(t[sel].max() - t[sel].min()) + 1
        if span < min_frames and not (keep_division_components and has_fork[sel][0]):
            keep[sel] = False
    if keep.all():
        return t, zyx, edges

    remap = np.full(n, -1, np.int64)
    remap[keep] = np.arange(int(keep.sum()))
    e = edges[keep[edges[:, 0]] & keep[edges[:, 1]]] if len(edges) else edges
    return t[keep], zyx[keep], (remap[e] if len(e) else np.zeros((0, 2), np.int64))
