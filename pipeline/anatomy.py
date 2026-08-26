"""Where the edge loss actually is: gap, mislink, or never detected.

`notes/25` decomposed the score *across* its three terms and found the whole gap in
divisions. `claude_div_probe` then priced divisions at **+0.0015** and closed that door.
So the loss is in the edge term, and this module splits it *within* that term — the
decomposition that should have come first.

On the budget-stratified 24, the pack's post-ILP graph scores `edge_jaccard = 0.8902`,
a deficit of **0.1098**, while node recall is **0.995**. Nearly every annotated cell is
found. So the missing edges are not missing cells: **both endpoints are detected and the
graph links them wrong, or not at all.** Those two failures have different fixes and
different costs, and nothing so far has said which dominates.

`notes/04` §7's "linking is worth at most 0.015" does not apply here and should not be
read across. That ceiling was measured with ground-truth nodes fed in, where each cell
competes with a handful of neighbours. The pack predicts 5,000-57,000 nodes against
50-1,950 annotated ones, so every real link is contested by 20-150x more candidates.
Linking in that field is a different problem.

## The buckets

Every GT edge lands in exactly one, and they sum to the GT edge count — which is also
`count_edges`'s FN plus its TP, making the sum a real check rather than a formality.

| bucket | meaning | the repair that would fix it |
|---|---|---|
| `tp` | both endpoints matched, and the prediction links them | — |
| `fn_gap` | both matched, source has no successor, target no predecessor | gap closing |
| `fn_mislink` | both matched, but one end is already committed elsewhere | motion relink |
| `fn_detect` | an endpoint was never matched | nothing at graph level |
| `fn_nonconsec` | the GT edge does not span exactly t->t+1 | nothing; the scorer drops it |

The classification runs against `purescore.survivors`, not against the raw predicted
edges. That matters: an edge the out-degree cap truncated is not a link the metric ever
saw, so counting it as "linked" would report a mislink as a success.
"""

from __future__ import annotations

import numpy as np

from harness.purescore import DEFAULT_SCALE, MAX_DISTANCE, survivors

__all__ = ["edge_anatomy", "BUCKETS"]

# Order matters: reports print in this order, and it runs from "we did it" through
# "fixable" to "not fixable at this level".
BUCKETS = ("tp", "fn_gap", "fn_mislink", "fn_detect", "fn_nonconsec")


def edge_anatomy(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    pred_edges: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    gt_edges: np.ndarray,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Classify every GT edge, and split the mislinks by which end was taken.

    Returns counts for each of `BUCKETS`, plus:

    * ``n_gt_edges``  — the sum of the buckets, by construction
    * ``source_busy`` — mislinks where the matched source already links elsewhere
    * ``target_busy`` — mislinks where the matched target already has a parent
                        (a subset overlap with ``source_busy`` is possible; both are
                        reported so the two relink directions can be sized separately)
    * ``reachable``   — ``(fn_gap + fn_mislink) / n_gt_edges``, the share of the edge
                        deficit that any graph-level repair could reach at all
    * ``edge_tp``/``edge_fp``/``edge_fn`` — the scorer's own counts, so a caller can
                        check the buckets against the metric rather than trusting them
    """
    gt_t = np.asarray(gt_t)
    gt_edges = np.asarray(gt_edges, int).reshape(-1, 2)
    pred_edges = np.asarray(pred_edges, int).reshape(-1, 2)
    n_gt_edges = len(gt_edges)

    out = {b: 0 for b in BUCKETS}
    out.update({"n_gt_edges": n_gt_edges, "source_busy": 0, "target_busy": 0,
                "reachable": float("nan"), "edge_tp": 0, "edge_fp": 0,
                "edge_fn": n_gt_edges, "n_matched_nodes": 0, "n_pred_nodes": len(pred_t)})
    if n_gt_edges == 0:
        return out

    s = survivors(pred_t, pred_zyx, pred_edges, gt_t, gt_zyx, gt_edges,
                  scale, max_distance)
    matched, keep = s["matched"], s["keep"]
    out["n_matched_nodes"] = s["n_matched"]

    # `matched` is per PREDICTION (the GT index it claimed). Invert it once; reading it
    # the other way round is a silent error whenever the graphs differ in size, which is
    # always. `pipeline/detector.py::paired_recall` records the same trap.
    gt_to_pred = np.full(len(gt_t), -1, np.int64)
    sel = matched >= 0
    gt_to_pred[matched[sel]] = np.flatnonzero(sel)

    # Only surviving edges count as links, and only their endpoints count as committed.
    live = pred_edges[keep] if len(pred_edges) else pred_edges
    n_pred = len(np.asarray(pred_t))
    out_deg = np.bincount(live[:, 0], minlength=n_pred) if len(live) else np.zeros(n_pred, int)
    in_deg = np.bincount(live[:, 1], minlength=n_pred) if len(live) else np.zeros(n_pred, int)
    linked = set(map(tuple, live.tolist())) if len(live) else set()

    u, v = gt_edges[:, 0], gt_edges[:, 1]
    consec = gt_t[v] == gt_t[u] + 1
    out["fn_nonconsec"] = int((~consec).sum())

    pu, pv = gt_to_pred[u], gt_to_pred[v]
    both = consec & (pu >= 0) & (pv >= 0)
    out["fn_detect"] = int((consec & ~both).sum())

    idx = np.flatnonzero(both)
    if len(idx):
        a, b = pu[idx], pv[idx]
        is_tp = np.fromiter(((int(x), int(y)) in linked for x, y in zip(a, b)),
                            bool, len(idx))
        out["tp"] = int(is_tp.sum())

        miss = ~is_tp
        src_busy = miss & (out_deg[a] > 0)
        tgt_busy = miss & (in_deg[b] > 0)
        out["source_busy"] = int(src_busy.sum())
        out["target_busy"] = int(tgt_busy.sum())
        out["fn_mislink"] = int((src_busy | tgt_busy).sum())
        # Both ends free and still unlinked: nothing competed for them, the linker just
        # did not join them. That is the gap-closing case.
        out["fn_gap"] = int((miss & ~src_busy & ~tgt_busy).sum())

    out["edge_tp"] = int(s["is_tp"].sum())
    out["edge_fp"] = int(s["pred_valid"].sum()) - out["edge_tp"]
    out["edge_fn"] = n_gt_edges - out["edge_tp"]
    out["reachable"] = (out["fn_gap"] + out["fn_mislink"]) / n_gt_edges
    return out


def summarise_anatomy(rows: list[dict]) -> dict:
    """Pool per-dataset anatomies by summing counts, the way the metric pools edges.

    Micro-averaged, not a mean of fractions: `harness/purescore.py::summarise` pools edge
    counts before dividing, so a per-dataset mean here would not line up with the score
    it is meant to explain.
    """
    keys = (*BUCKETS, "n_gt_edges", "source_busy", "target_busy",
            "edge_tp", "edge_fp", "edge_fn", "n_matched_nodes", "n_pred_nodes")
    total = {k: int(sum(r.get(k, 0) for r in rows)) for k in keys}
    n = total["n_gt_edges"]
    total["n"] = len(rows)
    total["reachable"] = (total["fn_gap"] + total["fn_mislink"]) / n if n else float("nan")
    denom = total["edge_tp"] + total["edge_fp"] + total["edge_fn"]
    total["edge_jaccard"] = total["edge_tp"] / denom if denom else float("nan")
    return total
