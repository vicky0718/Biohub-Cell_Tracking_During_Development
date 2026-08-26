"""The official edge metric, reimplemented on numpy/scipy alone.

**Why this exists.** `harness/scorer.py` calls the organisers' code directly, which is
the right thing to do — a local score should be the leaderboard's score. But that code
imports `tracksdata`, which requires `numpy>2`, and Kaggle's kernels pin `numpy<2`.
Installing it there either fails to resolve or rewrites numpy underneath the running
kernel and takes `scipy.spatial` down with it (both failure modes observed, see
`notes/04-recon-results.md`). So on Kaggle we cannot score at all — which means no
sweep, no fold gate, no honest comparison, and no reason to run anything.

This module is the way out: the same computation, expressed over plain arrays, with
**zero dependencies beyond numpy and scipy**. It is not a paraphrase — every step
below was transcribed from the source and is checked against the official scorer by
`probes/verify_purescore.py`, which fuzzes duplicate edges, merges, out-degree
overflow, non-consecutive edges and unmatchable frames and requires exact agreement
on TP/FP/FN.

**Scope.** Edge terms exactly. The division term is exact only under
`no_forks=True` — when the prediction has no node with out-degree > 1, no predicted
fork can be a TP or an evaluable FP, so `division_jaccard` is 0 when the GT has any
division and undefined (dropped from the score) when it does not. That is the regime
`pipeline.classical` runs in (`allow_divisions=False`, settled by recon §10), and the
function refuses to guess outside it.

Graphs are passed as arrays, not objects:

    t     : (N,) int      frame index per node
    zyx   : (N, 3) float  centroid in voxel units (scale is applied here)
    edges : (E, 2) int    (source_index, target_index) into the node arrays;
                          row order IS the edge id, which the scorer's tie-breaks use
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)
ADJUSTMENT_ALPHA = 0.1
SCORE_DIVISION_WEIGHT = 0.1
MAX_DISTANCE = 7.0


class EdgeCounts(NamedTuple):
    tp: int
    fp: int
    fn: int
    n_pred_nodes: int
    n_matched_nodes: int


# --------------------------------------------------------------------------
# node matching
# --------------------------------------------------------------------------

def _match_frame(
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One frame of `tracksdata.metrics._ctc_metrics._match_single_frame`.

    Returns ``(pred_idx, gt_idx)`` of matched pairs, local to this frame.

    Faithful to three details that are easy to get wrong:
      * the objective maximises ``1/(1+d)``, not minimises ``d`` — a different
        problem in principle, and the tie-breaks differ;
      * the sparse matrix's shape is *inferred* from the largest candidate index,
        so trailing nodes with no candidate at all are outside the matrix;
      * `min_weight_full_bipartite_matching` demands a perfect matching on the
        smaller side and raises otherwise, hence the two-step fallback.
    """
    if len(pred_xyz) == 0 or len(gt_xyz) == 0:
        return np.zeros(0, int), np.zeros(0, int)

    # cdist(comp, ref) in the original: rows = pred, cols = gt.
    d = cdist(pred_xyz, gt_xyz)
    comp_i, ref_i = np.nonzero(d <= max_distance)
    if len(comp_i) == 0:
        return np.zeros(0, int), np.zeros(0, int)

    w = 1.0 / (1.0 + d[comp_i, ref_i])
    # csr_array((data, (rows, cols))) with rows = ref (gt), cols = comp (pred).
    weights = sp.csr_array((w.astype(np.float32), (ref_i, comp_i)))

    try:
        rows_id, cols_id = sp.csgraph.min_weight_full_bipartite_matching(
            weights, maximize=True)
    except ValueError:
        fill_value = -1.0
        dense = np.full(weights.shape, fill_value, dtype=np.float32)
        coo = weights.tocoo()
        dense[coo.row, coo.col] = coo.data
        # The official code retries the sparse solver on the filled matrix first;
        # a full matrix always admits a full matching, so the dense assignment
        # below reaches the same optimum and skips a redundant conversion.
        rows_id, cols_id = linear_sum_assignment(dense, maximize=True)
        keep = ~np.isclose(dense[rows_id, cols_id], fill_value)
        rows_id, cols_id = rows_id[keep], cols_id[keep]

    return np.asarray(cols_id, int), np.asarray(rows_id, int)


def match_nodes(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> np.ndarray:
    """Match predicted nodes to GT nodes frame by frame.

    Returns ``matched`` of length ``len(pred_t)``: the GT node index each predicted
    node matched, or -1. Matching is one-to-one **within a frame**; nodes in frames
    the other graph does not have simply go unmatched.
    """
    s = np.asarray(scale, float)
    pred_s = np.asarray(pred_zyx, float) * s
    gt_s = np.asarray(gt_zyx, float) * s
    matched = np.full(len(pred_t), -1, dtype=np.int64)

    pred_t = np.asarray(pred_t)
    gt_t = np.asarray(gt_t)
    for t in np.intersect1d(np.unique(pred_t), np.unique(gt_t)):
        p_idx = np.flatnonzero(pred_t == t)
        g_idx = np.flatnonzero(gt_t == t)
        pi, gi = _match_frame(pred_s[p_idx], gt_s[g_idx], max_distance)
        if len(pi):
            matched[p_idx[pi]] = g_idx[gi]
    return matched


# --------------------------------------------------------------------------
# edge scoring
# --------------------------------------------------------------------------

def count_edges(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    pred_edges: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    gt_edges: np.ndarray,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> EdgeCounts:
    """Edge TP/FP/FN, reproducing `tracking_cellmot.metrics._evaluate_matched_graph`.

    The scorer silently repairs four things before counting, and each one can turn a
    prediction into something other than what was submitted:

    1. duplicate ``(source, target)`` pairs are collapsed to one;
    2. edges not spanning exactly ``t -> t+1`` are **dropped** (so a t->t+2 bridge
       scores the same as no edge at all);
    3. merges — several predicted edges landing on one GT edge — keep the lowest
       edge id only;
    4. out-degree above 2 is truncated to the two lowest edge ids.

    An edge then counts as a false positive only if it is ``pred_valid``: its source
    matched a GT node with an out-edge, **or** its target matched a GT node with an
    in-edge. Everything else is invisible to the metric, which is why FPs on the
    unannotated 96.5% of cells are free.
    """
    s = survivors(pred_t, pred_zyx, pred_edges, gt_t, gt_zyx, gt_edges,
                  scale, max_distance)
    tp = int(s["is_tp"].sum())
    fp = int(s["pred_valid"].sum()) - tp
    fn = s["n_gt_edges"] - tp
    # The official code asserts the same thing: every TP must be pred_valid.
    assert not (s["is_tp"] & ~s["pred_valid"]).any(), "TP outside pred_valid — matching is wrong"
    return EdgeCounts(tp, fp, fn, len(np.asarray(pred_t)), s["n_matched"])


def survivors(
    pred_t: np.ndarray,
    pred_zyx: np.ndarray,
    pred_edges: np.ndarray,
    gt_t: np.ndarray,
    gt_zyx: np.ndarray,
    gt_edges: np.ndarray,
    scale: tuple[float, float, float] = DEFAULT_SCALE,
    max_distance: float = MAX_DISTANCE,
) -> dict:
    """Everything `count_edges` computes, before it collapses to three integers.

    Extracted so that anything wanting to ask *which* edges survived the scorer's four
    silent repairs, or *which* GT node a prediction claimed, measures the same graph the
    metric counts rather than a hand-rolled second opinion. `pipeline/anatomy.py` is the
    caller that needed it: classifying a missing GT edge as a gap or a mislink is only
    meaningful against the post-repair prediction, because a link that the out-degree cap
    truncated is not a link the metric ever saw.

    Returns ``matched`` (per PREDICTION, the GT index it claimed, or -1), ``keep`` (per
    predicted edge, survived the four repairs), ``is_tp``, ``pred_valid``, ``ms``/``mt``
    (the matched GT index of each predicted edge's endpoints), and the two counts.

    This is a pure extraction: `count_edges` produced these exact arrays inline before,
    and its numbers are unchanged.
    """
    pred_edges = np.asarray(pred_edges, int).reshape(-1, 2)
    gt_edges = np.asarray(gt_edges, int).reshape(-1, 2)
    n_gt_edges = len(gt_edges)

    matched = match_nodes(pred_t, pred_zyx, gt_t, gt_zyx, scale, max_distance)
    n_matched = int((matched >= 0).sum())
    empty = np.zeros(len(pred_edges), bool)

    if len(pred_edges) == 0:
        return {"matched": matched, "keep": empty, "is_tp": empty, "pred_valid": empty,
                "ms": empty.astype(int), "mt": empty.astype(int),
                "n_matched": n_matched, "n_gt_edges": n_gt_edges}

    src, tgt = pred_edges[:, 0], pred_edges[:, 1]
    eid = np.arange(len(pred_edges))

    # (1) duplicate (source, target) pairs — keep the first occurrence.
    _, first = np.unique(np.stack([src, tgt], 1), axis=0, return_index=True)
    keep = np.zeros(len(pred_edges), bool)
    keep[first] = True

    # (2) consecutive frames only.
    pred_t = np.asarray(pred_t)
    keep &= (pred_t[tgt] - pred_t[src]) == 1

    # matched-edge mask: both endpoints matched, and the GT pair is a GT edge.
    gt_edge_set = set(map(tuple, gt_edges.tolist()))
    ms, mt = matched[src], matched[tgt]
    both = (ms >= 0) & (mt >= 0)
    is_tp = np.zeros(len(pred_edges), bool)
    for i in np.flatnonzero(both & keep):
        is_tp[i] = (int(ms[i]), int(mt[i])) in gt_edge_set

    # (3) merges: among both-matched survivors, one edge per (matched_src, matched_tgt).
    cand = np.flatnonzero(both & keep)
    if len(cand):
        _, first_merge = np.unique(np.stack([ms[cand], mt[cand]], 1), axis=0,
                                   return_index=True)
        drop = np.setdiff1d(cand, cand[first_merge])
        keep[drop] = False

    # (4) out-degree cap: two lowest edge ids per source. eid is already ascending,
    #     so a stable order by source is enough to find each source's first two.
    surv = np.flatnonzero(keep)
    if len(surv):
        order = surv[np.argsort(src[surv], kind="stable")]
        s_sorted = src[order]
        rank = np.ones(len(order), int)
        if len(order) > 1:
            same = s_sorted[1:] == s_sorted[:-1]
            r = 1
            for i in range(1, len(order)):
                r = r + 1 if same[i - 1] else 1
                rank[i] = r
        keep[order[rank > 2]] = False

    is_tp &= keep

    # pred_valid: the only edges the metric can see at all.
    gt_out = np.zeros(len(gt_t), bool)
    gt_in = np.zeros(len(gt_t), bool)
    if n_gt_edges:
        gt_out[gt_edges[:, 0]] = True
        gt_in[gt_edges[:, 1]] = True
    out_valid = np.where(ms >= 0, gt_out[np.clip(ms, 0, None)], False)
    in_valid = np.where(mt >= 0, gt_in[np.clip(mt, 0, None)], False)
    pred_valid = (out_valid | in_valid) & keep

    return {"matched": matched, "keep": keep, "is_tp": is_tp, "pred_valid": pred_valid,
            "ms": ms, "mt": mt, "n_matched": n_matched, "n_gt_edges": n_gt_edges}


# --------------------------------------------------------------------------
# aggregation — mirrors metrics.per_sample_metrics / metrics.summarise
# --------------------------------------------------------------------------

def per_sample(counts: EdgeCounts, n_total: float, n_gt_nodes: int,
               n_gt_divisions: int = 0, no_forks: bool = True) -> dict:
    """Per-dataset metric row.

    ``n_total`` is the GEFF ``estimated_number_of_nodes``; pass ``float('nan')`` when
    it is unavailable and the adjusted Jaccard comes back NaN rather than wrong.
    """
    if not no_forks:
        raise NotImplementedError(
            "The division term is only exact for fork-free predictions. Score those "
            "through harness/scorer.py against the official implementation."
        )
    denom = counts.tp + counts.fp + counts.fn
    edge_j = counts.tp / denom if denom > 0 else float("nan")

    ratio = ((counts.n_pred_nodes - n_total) / n_total) if n_total > 0 else float("nan")
    if edge_j == edge_j and ratio == ratio:
        adj = max(0.0, edge_j * (1 - ADJUSTMENT_ALPHA * ratio))
    else:
        adj = float("nan")

    return {
        "edge_tp": counts.tp, "edge_fp": counts.fp, "edge_fn": counts.fn,
        # No predicted fork can be a TP or an evaluable FP, so every GT division is a FN.
        "division_tp": 0, "division_fp": 0, "division_fn": int(n_gt_divisions),
        "num_pred_nodes": counts.n_pred_nodes,
        "node_recall": counts.n_matched_nodes / n_gt_nodes if n_gt_nodes else float("nan"),
        "total_node_ratio": ratio,
        "edge_jaccard": edge_j,
        "adj_edge_jaccard": adj,
    }


def summarise(rows: list[dict]) -> dict:
    """Aggregate per-dataset rows exactly as `metrics.summarise` does.

    Note the asymmetry, which decides how local validation must be weighted:
    edge and division Jaccard are **micro**-averaged (counts pooled, then divided),
    but the scoring term `adj_edge_jaccard` is a **weighted mean of per-dataset
    values**, weight `TP + FP + FN`. Recon §9: that makes the two `6bba_` test
    datasets ~95% of the leaderboard.
    """
    valid = [r for r in rows if r["edge_tp"] == r["edge_tp"]]
    if not valid:
        return {"n": 0, "edge_jaccard": float("nan"), "division_jaccard": float("nan"),
                "node_recall": float("nan"), "adj_edge_jaccard": float("nan"),
                "score": float("nan")}

    tp = sum(r["edge_tp"] for r in valid)
    fp = sum(r["edge_fp"] for r in valid)
    fn = sum(r["edge_fn"] for r in valid)
    dtp = sum(r["division_tp"] for r in valid)
    dfp = sum(r["division_fp"] for r in valid)
    dfn = sum(r["division_fn"] for r in valid)

    edge_j = tp / (tp + fp + fn) if (tp + fp + fn) else float("nan")
    has_div = (dtp + dfp + dfn) > 0
    div_j = dtp / (dtp + dfp + dfn) if has_div else float("nan")

    adj_rows = [r for r in valid if r["adj_edge_jaccard"] == r["adj_edge_jaccard"]]
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in adj_rows]
    total_w = sum(w)
    adj = (sum(r["adj_edge_jaccard"] * wi for r, wi in zip(adj_rows, w)) / total_w
           if total_w else float("nan"))

    nr = [r["node_recall"] for r in valid if r["node_recall"] == r["node_recall"]]
    score = adj + SCORE_DIVISION_WEIGHT * div_j if has_div else adj
    return {
        "n": len(valid), "edge_jaccard": edge_j, "division_jaccard": div_j,
        "division_tp": dtp, "division_fp": dfp, "division_fn": dfn,
        "node_recall": float(np.mean(nr)) if nr else float("nan"),
        "adj_edge_jaccard": adj, "n_adj": len(adj_rows), "score": score,
    }


__all__ = ["EdgeCounts", "match_nodes", "count_edges", "survivors", "per_sample", "summarise",
           "DEFAULT_SCALE", "MAX_DISTANCE"]
