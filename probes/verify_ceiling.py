"""Verify that the numpy-only linking-ceiling computation agrees EXACTLY with the
official scorer.

When the predicted nodes ARE the ground-truth nodes, the bipartite distance matching is
identity (every pair is at distance 0, which is the unique optimum). Under that condition
the official edge rules collapse to pure set arithmetic on edge index pairs — so this is a
specialisation of the official metric, not a reimplementation of it. This script proves
the specialisation is exact by running both on the same graphs.
"""
import sys
from pathlib import Path

import os
REPO = Path(os.environ.get("CELLMOT_REPO", "./kaggle-cell-tracking-competition"))
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import polars as pl
import tracksdata as td
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from tracking_cellmot.metrics import evaluate

SCALE = (1.625, 0.40625, 0.40625)
MATCH_UM = 7.0


def link_frames(coords, scale, mode, radius_um):
    by_t = {}
    for i in np.argsort(coords[:, 0], kind="stable"):
        by_t.setdefault(int(coords[i, 0]), []).append(int(i))
    phys = coords[:, 1:] * np.asarray(scale)[None, :]
    edges = []
    for t in sorted(by_t):
        a, b = by_t.get(t), by_t.get(t + 1)
        if not a or not b:
            continue
        A, B = phys[a], phys[b]
        if mode == "greedy":
            d, j = cKDTree(B).query(A, k=1, distance_upper_bound=radius_um)
            pairs = [(i, int(j[i])) for i in range(len(a)) if np.isfinite(d[i])]
        else:
            D = cdist(A, B)
            ri, ci = linear_sum_assignment(D)
            pairs = [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= radius_um]
        for i, j in pairs:
            edges.append((a[i], b[j]))
    return edges


def numpy_counts(t, src, dst, pred_edges):
    """Exact edge TP/FP/FN under identity node matching — the notebook's computation."""
    n = len(t)
    out_deg = np.bincount(src, minlength=n)
    in_deg = np.bincount(dst, minlength=n)
    gt_set = set(zip(src.tolist(), dst.tolist()))

    # The scorer drops edges not spanning exactly t -> t+1 ...
    pred = {(i, j) for i, j in pred_edges if t[j] - t[i] == 1}
    # ... and caps out-degree at 2, keeping the lowest edge ids. We emit at most one
    # edge per source here, so the cap cannot bind; assert rather than assume.
    from collections import Counter
    assert max(Counter(i for i, _ in pred).values(), default=0) <= 2

    tp = len(pred & gt_set)
    valid = sum(1 for i, j in pred if out_deg[i] > 0 or in_deg[j] > 0)
    return tp, valid - tp, len(gt_set) - tp


def build_td(coords, edges):
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t": int(a), "z": float(b), "y": float(c), "x": float(d)}
                            for a, b, c, d in coords])
    if edges:
        g.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in edges])
    return g


rng = np.random.default_rng(0)
mismatches = 0
print(f"{'case':<34} {'official TP/FP/FN':>22} {'numpy TP/FP/FN':>22}  match")
print("-" * 88)

for trial, (n_tracks, T, spread, radius) in enumerate([
    (40, 6, 200.0, 25.0),     # sparse, easy
    (120, 6, 60.0, 25.0),     # dense, ambiguous
    (200, 5, 40.0, 15.0),     # very dense, tight radius
    (30, 8, 300.0, 5.0),      # radius so tight most links are dropped
]):
    start = rng.uniform([0, 0, 0], [20, spread, spread], size=(n_tracks, 3))
    vel = rng.normal(0, 2.5, size=(n_tracks, 3)) * np.array([0.2, 1.0, 1.0])
    coords, owner = [], []
    for t in range(T):
        for i in range(n_tracks):
            p = start[i] + vel[i] * t
            coords.append([t, *p])
            owner.append(i)
    coords = np.array(coords, float)
    owner = np.array(owner)

    gt_pairs = []
    for t in range(T - 1):
        a = np.where(coords[:, 0] == t)[0]
        b = np.where(coords[:, 0] == t + 1)[0]
        for i in a:
            gt_pairs.append((int(i), int(b[owner[b] == owner[i]][0])))
    src = np.array([p[0] for p in gt_pairs])
    dst = np.array([p[1] for p in gt_pairs])

    for mode in ("hungarian", "greedy"):
        pred_edges = link_frames(coords, SCALE, mode, radius)

        gt_graph = build_td(coords, gt_pairs)
        pred_graph = build_td(coords, pred_edges)
        er = evaluate(pred_graph, gt_graph, scale=SCALE, max_distance=MATCH_UM)
        official = (er.edge_tp, er.edge_fp, er.edge_fn)

        mine = numpy_counts(coords[:, 0], src, dst, pred_edges)
        ok = official == mine
        if not ok:
            mismatches += 1
        print(f"trial{trial} {mode:<10} r={radius:<5} n={n_tracks:<4} "
              f"{str(official):>22} {str(mine):>22}  {'OK' if ok else 'MISMATCH'}")

print("-" * 88)
if mismatches:
    print(f"{mismatches} MISMATCH(ES) — the specialisation is NOT exact, do not ship it")
    sys.exit(1)
print("numpy specialisation reproduces the official scorer exactly on every case")
