"""Empirical probes of the official competition metric.

Runs the real `tracking_cellmot.metrics.evaluate` on hand-built graphs to
verify (not assume) how the score responds to over-detection, free-floating
predictions, and the node-count adjustment.
"""
import os
import sys
from pathlib import Path

# Point at a clone of https://github.com/royerlab/kaggle-cell-tracking-competition
REPO = Path(os.environ.get("CELLMOT_REPO", "./kaggle-cell-tracking-competition"))
if not (REPO / "src" / "tracking_cellmot").is_dir():
    raise SystemExit(
        f"Official baseline repo not found at {REPO}.\n"
        "Clone it and set CELLMOT_REPO:\n"
        "  git clone https://github.com/royerlab/kaggle-cell-tracking-competition\n"
        "  CELLMOT_REPO=./kaggle-cell-tracking-competition python metric_probe.py"
    )
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import polars as pl
import tracksdata as td

from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall

SCALE = (1.625, 0.40625, 0.40625)  # z, y, x microns/px


def make_graph(tracks):
    """tracks: list of list of (t, z, y, x). Consecutive entries are linked."""
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    for track in tracks:
        prev = None
        for (t, z, y, x) in track:
            nid = g.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)})
            if prev is not None:
                g.add_edge(prev, nid, {})
            prev = nid
    return g


def line_track(t0, n, z, y, x, dz=0.0, dy=2.0, dx=2.0):
    return [(t0 + i, z + i * dz, y + i * dy, x + i * dx) for i in range(n)]


def report(name, pred, gt, n_total=None):
    er = evaluate(pred, gt, scale=SCALE)
    rec = node_recall(pred, gt)
    line = (f"{name:<38} edge TP/FP/FN = {er.edge_tp:>3}/{er.edge_fp:>3}/{er.edge_fn:>3}"
            f"  J={er.edge_tp / max(1, er.edge_tp + er.edge_fp + er.edge_fn):.4f}"
            f"  n_pred={er.num_pred_nodes:>3}  node_recall={rec:.3f}")
    if n_total is not None:
        m = per_sample_metrics(er, n_total, rec)
        line += f"  | n_total={n_total} ratio={m['total_node_ratio']:+.3f} ADJ_J={m['adj_edge_jaccard']:.4f}"
    print(line)
    return er


print("=" * 120)
print("GT: 3 annotated tracks, 6 frames each (sparse annotation of a much denser embryo)")
print("=" * 120)

gt_tracks = [
    line_track(0, 6, z=10, y=100, x=100),
    line_track(0, 6, z=10, y=300, x=300),
    line_track(0, 6, z=10, y=500, x=500),
]
gt = make_graph(gt_tracks)
print(f"GT: {gt.num_nodes()} nodes, {gt.num_edges()} edges\n")

# --- 1. perfect prediction on the annotated cells only -----------------------
report("1. perfect (annotated cells only)", make_graph(gt_tracks), make_graph(gt_tracks))

# --- 2. same, PLUS many extra cells nowhere near any GT cell -----------------
# This is the key question: does predicting unannotated cells cost anything?
for n_extra in (3, 30, 300):
    extra = [line_track(0, 6, z=30, y=1000 + 50 * k, x=1000 + 50 * k) for k in range(n_extra)]
    report(f"2. perfect + {n_extra:>3} unannotated tracks", make_graph(gt_tracks + extra), make_graph(gt_tracks))

print()
print("=" * 120)
print("Node-count adjustment: same predictions, varying the GT's estimated_number_of_nodes")
print("=" * 120)
# 18 GT nodes annotated, but the embryo really contains ~500 cells x 6 frames.
extra = [line_track(0, 6, z=30, y=1000 + 50 * k, x=1000 + 50 * k) for k in range(30)]
pred_tracks = gt_tracks + extra  # 33 tracks x 6 = 198 nodes
for n_total in (99, 198, 396, 1000):
    report(f"3. 198 pred nodes vs n_total={n_total}", make_graph(pred_tracks), make_graph(gt_tracks), n_total=n_total)

print()
print("=" * 120)
print("Cost of a WRONG link on an annotated cell (vs. no link at all)")
print("=" * 120)

# swap: link track0's t=2 node to track1's t=3 node (both annotated)
swapped = [
    line_track(0, 3, z=10, y=100, x=100),           # track0 frames 0-2
    line_track(3, 3, z=10, y=106, x=106),           # track0 frames 3-5 (disconnected)
    line_track(0, 6, z=10, y=300, x=300),
    line_track(0, 6, z=10, y=500, x=500),
]
report("4a. one link MISSING (broken track)", make_graph(swapped), make_graph(gt_tracks))

# Predict a link from an annotated cell to the WRONG annotated cell
wrong = [
    [(0, 10, 100, 100), (1, 10, 102, 102), (2, 10, 104, 104), (3, 10, 306, 306)],  # jumps to track1
    line_track(0, 6, z=10, y=300, x=300),
    line_track(0, 6, z=10, y=500, x=500),
]
report("4b. one link WRONG (jumps tracks)", make_graph(wrong), make_graph(gt_tracks))

print()
print("=" * 120)
print("Over-detection NEAR an annotated cell (duplicate detections within the 7um match radius)")
print("=" * 120)
# duplicate each GT track, offset by 2px in y (~0.8um) -> inside the 7um radius
dupes = [line_track(0, 6, z=10, y=100 + 2, x=100), line_track(0, 6, z=10, y=300 + 2, x=300)]
report("5. perfect + near-duplicate detections", make_graph(gt_tracks + dupes), make_graph(gt_tracks))

print()
print("=" * 120)
print("Recall vs precision: predicting only SOME of the annotated tracks")
print("=" * 120)
for k in (1, 2, 3):
    report(f"6. only {k}/3 annotated tracks predicted", make_graph(gt_tracks[:k]), make_graph(gt_tracks), n_total=18)
