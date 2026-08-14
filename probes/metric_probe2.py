"""Probe 2: what a *wrong* link actually costs, and whether frame gaps can be bridged."""
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
        "  CELLMOT_REPO=./kaggle-cell-tracking-competition python metric_probe2.py"
    )
sys.path.insert(0, str(REPO / "src"))

import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import evaluate, node_recall

SCALE = (1.625, 0.40625, 0.40625)


def build(nodes, edges):
    """nodes: list of (t,z,y,x); edges: list of (i,j) index pairs into nodes."""
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    ids = [g.add_node({"t": int(t), "z": float(z), "y": float(y), "x": float(x)}) for t, z, y, x in nodes]
    for i, j in edges:
        g.add_edge(ids[i], ids[j], {})
    return g


def report(name, pred, gt):
    er = evaluate(pred, gt, scale=SCALE)
    d = er.edge_tp + er.edge_fp + er.edge_fn
    print(f"{name:<52} TP/FP/FN = {er.edge_tp:>2}/{er.edge_fp:>2}/{er.edge_fn:>2}   J={er.edge_tp/max(1,d):.4f}"
          f"   div TP/FP/FN={er.division_tp}/{er.division_fp}/{er.division_fn}")


print("=" * 118)
print("A. One annotated track, 4 frames. What does a wrong outgoing link cost vs no link?")
print("=" * 118)
# GT: single track A at y=100, 4 frames
gt_nodes = [(0, 10, 100, 100), (1, 10, 102, 100), (2, 10, 104, 100), (3, 10, 106, 100)]
gt = build(gt_nodes, [(0, 1), (1, 2), (2, 3)])
print(f"GT: {gt.num_nodes()} nodes, {gt.num_edges()} edges (3 edges max)\n")

# a distractor cell far from any GT node (unannotated neighbour), present at t=3
DISTRACTOR = (3, 10, 400, 400)

# a1: perfect
report("A1. perfect (3/3 links)", build(gt_nodes, [(0, 1), (1, 2), (2, 3)]), gt)
# a2: last link simply omitted
report("A2. last link OMITTED", build(gt_nodes, [(0, 1), (1, 2)]), gt)
# a3: last link goes to an unannotated distractor instead
report("A3. last link -> UNANNOTATED distractor",
       build(gt_nodes + [DISTRACTOR], [(0, 1), (1, 2), (2, 4)]), gt)
# a4: both the correct link AND a spurious extra link (predicted division)
report("A4. correct link + extra link (fake division)",
       build(gt_nodes + [DISTRACTOR], [(0, 1), (1, 2), (2, 3), (2, 4)]), gt)

print()
print("=" * 118)
print("B. Missed detection in the middle: can a t -> t+2 edge bridge the gap?")
print("=" * 118)
# pred is missing the t=2 node entirely; try to link t=1 -> t=3 directly
pred_nodes = [(0, 10, 100, 100), (1, 10, 102, 100), (3, 10, 106, 100)]
report("B1. gap bridged with a t=1 -> t=3 edge", build(pred_nodes, [(0, 1), (1, 2)]), gt)
report("B2. gap left unbridged", build(pred_nodes, [(0, 1)]), gt)

print()
print("=" * 118)
print("C. Division handling: GT has a real division at t=1")
print("=" * 118)
# GT: parent at t=0,1 then splits into two daughters at t=2
gt_div_nodes = [(0, 10, 200, 200), (1, 10, 202, 200), (2, 10, 204, 190), (2, 10, 204, 210),
                (3, 10, 206, 188), (3, 10, 206, 212)]
gt_div = build(gt_div_nodes, [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)])
print(f"GT: {gt_div.num_nodes()} nodes, {gt_div.num_edges()} edges, 1 division\n")
report("C1. division predicted correctly", build(gt_div_nodes, [(0, 1), (1, 2), (1, 3), (2, 4), (3, 5)]), gt_div)
report("C2. division MISSED (one daughter dropped)", build(gt_div_nodes, [(0, 1), (1, 2), (2, 4), (3, 5)]), gt_div)
report("C3. division predicted one frame LATE",
       build(gt_div_nodes, [(0, 1), (1, 2), (2, 4), (2, 5)]), gt_div)
