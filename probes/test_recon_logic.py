"""Dry-run the recon notebook's riskiest logic (cells 5 and 7) on synthetic data.

We cannot touch the competition data from here, but the tracksdata/scorer API calls
are the same, so this catches API mistakes before the notebook is run on Kaggle.
Extracts the real cell sources from the built .ipynb so we test what ships.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import os

REPO = Path(os.environ.get("CELLMOT_REPO", "./kaggle-cell-tracking-competition"))
if not (REPO / "src" / "tracking_cellmot").is_dir():
    raise SystemExit(
        f"Official baseline repo not found at {REPO}. Clone it and set CELLMOT_REPO:\n"
        "  git clone https://github.com/royerlab/kaggle-cell-tracking-competition"
    )
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import polars as pl
import tracksdata as td
from scipy.spatial import cKDTree
from scipy.optimize import linear_sum_assignment
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise

K = td.DEFAULT_ATTR_KEYS
SCALE = (1.625, 0.40625, 0.40625)
MATCH_UM = 7.0

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "01_recon.ipynb"
NB = json.loads(NB_PATH.read_text())
sources = ["".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"]

# Pull the two function definitions verbatim out of the notebook and exec them here,
# so this test exercises the shipped code rather than a copy.
ceiling_src = next(s for s in sources if "def link_nearest" in s)
<<<<<<< HEAD
# Everything before the driver section is the reusable function block.
fn_src = ceiling_src.split("import time")[0]
assert "def link_nearest" in fn_src and "def _link_frame" in fn_src, \
    "notebook layout changed — the function block no longer precedes 'import time'"
fn_src = fn_src.replace(
    "from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise", ""
)
ns = {"np": np, "pl": pl, "td": td, "linear_sum_assignment": linear_sum_assignment,
      "cKDTree": cKDTree}
exec(fn_src, ns)
build_graph, link_nearest = ns["build_graph"], ns["link_nearest"]
print(f"extracted build_graph + link_nearest from the notebook "
      f"(sparse matcher available: {ns.get('_sparse_lsa') is not None})\n")
=======
fn_src = ceiling_src.split("results = {}")[0]
fn_src = fn_src.replace(
    "from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise", ""
)
ns = {"np": np, "pl": pl, "td": td, "linear_sum_assignment": linear_sum_assignment}
exec(fn_src, ns)
build_graph, link_nearest = ns["build_graph"], ns["link_nearest"]
print("extracted build_graph + link_nearest from the notebook\n")
>>>>>>> 910b533ef98b4bffb5e96f3c42f5228827f76339

# --- synthetic "ground truth": 40 cells drifting over 8 frames ----------------
rng = np.random.default_rng(0)
T, N = 8, 250
start = rng.uniform([0, 0, 0], [25, 120, 120], size=(N, 3))
vel = rng.normal(0, 6.0, size=(N, 3)) * np.array([0.3, 1.0, 1.0])

coords, owner = [], []
for t in range(T):
    for i in range(N):
        p = start[i] + vel[i] * t + rng.normal(0, 0.15, 3)
        coords.append([t, p[0], p[1], p[2]])
        owner.append(i)
coords = np.array(coords, dtype=float)
owner = np.array(owner)

true_pairs = []  # index pairs into `coords`
for t in range(T - 1):
    a = np.where(coords[:, 0] == t)[0]
    b = np.where(coords[:, 0] == t + 1)[0]
    for i in a:
        true_pairs.append((int(i), int(b[owner[b] == owner[i]][0])))


def make_gt():
    g, ids = build_graph(coords)
    g.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in true_pairs])
    return g


gt = make_gt()
print(f"synthetic GT: {gt.num_nodes()} nodes, {gt.num_edges()} edges")

for mode in ("hungarian", "greedy"):
    pred, ids = build_graph(coords)
<<<<<<< HEAD
    e = link_nearest(coords, SCALE, mode=mode)
=======
    e = link_nearest(coords, SCALE, max_um=None, mode=mode)
>>>>>>> 910b533ef98b4bffb5e96f3c42f5228827f76339
    pred.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in e])

    gt2 = make_gt()
    er = evaluate(pred, gt2, scale=SCALE, max_distance=MATCH_UM)
    rec = node_recall(pred, gt2)
    row = per_sample_metrics(er, float(gt2.num_nodes()), rec)
    s = summarise([row])
    print(f"[{mode:>9}] {len(e):>4} links  TP/FP/FN={er.edge_tp}/{er.edge_fp}/{er.edge_fn} "
          f"J={row['edge_jaccard']:.4f}  adj={row['adj_edge_jaccard']:.4f}  score={s['score']:.4f}")

# --- cell 5 logic: is the nearest neighbour the true successor? ---------------
rank_hist = Counter()
ok = tot = 0
for t in range(T - 1):
    a = np.where(coords[:, 0] == t)[0]
    b = np.where(coords[:, 0] == t + 1)[0]
    P = coords[a][:, 1:] * np.asarray(SCALE)
    Q = coords[b][:, 1:] * np.asarray(SCALE)
    tree = cKDTree(Q)
    kk = min(5, len(Q))
    _, idx = tree.query(P, k=kk)
    idx = np.asarray(idx).reshape(len(P), -1)
    for row_i, i in enumerate(a):
        true_j = b[owner[b] == owner[i]][0]
        tot += 1
        cand = b[idx[row_i]]
        hit = np.where(cand == true_j)[0]
        rank_hist[int(hit[0]) + 1 if len(hit) else ">5"] += 1
        if len(hit) and hit[0] == 0:
            ok += 1
print(f"\nNN-is-true-link: {ok}/{tot} = {ok/tot:.2%}   rank histogram: {dict(rank_hist)}")
print("\nOK - notebook logic runs end to end against the official scorer.")
<<<<<<< HEAD

# --- force the SPARSE path -----------------------------------------------------
# The dense branch is what a small frame takes; on real data a frame may hold
# thousands of cells and fall through to sparse matching. That branch is easy to
# ship broken because nothing small ever executes it, so exercise it explicitly.
print("\n[sparse path]")
saved_cap = ns["DENSE_CAP"]
try:
    ns["DENSE_CAP"] = 1          # every frame now exceeds the cap
    e_sparse = link_nearest(coords, SCALE, mode="hungarian")
    ns["DENSE_CAP"] = saved_cap
    e_dense = link_nearest(coords, SCALE, mode="hungarian")

    pred_s, ids_s = build_graph(coords)
    pred_s.bulk_add_edges([{"source_id": ids_s[i], "target_id": ids_s[j]} for i, j in e_sparse])
    er_s = evaluate(pred_s, make_gt(), scale=SCALE, max_distance=MATCH_UM)
    j_s = er_s.edge_tp / max(1, er_s.edge_tp + er_s.edge_fp + er_s.edge_fn)

    print(f"  sparse: {len(e_sparse):>5} links  J={j_s:.4f}")
    print(f"  dense:  {len(e_dense):>5} links")
    agree = set(e_sparse) == set(e_dense)
    print(f"  {'PASS' if e_sparse else 'FAIL'}  sparse path produces links")
    print(f"  {'PASS' if abs(len(e_sparse) - len(e_dense)) <= 0.02 * len(e_dense) else 'WARN'}"
          f"  link counts agree within 2%  (identical sets: {agree})")
finally:
    ns["DENSE_CAP"] = saved_cap
=======
>>>>>>> 910b533ef98b4bffb5e96f3c42f5228827f76339
