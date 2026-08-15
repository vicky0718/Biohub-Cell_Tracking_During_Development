"""Prove `harness/purescore.py` agrees EXACTLY with the official scorer.

`purescore` exists so we can score on Kaggle, where `tracksdata` cannot be installed
(numpy pin). That is only useful if it is the same computation — a scorer that is
"close" is worse than no scorer, because it moves the decision threshold silently.

So this fuzzes both implementations over graphs built to hit every repair path the
official metric performs: duplicate edges, merges onto one GT edge, out-degree above
two, backward and skipping edges, frames with no GT at all, detections just inside and
just outside the 7 µm match radius, and node counts that make the bipartite matching
non-square (which is what triggers the solver's fallback). Agreement is required on
TP/FP/FN exactly, not approximately.

    CELLMOT_REPO=/path/to/kaggle-cell-tracking-competition python probes/verify_purescore.py
"""
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("CELLMOT_REPO", "./kaggle-cell-tracking-competition"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import polars as pl
import tracksdata as td
from tracking_cellmot.metrics import evaluate, per_sample_metrics, summarise

from harness import purescore

SCALE = (1.625, 0.40625, 0.40625)
FAILURES = []


def build_graph(t, zyx, edges):
    """A tracksdata graph whose node ids are 0..N-1 in the given order."""
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, 0.0)
    ids = [g.add_node({"t": int(t[i]), "z": float(zyx[i, 0]),
                       "y": float(zyx[i, 1]), "x": float(zyx[i, 2])})
           for i in range(len(t))]
    for s, d in edges:
        g.add_edge(ids[int(s)], ids[int(d)], {})
    return g, ids


def make_case(rng, n_tracks, T, jitter_um, noise_nodes, mangle):
    """A GT track set plus a perturbed prediction of it.

    `jitter_um` pushes detections around the 7 µm cliff; `mangle` switches on the
    pathological edges the scorer has repair rules for.
    """
    scale = np.array(SCALE)
    pos = np.column_stack([rng.uniform(5, 55, n_tracks),
                           rng.uniform(20, 230, n_tracks),
                           rng.uniform(20, 230, n_tracks)])
    vel = rng.normal(0, 1.5, (n_tracks, 3)) * np.array([0.3, 1.0, 1.0])

    gt_t, gt_zyx, gt_edges, prev = [], [], [], [None] * n_tracks
    for t in range(T):
        for i in range(n_tracks):
            gt_t.append(t)
            gt_zyx.append(pos[i] + vel[i] * t)
            nid = len(gt_t) - 1
            if prev[i] is not None:
                gt_edges.append((prev[i], nid))
            prev[i] = nid
    gt_t = np.array(gt_t)
    gt_zyx = np.array(gt_zyx)

    # prediction: jittered copies of a random subset, plus pure noise nodes
    keep = rng.random(len(gt_t)) < 0.85
    idx = np.flatnonzero(keep)
    jitter = rng.normal(0, jitter_um, (len(idx), 3)) / scale[None, :]
    p_t = list(gt_t[idx])
    p_zyx = list(gt_zyx[idx] + jitter)
    for _ in range(noise_nodes):
        p_t.append(int(rng.integers(0, T)))
        p_zyx.append(np.array([rng.uniform(0, 60), rng.uniform(0, 250), rng.uniform(0, 250)]))
    p_t = np.array(p_t)
    p_zyx = np.array(p_zyx)

    # predicted edges: nearest neighbour in the next frame, then deliberate damage
    order = np.argsort(p_t, kind="stable")
    by_t = {}
    for i in order:
        by_t.setdefault(int(p_t[i]), []).append(int(i))
    phys = p_zyx * scale[None, :]
    p_edges = []
    for t in sorted(by_t):
        a, b = by_t.get(t, []), by_t.get(t + 1, [])
        if not a or not b:
            continue
        D = np.linalg.norm(phys[a][:, None, :] - phys[b][None, :, :], axis=2)
        for ii in range(len(a)):
            jj = int(np.argmin(D[ii]))
            if D[ii, jj] <= 12.0:
                p_edges.append((a[ii], b[jj]))

    if mangle and p_edges:
        e = np.array(p_edges)
        extra = []
        extra += [tuple(x) for x in e[rng.random(len(e)) < 0.10]]          # duplicates
        for s, d in e[rng.random(len(e)) < 0.15]:                          # extra children
            cands = by_t.get(int(p_t[s]) + 1, [])
            if len(cands) > 2:
                extra += [(int(s), int(c)) for c in rng.choice(cands, 2, replace=False)]
        for s, d in e[rng.random(len(e)) < 0.08]:                          # skip / backward
            far = [i for i in range(len(p_t)) if p_t[i] == p_t[s] + 2]
            if far:
                extra.append((int(s), int(far[0])))
            extra.append((int(d), int(s)))
        p_edges = p_edges + extra
        rng.shuffle(p_edges)

    return gt_t, gt_zyx, np.array(gt_edges), p_t, p_zyx, np.array(p_edges)


def check(name, official, pure):
    ok = official == pure
    print(f"  {'OK  ' if ok else 'FAIL'}  {name:<44} official {official}   pure {pure}")
    if not ok:
        FAILURES.append(name)


def main():
    rng = np.random.default_rng(0)
    print("fuzzing purescore against the official scorer\n")

    cases = [
        ("clean, tight jitter",        dict(n_tracks=30, T=6, jitter_um=0.5, noise_nodes=0,   mangle=False)),
        ("jitter across the 7um cliff", dict(n_tracks=30, T=6, jitter_um=4.0, noise_nodes=0,   mangle=False)),
        ("noise nodes (FP on unannotated)", dict(n_tracks=25, T=6, jitter_um=1.0, noise_nodes=120, mangle=False)),
        ("mangled edges (dup/merge/outdeg/skip)", dict(n_tracks=25, T=6, jitter_um=1.0, noise_nodes=40, mangle=True)),
        ("dense + mangled",            dict(n_tracks=60, T=5, jitter_um=2.5, noise_nodes=200, mangle=True)),
        ("sparse GT, many detections", dict(n_tracks=6,  T=8, jitter_um=1.5, noise_nodes=300, mangle=True)),
        ("heavy jitter, everything on", dict(n_tracks=40, T=7, jitter_um=6.0, noise_nodes=150, mangle=True)),
    ]

    rows_off, rows_pure = [], []
    for label, kw in cases:
        gt_t, gt_zyx, gt_e, p_t, p_zyx, p_e = make_case(rng, **kw)
        gt_g, _ = build_graph(gt_t, gt_zyx, gt_e)
        p_g, _ = build_graph(p_t, p_zyx, p_e)

        er = evaluate(p_g, gt_g, scale=SCALE, max_distance=7.0)
        pc = purescore.count_edges(p_t, p_zyx, p_e, gt_t, gt_zyx, gt_e, scale=SCALE)

        check(f"{label}  [n_pred={len(p_t)}, edges={len(p_e)}]",
              (er.edge_tp, er.edge_fp, er.edge_fn), (pc.tp, pc.fp, pc.fn))

        n_total = float(len(gt_t)) * 3.0
        from tracking_cellmot.metrics import node_recall as official_node_recall
        rows_off.append(per_sample_metrics(er, n_total, official_node_recall(p_g, gt_g)))
        rows_pure.append(purescore.per_sample(pc, n_total, len(gt_t), n_gt_divisions=0))

    print("\n--- per-sample derived columns ---")
    for i, (a, b) in enumerate(zip(rows_off, rows_pure)):
        for col in ("edge_jaccard", "total_node_ratio", "adj_edge_jaccard", "node_recall"):
            same = abs(a[col] - b[col]) < 1e-9
            if not same:
                FAILURES.append(f"case {i} {col}")
            print(f"  {'OK  ' if same else 'FAIL'}  case {i} {col:<18} "
                  f"official {a[col]:.10f}  pure {b[col]:.10f}")

    print("\n--- aggregation ---")
    so, sp_ = summarise(rows_off), purescore.summarise(rows_pure)
    for col in ("edge_jaccard", "adj_edge_jaccard", "score"):
        same = abs(so[col] - sp_[col]) < 1e-9
        if not same:
            FAILURES.append(f"summarise {col}")
        print(f"  {'OK  ' if same else 'FAIL'}  {col:<18} official {so[col]:.10f}  pure {sp_[col]:.10f}")

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} DISAGREEMENT(S) — purescore is NOT safe to score with:")
        for f in FAILURES:
            print("  ", f)
        return 1
    print("purescore reproduces the official scorer exactly on every case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
