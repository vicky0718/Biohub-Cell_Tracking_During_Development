"""Tests for the harness, runnable without competition data.

Builds synthetic ground truth on disk as real .geff files, so the harness is
exercised through its actual I/O path and the official scorer.

    CELLMOT_REPO=/path/to/kaggle-cell-tracking-competition python tests/test_harness.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness import Harness, build_submission, gate, validate_submission  # noqa: E402
from harness.scorer import DEFAULT_SCALE  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAILURES.append(f"{name} {detail}")


def make_graph(coords, edges):
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                            for t, z, y, x in coords])
    if edges:
        g.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in edges])
    return g, ids


def synth(n_tracks=12, T=6, seed=0):
    """Well-separated tracks so nearest-neighbour linking is unambiguous."""
    rng = np.random.default_rng(seed)
    start = rng.uniform([0, 0, 0], [30, 800, 800], size=(n_tracks, 3))
    vel = rng.normal(0, 1.0, size=(n_tracks, 3)) * np.array([0.2, 1.0, 1.0])
    coords, owner = [], []
    for t in range(T):
        for i in range(n_tracks):
            p = start[i] + vel[i] * t
            coords.append([t, p[0], p[1], p[2]])
            owner.append(i)
    coords = np.array(coords, float)
    owner = np.array(owner)
    edges = []
    for t in range(T - 1):
        a = np.where(coords[:, 0] == t)[0]
        b = np.where(coords[:, 0] == t + 1)[0]
        for i in a:
            edges.append((int(i), int(b[owner[b] == owner[i]][0])))
    return coords, edges


def write_dataset(dirpath: Path, name: str, coords, edges):
    g, _ = make_graph(coords, edges)
    out = dirpath / f"{name}.geff"
    if out.exists():
        shutil.rmtree(out) if out.is_dir() else out.unlink()
    g.to_geff(str(out))
    return out


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="harness_test_"))
    data = tmp / "train"
    data.mkdir(parents=True)

    names = [f"ds_{i}" for i in range(6)]
    truth = {}
    for i, n in enumerate(names):
        coords, edges = synth(seed=i)
        truth[n] = (coords, edges)
        write_dataset(data, n, coords, edges)

    # No .zarr files here, so scale must fall back to DEFAULT_SCALE.
    # No estimated_number_of_nodes either, so pass an explicit budget.
    budget = {n: float(len(truth[n][0])) for n in names}
    h = Harness(data_dir=data, cache_dir=tmp / "cache", n_total_override=budget)

    print("\n[discovery]")
    check("finds all datasets", h.dataset_names() == names, str(h.dataset_names()))
    folds = {h.fold_of(n) for n in names}
    check("assigns folds without official splits", len(folds) > 1, f"folds={sorted(folds)}")
    h2 = Harness(data_dir=data, n_total_override=budget)
    check("fold assignment is deterministic across instances",
          all(h.fold_of(n) == h2.fold_of(n) for n in names))

    print("\n[scoring]")

    def perfect(name, _dir):
        c, e = truth[name]
        return make_graph(c, e)[0]

    res = h.evaluate(perfect, arm="perfect", verbose=False)
    check("perfect prediction scores 1.0", abs(res.summary["edge_jaccard"] - 1.0) < 1e-9,
          f"edge_J={res.summary['edge_jaccard']:.6f}")
    check("adjusted score computed when budget known",
          abs(res.summary["adj_edge_jaccard"] - 1.0) < 1e-9,
          f"adj={res.summary['adj_edge_jaccard']:.6f}")

    def drop_last_link(name, _dir):
        c, e = truth[name]
        return make_graph(c, e[:-3])[0]

    worse = h.evaluate(drop_last_link, arm="dropped", verbose=False)
    check("dropping links lowers the score", worse.score < res.score,
          f"{worse.score:.4f} < {res.score:.4f}")

    print("\n[node budget]")

    def over_detect(name, _dir):
        """Perfect links plus far-away fictional cells — free per the metric probes,
        except for the node-count multiplier."""
        c, e = truth[name]
        extra = np.array([[t, 5.0, 5000.0 + 40 * k, 5000.0] for k in range(20) for t in range(6)])
        coords = np.vstack([c, extra])
        return make_graph(coords, e)[0]

    over = h.evaluate(over_detect, arm="over", verbose=False)
    check("extra unmatched nodes do not create edge FPs",
          abs(over.summary["edge_jaccard"] - 1.0) < 1e-9,
          f"edge_J={over.summary['edge_jaccard']:.6f}")
    check("but they do trigger the node-budget penalty",
          over.summary["adj_edge_jaccard"] < res.summary["adj_edge_jaccard"],
          f"adj {over.summary['adj_edge_jaccard']:.4f} < {res.summary['adj_edge_jaccard']:.4f}")

    print("\n[gate]")
    v_better = gate(worse, res)
    check("promotes a uniform improvement", v_better.promote, str(v_better.pooled_delta))
    v_worse = gate(res, worse)
    check("rejects a uniform regression", not v_worse.promote)

    # A gain in one fold paid for by a loss in another must be REJECTED even when
    # the pooled number improves — that is the entire point of the gate.
    # Controlled fold map so this tests the gate, not the fold assignment:
    # fold 0 holds one dataset that gets WORSE, fold 1 holds three that get BETTER,
    # so the pooled score rises while fold 0 regresses.
    from harness.harness import Result
    good, bad = res.rows, worse.rows
    fmap = {"a": 0, "b": 1, "c": 1, "d": 1}
    b_rows = {"a": good[names[0]], "b": bad[names[1]], "c": bad[names[2]], "d": bad[names[3]]}
    a_rows = {"a": bad[names[0]], "b": good[names[1]], "c": good[names[2]], "d": good[names[3]]}
    v_mixed = gate(Result("base", b_rows, fmap), Result("mixed", a_rows, fmap))
    check("rejects pooled gain that regresses a fold",
          not v_mixed.promote and v_mixed.pooled_delta > 0
          and any(d < 0 for d in v_mixed.fold_deltas.values()),
          f"pooled={v_mixed.pooled_delta:+.4f} folds={ {k: round(x, 4) for k, x in v_mixed.fold_deltas.items()} }")

    print("\n[caching]")
    calls = {"n": 0}

    def counted(name, _dir):
        calls["n"] += 1
        return perfect(name, _dir)

    h.evaluate(counted, arm="cached_arm", verbose=False)
    first = calls["n"]
    h.evaluate(counted, arm="cached_arm", verbose=False)
    check("second run is served from cache", calls["n"] == first, f"{first} then {calls['n']}")

    print("\n[submission]")
    graphs = {n: perfect(n, data) for n in names}
    csv = build_submission(graphs, tmp / "submission.csv")
    probs = validate_submission(csv, expected_datasets=names, node_budget=budget)
    check("clean submission validates", probs == [], str(probs))

    # now a deliberately broken one: a t -> t+2 edge the scorer would silently drop
    c, e = truth[names[0]]
    bad_edges = list(e) + [(0, int(np.where(c[:, 0] == 2)[0][0]))]
    broken = dict(graphs)
    broken[names[0]] = make_graph(c, bad_edges)[0]
    csv2 = build_submission(broken, tmp / "broken.csv")
    probs2 = validate_submission(csv2, expected_datasets=names, node_budget=budget)
    check("validator catches non-consecutive edges",
          any("t->t+1" in p for p in probs2), str(probs2))

    missing = {k: v for k, v in graphs.items() if k != names[-1]}
    csv3 = build_submission(missing, tmp / "missing.csv")
    probs3 = validate_submission(csv3, expected_datasets=names, node_budget=budget)
    check("validator catches a missing test dataset",
          any("no predictions" in p for p in probs3), str(probs3))

    shutil.rmtree(tmp, ignore_errors=True)
    print("\n" + "=" * 60)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all harness tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
