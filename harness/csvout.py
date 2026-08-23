"""Stdlib-only submission writer, for the scored rerun where nothing can be installed.

`harness.submission` does the same job through polars and prints a nicer report, but it
is one more package that has to be present in an environment we do not control and
cannot `pip install` into. This module needs only `csv` and `numpy`, and it *streams*:
a ~200-dataset test set at the densities we predict is on the order of 8 million rows,
so nothing here ever holds the whole table.

The checks in `check_graph` matter as much as the writing. The scorer silently repairs
malformed graphs — it drops edges that do not span exactly t->t+1, de-duplicates
repeated source->target pairs, collapses merges, and truncates out-degree above 2. Each
one is points evaporating with no error and no leaderboard clue, so catch them here.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

# Column order from the official scripts/geffs_to_csv.py, with `id` prepended.
COLUMNS: tuple[str, ...] = (
    "id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


def check_graph(tracks, name: str = "", allow_divisions: bool = False) -> list[str]:
    """Everything the scorer would quietly fix. Empty list means clean."""
    problems: list[str] = []
    t, e = np.asarray(tracks.t), np.asarray(tracks.edges)
    n = len(t)

    if e.size:
        if e.min() < 0 or e.max() >= n:
            problems.append(f"{name}: edge endpoints outside [0, {n})")
            return problems  # everything below would index out of range
        dt = t[e[:, 1]] - t[e[:, 0]]
        bad = int((dt != 1).sum())
        if bad:
            problems.append(f"{name}: {bad:,} edges do not span exactly t->t+1 "
                            "(the scorer drops these)")
        if int((e[:, 0] == e[:, 1]).sum()):
            problems.append(f"{name}: self-loops present")
        uniq = np.unique(e, axis=0)
        if len(uniq) != len(e):
            problems.append(f"{name}: {len(e) - len(uniq):,} duplicate edges")
        # A cell has one predecessor; two incoming edges is a merge, which is not
        # biology and which the scorer collapses.
        _, in_counts = np.unique(e[:, 1], return_counts=True)
        if in_counts.max(initial=0) > 1:
            problems.append(f"{name}: {int((in_counts > 1).sum()):,} nodes have >1 "
                            "incoming edge (merge)")
        _, out_counts = np.unique(e[:, 0], return_counts=True)
        limit = 2 if allow_divisions else 1
        if out_counts.max(initial=0) > limit:
            problems.append(f"{name}: out-degree above {limit} on "
                            f"{int((out_counts > limit).sum()):,} nodes")

    zyx = np.asarray(tracks.zyx)
    if n and not np.isfinite(zyx).all():
        problems.append(f"{name}: non-finite coordinates")
    return problems


def graph_rows(tracks, name: str, start_id: int):
    """Node rows then edge rows for one dataset, as lists ready for `csv.writer`.

    Node ids are array indices, so the edge rows stay consistent with the node rows by
    construction. Coordinates are rounded to int — that is the submission space, and it
    round-trips exactly through the official `csv_to_geffs`.
    """
    t = np.asarray(tracks.t, dtype=np.int64)
    zyx = np.rint(np.asarray(tracks.zyx)).astype(np.int64)
    e = np.asarray(tracks.edges, dtype=np.int64).reshape(-1, 2)

    rid = start_id
    for i in range(len(t)):
        yield [rid, name, "node", i, int(t[i]),
               int(zyx[i, 0]), int(zyx[i, 1]), int(zyx[i, 2]), -1, -1]
        rid += 1
    for s, d in e:
        yield [rid, name, "edge", -1, -1, -1, -1, -1, int(s), int(d)]
        rid += 1


def write_submission(graphs, csv_path: Path | str, verbose: bool = True,
                     allow_divisions: bool = False) -> dict:
    """Stream `{dataset_name: Tracks}` (or an iterable of pairs) to a submission CSV.

    Accepts an iterable of `(name, tracks)` as well as a dict, so a caller can predict
    one dataset at a time and never hold more than one graph in memory.

    Returns a summary dict; `problems` is empty when every graph passed `check_graph`.

    ``allow_divisions`` must match what the predictor actually emits. Every arm in this
    project up to `notes/23` was fork-free, so False was right; a model that predicts
    divisions -- the public pack's does, hundreds per dataset -- would otherwise have every
    one of them reported as an out-degree violation, burying any real problem in noise.
    Divisions are worth 0.1 of the 1.1 maximum, so emitting them is the point, not a fault.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(graphs.items()) if hasattr(graphs, "items") else graphs

    rid, n_nodes, n_edges, names, problems = 0, 0, 0, [], []
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLUMNS)
        for name, tr in items:
            problems += check_graph(tr, name, allow_divisions=allow_divisions)
            for row in graph_rows(tr, name, rid):
                w.writerow(row)
            rid += tr.n_nodes + tr.n_edges
            n_nodes += tr.n_nodes
            n_edges += tr.n_edges
            names.append(name)
            if verbose:
                print(f"  {name:<24} {tr.n_nodes:>8,} nodes  {tr.n_edges:>8,} edges",
                      flush=True)

    out = {"rows": rid, "datasets": len(names), "names": names,
           "nodes": n_nodes, "edges": n_edges, "problems": problems,
           "path": str(csv_path)}
    if verbose:
        print(f"\nwrote {rid:,} rows for {len(names)} datasets -> {csv_path}")
        if problems:
            print(f"!! {len(problems)} problem(s) the scorer would silently repair:")
            for p in problems[:20]:
                print(f"   {p}")
        else:
            print("no malformed-graph problems found")
    return out


__all__ = ["COLUMNS", "check_graph", "graph_rows", "write_submission"]
