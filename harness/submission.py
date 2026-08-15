"""Build and validate the submission CSV.

The scorer silently repairs several classes of malformed graph — it drops edges
that don't span exactly t→t+1, de-duplicates repeated source→target pairs,
collapses merges, and truncates out-degree above 2 by edge id. Silently. Every
one of those is points evaporating with no error message and no leaderboard clue.

`validate_submission` surfaces each of them before you spend a submission slot.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .tracks import Tracks

# Column order from the official scripts/geffs_to_csv.py; `id` is prepended.
COLUMNS: tuple[str, ...] = (
    "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id",
)


def graph_to_rows(graph, name: str) -> pl.DataFrame:
    """Flatten one prediction into node rows then edge rows, matching the official schema.

    `graph` is `Tracks`, or a `tracksdata` graph (converted). Node ids are the array
    indices, so the edge rows stay consistent with the node rows by construction.

    Coordinates are rounded to int, which is the submission space — so this
    round-trips exactly through csv_to_geffs.
    """
    tr = graph if isinstance(graph, Tracks) else Tracks.from_tracksdata(graph)
    n, m = tr.n_nodes, tr.n_edges
    nodes = pl.DataFrame({
        "dataset": [name] * n,
        "row_type": ["node"] * n,
        "node_id": np.arange(n, dtype=np.int64),
        "t": tr.t.astype(np.int64),
        "z": np.rint(tr.zyx[:, 0]).astype(np.int64),
        "y": np.rint(tr.zyx[:, 1]).astype(np.int64),
        "x": np.rint(tr.zyx[:, 2]).astype(np.int64),
        "source_id": np.full(n, -1, dtype=np.int64),
        "target_id": np.full(n, -1, dtype=np.int64),
    })
    edges = pl.DataFrame({
        "dataset": [name] * m,
        "row_type": ["edge"] * m,
        "node_id": np.full(m, -1, dtype=np.int64),
        "t": np.full(m, -1, dtype=np.int64),
        "z": np.full(m, -1, dtype=np.int64),
        "y": np.full(m, -1, dtype=np.int64),
        "x": np.full(m, -1, dtype=np.int64),
        "source_id": tr.edges[:, 0].astype(np.int64),
        "target_id": tr.edges[:, 1].astype(np.int64),
    })
    return pl.concat([nodes.select(COLUMNS), edges.select(COLUMNS)])


def build_submission(graphs: dict, csv_path: Path | str) -> Path:
    """Write one submission CSV from {dataset_name: graph}."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    frames = [graph_to_rows(g, name) for name, g in sorted(graphs.items())]
    table = (pl.concat(frames) if frames
             else pl.DataFrame(schema={c: (pl.String if c in ("dataset", "row_type") else pl.Int64)
                                       for c in COLUMNS}))
    table = table.with_row_index("id")
    table.write_csv(csv_path)
    print(f"wrote {table.height:,} rows for {len(graphs)} datasets -> {csv_path}")
    return csv_path


def validate_submission(
    csv_path: Path | str,
    expected_datasets: list[str] | None = None,
    node_budget: dict[str, float] | None = None,
) -> list[str]:
    """Check a submission for the failure modes the scorer handles silently.

    Returns a list of problem strings — empty means clean. Prints a readable report.
    """
    csv_path = Path(csv_path)
    df = pl.read_csv(csv_path)
    problems: list[str] = []
    notes: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"FATAL: missing columns {missing}")
        _report(problems, notes)
        return problems

    bad_types = set(df["row_type"].unique()) - {"node", "edge"}
    if bad_types:
        problems.append(f"FATAL: unexpected row_type values {sorted(bad_types)}")

    datasets = sorted(df["dataset"].unique())
    notes.append(f"{len(datasets)} datasets, {df.height:,} rows total")

    if expected_datasets is not None:
        missing_ds = sorted(set(expected_datasets) - set(datasets))
        extra_ds = sorted(set(datasets) - set(expected_datasets))
        if missing_ds:
            problems.append(f"FATAL: no predictions for {len(missing_ds)} test datasets: {missing_ds}")
        if extra_ds:
            problems.append(f"unexpected datasets not in the test set: {extra_ds}")

    for name in datasets:
        g = df.filter(pl.col("dataset") == name)
        nodes = g.filter(pl.col("row_type") == "node")
        edges = g.filter(pl.col("row_type") == "edge")

        if nodes.height == 0:
            problems.append(f"{name}: no node rows")
            continue
        if edges.height == 0:
            problems.append(f"{name}: no edge rows — scores 0, every GT edge is an FN")
            continue

        # node ids must be unique within a dataset, or edges are ambiguous
        n_dupe = nodes.height - nodes["node_id"].n_unique()
        if n_dupe:
            problems.append(f"{name}: {n_dupe} duplicate node_id values")

        ids = set(nodes["node_id"].to_list())
        dangling = sum(1 for s, t in zip(edges["source_id"].to_list(), edges["target_id"].to_list())
                       if s not in ids or t not in ids)
        if dangling:
            problems.append(f"{name}: {dangling} edges reference a missing node_id")

        t_of = dict(zip(nodes["node_id"].to_list(), nodes["t"].to_list()))
        dt = [t_of.get(t_) - t_of.get(s_)
              for s_, t_ in zip(edges["source_id"].to_list(), edges["target_id"].to_list())
              if s_ in t_of and t_ in t_of]
        bad_dt = sum(1 for d in dt if d != 1)
        if bad_dt:
            problems.append(
                f"{name}: {bad_dt}/{len(dt)} edges do not span t->t+1 — the scorer DROPS "
                "these silently (a t->t+2 'gap bridge' scores exactly the same as no edge)")

        pairs = list(zip(edges["source_id"].to_list(), edges["target_id"].to_list()))
        n_dupe_edges = len(pairs) - len(set(pairs))
        if n_dupe_edges:
            problems.append(f"{name}: {n_dupe_edges} duplicate source->target pairs (scorer de-dupes)")

        outdeg: dict[int, int] = {}
        indeg: dict[int, int] = {}
        for s_, t_ in pairs:
            outdeg[s_] = outdeg.get(s_, 0) + 1
            indeg[t_] = indeg.get(t_, 0) + 1
        over_out = sum(1 for v in outdeg.values() if v > 2)
        if over_out:
            problems.append(
                f"{name}: {over_out} nodes have out-degree > 2 — the scorer keeps only the "
                "two lowest edge ids and drops the rest, so which links survive is arbitrary")
        over_in = sum(1 for v in indeg.values() if v > 1)
        if over_in:
            notes.append(
                f"{name}: {over_in} nodes have >1 parent (a merge). Biologically invalid and "
                "the scorer collapses them; usually a linking bug worth fixing")

        n_div = sum(1 for v in outdeg.values() if v == 2)
        budget_note = ""
        if node_budget and name in node_budget and node_budget[name] > 0:
            ratio = (nodes.height - node_budget[name]) / node_budget[name]
            mult = max(0.0, 1 - 0.1 * ratio)
            budget_note = f"  budget ratio={ratio:+.2f} -> adj multiplier x{mult:.3f}"
        notes.append(f"{name}: {nodes.height:,} nodes, {edges.height:,} edges, "
                     f"{n_div:,} divisions{budget_note}")

    _report(problems, notes)
    return problems


def _report(problems: list[str], notes: list[str]) -> None:
    print("=== submission validation ===")
    for n in notes:
        print(f"  {n}")
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  !! {p}")
    else:
        print("\nno problems found")


__all__ = ["build_submission", "validate_submission", "graph_to_rows", "COLUMNS"]
