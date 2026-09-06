"""Compare two finished arms' `submission.csv` without spending a submission slot.

    python tools/diff_arms.py claude-arm-lb941 claude-arm-gap44

`notes/65` §2: this is a kernels-only competition, so an arm can only be scored by a human
pressing Submit, and the daily five is the budget for *measured* arms. That makes it worth
knowing, before spending one, whether an arm's change reached the output at all.

**This is a mechanism check, not a score prediction** — the distinction `notes/64` was
written to enforce. It never says an arm is better. It says whether the arm is *different*,
and along which axis, which is precisely the failure mode `notes/60` recorded for
`close_gaps`: a parameter moved, a volume cap bound first, and the run reproduced its base
exactly while looking like a real experiment. An arm whose output is byte-identical to its
base has already been measured — it scores what the base scored — and should not be
submitted.

What is reported per arm, and as a delta:

* **nodes** — rows of `row_type == node`. This is `N_pred`, which enters the metric directly:
  `adj = edge_J * (1 - 0.1 * (N_pred - N_est) / N_est)`, floored at 0 and uncapped above, so
  predicting fewer nodes than the organisers' estimate pays a bonus.
* **edges** — rows of `row_type == edge`, the numerator and denominator of `edge_jaccard`.
* **forks** — nodes appearing as `source_id` on two or more edges: the divisions, whose
  Jaccard carries the metric's other 10%.
* **moved / added / dropped nodes** — keyed by `(dataset, t, node_id)`, so a change that only
  shifts coordinates is distinguished from one that changes the node set.

Both files come from `/kernels/output/download`, which is served by `www.kaggle.com`; the
signed URLs inside `/kernels/output` point at `www.kaggleusercontent.com` and this
container's proxy denies that host outright (`notes/65` §2).
"""
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K   # noqa: E402

CACHE = Path("/tmp/claude-0/arm_csv")


def load(slug: str) -> dict:
    """Read one arm's submission into node/edge maps, downloading it once."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{slug}.csv"
    if not path.exists():
        if K.kernel_output_file(slug, path) is None:
            raise SystemExit(f"{slug}: the run produced no submission.csv")
    nodes, edges = {}, defaultdict(set)
    with path.open() as f:
        header = next(f).rstrip("\n").split(",")
        col = {name: i for i, name in enumerate(header)}
        for line in f:
            r = line.rstrip("\n").split(",")
            kind = r[col["row_type"]]
            ds = r[col["dataset"]]
            if kind == "node":
                nodes[(ds, r[col["t"]], r[col["node_id"]])] = (
                    r[col["z"]], r[col["y"]], r[col["x"]])
            elif kind == "edge":
                edges[(ds, r[col["source_id"]])].add(r[col["target_id"]])
    forks = sum(1 for tgts in edges.values() if len(tgts) > 1)
    return {"nodes": nodes, "edges": edges, "forks": forks,
            "n_edges": sum(len(t) for t in edges.values()),
            "per_ds": Counter(k[0] for k in nodes)}


def main(a: str, b: str) -> int:
    A, B = load(a), load(b)
    ka, kb = set(A["nodes"]), set(B["nodes"])
    both = ka & kb
    moved = sum(1 for k in both if A["nodes"][k] != B["nodes"][k])

    print(f"{'':<22}{a[-24:]:>26}{b[-24:]:>26}{'delta':>12}")
    for label, x, y in (("nodes", len(ka), len(kb)),
                        ("edges", A["n_edges"], B["n_edges"]),
                        ("forks (divisions)", A["forks"], B["forks"]),
                        ("datasets", len(A["per_ds"]), len(B["per_ds"]))):
        print(f"{label:<22}{x:>26,}{y:>26,}{y - x:>+12,}")

    print(f"\nnode set:  shared {len(both):,}   only in {a}: {len(ka - kb):,}   "
          f"only in {b}: {len(kb - ka):,}")
    print(f"of the shared nodes, {moved:,} moved position "
          f"({100.0 * moved / max(len(both), 1):.2f}%)")

    if not (ka - kb) and not (kb - ka) and moved == 0 and A["n_edges"] == B["n_edges"]:
        print(f"\nIDENTICAL OUTPUT — {b} is a no-op against {a}. It will score exactly what\n"
              f"{a} scores; do not spend a submission slot on it. Check whether a volume cap\n"
              f"bound before the parameter did (notes/60).")
        return 0

    worst = sorted(B["per_ds"].keys() | A["per_ds"].keys(),
                   key=lambda d: -abs(B["per_ds"][d] - A["per_ds"][d]))[:5]
    print("\nlargest per-dataset node changes:")
    for d in worst:
        print(f"   {d:<24}{A['per_ds'][d]:>8,} -> {B['per_ds'][d]:>8,}"
              f"{B['per_ds'][d] - A['per_ds'][d]:>+8,}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
