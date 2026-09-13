"""Score already-produced submissions against the training ground truth. CPU, minutes.

    python notebooks/_mk_claude_score.py claude-eval-ttasec claude-eval-ttaz16

`_mk_claude_eval.py` runs inference *and* scoring in one kernel, which is right for a new
arm but wrong for fixing the scorer: the two eval runs cost ~50 GPU-minutes each and their
scoring died on one line --

    ValueError: node attribute key 'z' not found in existing keys: '['t']'
    Initialize with `graph.add_node_attr_key(key, default_value)`

-- because a fresh `IndexedRXGraph` declares only `t`. The wrapper around the scoring block
did its job: both submissions survived, so the fix must not cost the inference again.

This notebook mounts those kernels' **outputs** instead of recomputing them, so a scoring
change costs two minutes of CPU rather than two hours of GPU. It carries the official metric
modules inline (31 KB of pure Python needing only `polars` and `tracksdata`) rather than
re-materialising the support pack, which exists to ship model weights this does not need.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import claude_kaggle_api as K          # noqa: E402

HERE = Path(__file__).resolve().parent
PACK = Path("/tmp/pack")

NB = '''import json, subprocess, sys, traceback
from pathlib import Path

# Install from the support pack's OFFLINE WHEELS, with --no-deps, which is what every arm
# notebook does and what its own log explains:
#
#   "Dependency resolver is disabled with --no-deps to avoid replacing Kaggle numpy/scipy
#    in a live kernel."
#
# A plain `pip install tracksdata geff` ignored that and pulled a numpy upgrade, and the
# kernel died on `ImportError: cannot import name '_center' from 'numpy._core.umath'` --
# the image's compiled extensions built against the numpy that was just replaced. The pack's
# authors solved this before I arrived; the fix is to use their solution.
_wheels = next((p.parent for p in Path("/kaggle/input").glob("*/**/tracksdata-*.whl")), None)
if _wheels is None:
    _wheels = next((p for p in Path("/kaggle/input").glob("*/**/wheels") if p.is_dir()), None)
print("wheel dir:", _wheels, flush=True)
if _wheels is None:
    for _p in sorted(Path("/kaggle/input").glob("*/*")):
        print("   mounted:", _p, flush=True)
    raise RuntimeError("no offline wheels mounted -- add the support pack as a data source")
_r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                     "--find-links", str(_wheels),
                     "tracksdata", "geff", "geff_spec", "rustworkx",
                     "numcodecs", "donfig", "bidict", "zarr"],
                    capture_output=True, text=True)
print("pip rc", _r.returncode, flush=True)
if _r.returncode:
    print(_r.stdout[-1500:], _r.stderr[-1500:], flush=True)

# The official metric code, carried inline. Same bytes as the support pack ships at
# src/biohub_tracking/{metrics,division_metrics}.py -- `metrics` does a relative
# `from .division_metrics import evaluate_divisions`, so both live in one package.
_pkg = Path("/kaggle/working/bt")
_pkg.mkdir(parents=True, exist_ok=True)
(_pkg / "__init__.py").write_text("")
(_pkg / "metrics.py").write_text(METRICS_SRC)
(_pkg / "division_metrics.py").write_text(DIVISION_SRC)
sys.path.insert(0, "/kaggle/working")

import pandas as pd
import tracksdata as td
from bt import metrics as M

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = Path("/kaggle/input/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
VOXEL = (1.625, 0.40625, 0.40625)


def graph_from_geff(path):
    g = td.graph.IndexedRXGraph.from_geff(str(path))
    return g[0] if isinstance(g, tuple) else g


def node_budget(geff_path):
    try:
        from geff import GeffMetadata
        meta = GeffMetadata.read(str(geff_path))
        v = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception as e:
        print("   node budget unavailable:", e, flush=True)
        return float("nan")


def graph_from_rows(nodes, edges):
    """Build a tracksdata graph from submission rows.

    Every attribute key must be declared before the first `add_node`; a fresh graph knows
    only `t`, which is exactly what killed the previous attempt.
    """
    g = td.graph.IndexedRXGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, 0.0)
    idx = {}
    for r in nodes.itertuples(index=False):
        idx[int(r.node_id)] = g.add_node(
            {"t": int(r.t), "z": float(r.z), "y": float(r.y), "x": float(r.x)})
    for r in edges.itertuples(index=False):
        g.add_edge(source_id=idx[int(r.source_id)], target_id=idx[int(r.target_id)], attrs={})
    return g


results = {}
for name, csv_path in SUBMISSIONS.items():
    print("=" * 70, flush=True)
    print(name, csv_path, flush=True)
    if not Path(csv_path).exists():
        print("   MISSING -- mount is not where expected", flush=True)
        for p in sorted(Path("/kaggle/input").glob("*/*/*")):
            print("     ", p, flush=True)
        continue
    sub = pd.read_csv(csv_path)
    rows = []
    for ds in sorted(sub["dataset"].astype(str).unique()):
        d = sub[sub["dataset"].astype(str) == ds]
        gt_path = TRAIN / f"{ds}.geff"
        if not gt_path.exists():
            print(f"   {ds}: no ground truth at {gt_path}", flush=True)
            continue
        try:
            pred = graph_from_rows(d[d["row_type"] == "node"], d[d["row_type"] == "edge"])
            gt = graph_from_geff(gt_path)
            er = M.evaluate(pred, gt, VOXEL)
            row = M.per_sample_metrics(er, node_budget(gt_path), M.node_recall(pred, gt))
            row["dataset"] = ds
            rows.append(row)
            print(f"   {ds:<16} edge_J={row['edge_jaccard']:.4f} "
                  f"adj={row['adj_edge_jaccard']:.4f} "
                  f"div tp/fp/fn={row['division_tp']}/{row['division_fp']}/{row['division_fn']} "
                  f"ratio={row['total_node_ratio']:+.4f}", flush=True)
        except Exception:
            print(f"   {ds}: FAILED", flush=True)
            traceback.print_exc()
    if rows:
        s = M.summarise(rows)
        results[name] = {"summary": s, "per_dataset": rows}
        print(f"\\n   SUMMARY {name}", flush=True)
        for k in ("n", "edge_jaccard", "adj_edge_jaccard", "division_jaccard",
                  "division_tp", "division_fp", "division_fn", "node_recall", "score"):
            if k in s:
                print(f"      {k:<20} {s[k]}", flush=True)

print("\\n" + "=" * 70, flush=True)
for name, r in results.items():
    print(f"FINAL {name:<24} score={r['summary'].get('score')} "
          f"adj_edge={r['summary'].get('adj_edge_jaccard')} "
          f"div_J={r['summary'].get('division_jaccard')}", flush=True)
Path("/kaggle/working/score_summary.json").write_text(json.dumps(results, indent=1, default=str))
'''


def build(kernels: list[str]) -> int:
    subs = {k.replace("claude-eval-", ""):
            f"/kaggle/input/notebooks/{K.username()}/{k}/submission.csv" for k in kernels}
    src = ("METRICS_SRC = " + repr((PACK / "metrics.py").read_text()) + "\n"
           + "DIVISION_SRC = " + repr((PACK / "division_metrics.py").read_text()) + "\n"
           + "SUBMISSIONS = " + repr(subs) + "\n"
           + NB)
    compile(src, "claude_score", "exec")

    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": src.splitlines(keepends=True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = HERE / "claude_score.ipynb"
    out.write_text(json.dumps(nb, indent=1))
    (HERE / "claude_score_push.json").write_text(json.dumps({
        "slug": "claude-score", "title": "Claude score",
        "notebook": str(out),
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": kernels if all("/" in k for k in kernels)
                          else [f"{K.username()}/{k}" for k in kernels],
        "enable_gpu": False, "enable_internet": True,
    }, indent=1))
    print(f"wrote {out.name}: {len(src):,} chars, scoring {list(subs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build(sys.argv[1:] or ["claude-eval-ttasec", "claude-eval-ttaz16"]))
