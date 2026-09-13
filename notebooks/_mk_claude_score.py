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

NB = '''import json, subprocess, sys
from pathlib import Path

# Install from the support pack's OFFLINE WHEELS, with --no-deps, which is what every arm
# notebook does and what its own log explains: "Dependency resolver is disabled with
# --no-deps to avoid replacing Kaggle numpy/scipy in a live kernel."
_wheels = next((p.parent for p in Path("/kaggle/input").glob("*/**/tracksdata-*.whl")), None)
if _wheels is None:
    _wheels = next((p for p in Path("/kaggle/input").glob("*/**/wheels") if p.is_dir()), None)
print("wheel dir:", _wheels, flush=True)
if _wheels is None:
    for _p in sorted(Path("/kaggle/input").glob("*/*")):
        print("   mounted:", _p, flush=True)
    raise RuntimeError("no offline wheels mounted -- add the support pack as a data source")

# polars needs --force-reinstall: the image ships an older one, pip calls the requirement
# satisfied and skips it, and tracksdata then raises `no attribute 'Float16'`.
# polars comes from PyPI, everything else from the pack's wheels.
#
# The pack's polars wheel installs but its compiled extension does not load in this image,
# and polars hides that: `with contextlib.suppress(ImportError): from polars.polars import
# PySeries` means a failed extension import silently deletes module-level names, surfacing
# much later as `NameError: name 'PySeries' is not defined` and then
# `NameError: name '_POLARS_TYPE_TO_CONSTRUCTOR' is not defined`.
#
# polars has no dependencies, so --no-deps from PyPI cannot disturb the image's numpy, which
# is the only thing the earlier plain `pip install` got wrong. PyPI polars + the pack's
# tracksdata is the combination verified in this container.
for _stage, _pkgs, _force, _pypi in (
        ("polars", ["polars==1.44.2"], True, True),
        # The arm notebooks' list, verbatim. tracksdata imports its solvers at package load,
        # so ilpy and pyscipopt are needed whether or not anything is solved here.
        ("graph stack", ["tracksdata", "zarr", "pyscipopt", "geff", "geff_spec", "ilpy",
                         "imagecodecs", "rustworkx", "numcodecs", "donfig", "bidict"],
         False, False)):
    # --only-binary=:all: on the PyPI leg. Without it pip fell back to polars' sdist,
    # "succeeded" with rc 0, and left a package whose own warning was the giveaway:
    # "Polars binary is missing!" -- an install with no compiled extension, which is what
    # made every later name vanish. A missing wheel must be an error, not a silent build.
    _cmd = ([sys.executable, "-m", "pip", "install", "-q", "--no-deps"]
            + (["--only-binary=:all:"] if _pypi
               else ["--no-index", "--find-links", str(_wheels)])
            + (["--force-reinstall"] if _force else []) + _pkgs)
    _r = subprocess.run(_cmd, capture_output=True, text=True)
    print(f"pip [{_stage}] rc {_r.returncode}", flush=True)
    if _r.returncode:
        print(_r.stdout[-1200:], _r.stderr[-1200:], flush=True)

# Check the compiled extension directly, in a fresh interpreter, because polars swallows its
# own import failure and reports it only as a NameError somewhere else entirely -- two rounds
# were spent chasing those NameErrors instead of the import that caused them.
_chk = subprocess.run(
    [sys.executable, "-c",
     "import polars as pl, polars.polars as pp;"
     "print('polars', pl.__version__, pl.__file__);"
     "print('extension', pp.__file__);"
     "print('Series dtype', pl.Series([1.0]).dtype, '| Float16', hasattr(pl, 'Float16'))"],
    capture_output=True, text=True)
print(_chk.stdout.strip(), flush=True)
if _chk.returncode:
    print("POLARS IS BROKEN:", _chk.stderr[-900:], flush=True)
    raise RuntimeError("polars will not import cleanly -- scoring cannot run")

# The official metric code, carried inline -- the same bytes the support pack ships at
# src/biohub_tracking/{metrics,division_metrics}.py. `metrics` does a relative
# `from .division_metrics import evaluate_divisions`, so both live in one package.
_pkg = Path("/kaggle/working/bt")
_pkg.mkdir(parents=True, exist_ok=True)
(_pkg / "__init__.py").write_text("")
(_pkg / "metrics.py").write_text(METRICS_SRC)
(_pkg / "division_metrics.py").write_text(DIVISION_SRC)
Path("/kaggle/working/score_impl.py").write_text(IMPL_SRC)

# Run the scoring in a FRESH INTERPRETER. Force-reinstalling polars under a process that has
# already imported it leaves a half-replaced package: the version string comes back empty and
# every dataset dies on `NameError: name 'PySeries' is not defined`. The arm notebooks never
# see this because they install in the notebook and predict in a subprocess; so does this.
_rc = subprocess.run([sys.executable, "/kaggle/working/score_impl.py",
                      json.dumps(SUBMISSIONS)], cwd="/kaggle/working")
print("scoring rc", _rc.returncode, flush=True)
if _rc.returncode:
    raise RuntimeError(f"scoring failed rc={_rc.returncode}")
'''

IMPL = '''"""Scoring, run in its own interpreter so the freshly installed polars loads clean."""
import json, sys, traceback
from pathlib import Path

import pandas as pd
import polars as pl
import tracksdata as td

sys.path.insert(0, "/kaggle/working")
from bt import metrics as M

print("polars", pl.__version__, "| tracksdata ok", flush=True)

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():
    COMP = Path("/kaggle/input/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
VOXEL = (1.625, 0.40625, 0.40625)


def graph_from_geff(path):
    """Read the GT graph WITHOUT tracksdata's own geff loader.

    `IndexedRXGraph.from_geff` calls `pl.Series([value])` on each default attribute, and on
    these ground-truth files that lands in a polars branch referencing `PySeries`, which this
    polars only imports under TYPE_CHECKING:

        File "polars/_utils/construction/series.py", line 322, in sequence_to_pyseries
            elif python_dtype == PySeries:
        NameError: name 'PySeries' is not defined

    The arm notebooks never hit it because they only ever load *prediction* geffs they wrote
    themselves. Reading via `geff` into networkx and then building the graph with the same
    code that builds the prediction graph sidesteps the incompatibility -- and has the better
    property that both sides of the comparison are now constructed identically.
    """
    import geff
    nxg, _meta = geff.read(str(path), backend="networkx")
    return graph_from_nodes(
        [(n, d) for n, d in nxg.nodes(data=True)],
        list(nxg.edges()),
        label=str(path.name),
    )


def graph_from_nodes(nodes, edges, label=""):
    g = td.graph.IndexedRXGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, 0.0)
    idx = {}
    for nid, d in nodes:
        missing = [k for k in ("t", "z", "y", "x") if k not in d]
        if missing:
            raise KeyError(f"{label}: node {nid} lacks {missing}; has {sorted(d)}")
        idx[nid] = g.add_node({"t": int(d["t"]), "z": float(d["z"]),
                               "y": float(d["y"]), "x": float(d["x"])})
    for u, v in edges:
        g.add_edge(source_id=idx[u], target_id=idx[v], attrs={})
    return g


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
    only `t`, which is what killed the first attempt. Verified locally against a real
    submission: 25,622 nodes in 0.6s.
    """
    return graph_from_nodes(
        [(int(r.node_id), {"t": r.t, "z": r.z, "y": r.y, "x": r.x})
         for r in nodes.itertuples(index=False)],
        [(int(r.source_id), int(r.target_id)) for r in edges.itertuples(index=False)],
        label="submission")


results = {}
for name, csv_path in json.loads(sys.argv[1]).items():
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
           + "IMPL_SRC = " + repr(IMPL) + "\n"
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
