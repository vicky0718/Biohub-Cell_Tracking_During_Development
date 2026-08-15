"""Build notebooks/02_classical_baseline.ipynb."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "02_classical_baseline.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Classical baseline — first submission, and experiment #1

No training. Local-maxima detection on the quantile-normalised image, then optimal
assignment between consecutive frames. Three jobs:

0. **Answer two open questions from recon** before anything else: what the `test/`
   folder actually contains (which settles the train/test name overlap flagged in
   `notes/04-recon-results.md` §0), and whether the per-dataset **node budget** is
   readable at test time. The second one decides whether the pipeline can protect
   itself from the trap in §9 — where a fixed detection density zeroes the adjusted
   Jaccard on 39.7% of the test weight.
1. **Put a real number on the leaderboard**, so every later change has something to beat.
2. **Run experiment #1** — the detection-threshold sweep.

`notes/02-metric-findings.md` argues the official baseline's `--det-threshold 0.99` is
tuned on the axis that barely matters: unmatched detections cost little, while a missed
detection is 2 permanent FN and unrecoverable. If that reading is right, the score should
*improve* as the threshold falls, until the node budget bites.

**That is a prediction, not a result.** The sweep is how we find out, and the honest
outcome is whatever the harness reports.

> **No `tracksdata` here, on purpose.** It needs `numpy>2`, Kaggle pins `numpy<2`, and
> installing it rewrites numpy underneath the running kernel — that killed two recon
> runs. Scoring goes through `harness/purescore.py`, a numpy-only reimplementation that
> `probes/verify_purescore.py` shows reproduces the official TP/FP/FN **exactly**.
""")

code(r"""
# --- deps -------------------------------------------------------------------
# geff reads the ground-truth graphs, zarr reads the images. Nothing else.
# Do NOT add tracksdata: it drags numpy>2 in and breaks scipy mid-kernel.
import subprocess, sys, os

def pip_install(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode

print("installing geff + zarr ...")
pip_install(["geff", "zarr"])

import importlib
for m in ("numpy", "scipy", "zarr", "geff", "polars"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<8} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"  {m:<8} MISSING — {e}")
""")

code(r"""
# Self-contained on purpose: this cell must work after a kernel restart, without
# re-running the install cell above.
import sys, os, time
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")

# Find our harness/ and pipeline/, and the competition data, by SEARCHING the mounts.
# Kaggle nests dataset paths differently depending on how the dataset was created
# (/kaggle/input/<slug>/, /kaggle/input/<slug>/<zip-root>/,
#  /kaggle/input/datasets/<user>/<slug>/<zip-root>/ ...), so hardcoding one shape just
# breaks on the next upload. Never descend into .zarr/.geff — those hold thousands of
# chunk files and would make this crawl take minutes.
def find_dir(is_match, roots, max_depth=5):
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        stack = [(root, 0)]
        while stack:
            d, depth = stack.pop(0)
            try:
                if is_match(d):
                    return d
                if depth >= max_depth:
                    continue
                kids = [e for e in d.iterdir()
                        if e.is_dir() and e.suffix not in (".zarr", ".geff")]
            except (PermissionError, OSError):
                continue
            stack += [(k, depth + 1) for k in kids]
    return None


REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
if REPO is None:
    raise SystemExit(
        "Could not find our harness/ and pipeline/ code under /kaggle/input or "
        "/kaggle/working.\nUpload the project zip as a Kaggle Dataset and add it as an "
        "input (Add Input -> Datasets), or unzip it into /kaggle/working/."
    )
sys.path.insert(0, str(REPO))

from harness import (Harness, Tracks, build_submission, gate, purescore,
                     read_estimated_nodes, validate_submission)
from pipeline.classical import Config, estimated_total_nodes, make_predictor, predict_dataset

# The competition mount is whichever directory holds train/ and test/ with .zarr in them.
COMP = find_dir(
    lambda p: (p / "train").is_dir() and (p / "test").is_dir()
    and any((p / "train").glob("*.zarr")),
    ["/kaggle/input"])
if COMP is None:
    raise SystemExit(
        "Could not find the competition data (a folder with train/ and test/ full of "
        ".zarr).\nAdd Input -> Competitions -> Biohub Cell Tracking During Development."
    )
TRAIN, TEST = COMP / "train", COMP / "test"

CACHE = WORK / "cache"
CACHE.mkdir(exist_ok=True, parents=True)
print("project:", REPO, "| data:", COMP)
""")

md("""## 0. What is actually in `test/`?

Two things recon left open, and both change what we do next.

**(a) The train/test name overlap.** All four test dataset names also appear in train and
we hold their ground truth. If `sample_submission.csv` names exactly those four, the
visible test set is very likely the scored set — which would be a competition-breaking
flaw, and the right response is to **tell the organisers, not exploit it**. If it names
other datasets, the visible four are a worked example and nothing more.

**(b) Is the node budget readable at test time?** `estimated_number_of_nodes` lives in
the GEFF metadata. For train that is the `.geff` next to each image. If `test/` ships no
`.geff`, we cannot read the budget for the datasets we are actually scored on, and the
per-dataset cap that §9 says is required has nothing to key off. That would need a
different fallback — estimate density from the image itself — so find out now.
""")

code(r"""
print("=== what the competition mount contains ===")
for d in sorted(COMP.iterdir()) if COMP.exists() else []:
    kind = "dir " if d.is_dir() else "file"
    size = "" if d.is_dir() else f"  {d.stat().st_size:,} bytes"
    print(f"  {kind}  {d.name}{size}")

print("\n=== per-folder file types ===")
for label, folder in (("train", TRAIN), ("test", TEST)):
    if not folder.exists():
        print(f"  {label}: MISSING"); continue
    from collections import Counter
    ext = Counter(p.suffix for p in folder.iterdir())
    print(f"  {label}: " + ", ".join(f"{n}x {e or '(none)'}" for e, n in ext.most_common()))

train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
test_names = sorted(p.stem for p in TEST.glob("*.zarr"))
test_geffs = sorted(p.stem for p in TEST.glob("*.geff"))
print(f"\n{len(train_names)} train (image+GT) / {len(test_names)} test images / "
      f"{len(test_geffs)} test .geff")
print("test:", test_names)
""")

code(r"""
# (a) the overlap question
ss = next((p for p in COMP.glob("sample_submission*") if COMP.exists()), None)
if ss is not None:
    import polars as pl
    df = pl.read_csv(ss)
    named = sorted(df["dataset"].unique().to_list()) if "dataset" in df.columns else []
    print(f"sample_submission: {ss.name}  {df.height:,} rows  columns={df.columns}")
    print(f"  datasets named ({len(named)}): {named}")
    overlap = sorted(set(named) & set(train_names))
    if named and set(named) == set(test_names) and overlap:
        print("\n!! The submission file asks for exactly the datasets whose ground truth")
        print("   ships in train/. If the leaderboard scores these, it is trivially")
        print("   saturable. REPORT THIS TO THE ORGANISERS rather than exploiting it,")
        print("   and do not read any leaderboard position built on it as signal.")
    elif named:
        print(f"\n  {len(overlap)} of the {len(named)} scored datasets also appear in train.")
else:
    print("no sample_submission file found on the mount — check the competition Data tab")
""")

code(r"""
# (b) can we read the node budget for the datasets we are SCORED on?
print("=== node budget availability ===")
rows = []
for n in test_names:
    from_geff = read_estimated_nodes(TEST / f"{n}.geff")
    from_any = estimated_total_nodes(TEST / f"{n}.zarr")
    rows.append((n, from_geff, from_any))
    print(f"  {n:<28} geff={from_geff if from_geff == from_geff else 'n/a':>12}  "
          f"resolved={from_any if from_any is not None else 'NONE':>12}")

have = [r for r in rows if r[2] is not None]
print(f"\n{len(have)}/{len(rows)} test datasets expose a node budget")
if len(have) < len(rows):
    print("!! Without it, Config.budget_fill has nothing to key off on those datasets and")
    print("   detection runs UNCAPPED there. recon 04 §9: a fixed density against a sparse")
    print("   crop drives the adjusted Jaccard to 0. Fall back to estimating cells/frame")
    print("   from the image (foreground volume / nucleus volume) before submitting.")
else:
    print("Good — the per-dataset cap in Config.budget_fill will bind on test.")
""")

md("""## 1. Smoke test and timing

Before committing to a full run, check the pipeline works on **one dataset, a few frames**,
and measure how long a frame takes. A sweep that would need 40 hours is worth knowing about
now rather than at hour six.
""")

code(r"""
name = train_names[0]
cfg = Config(max_frames=5)          # everything else is the recon-derived default
print(f"config: det_threshold={cfg.det_threshold} min_sep={cfg.min_separation_um}um "
      f"link_radius={cfg.link_radius_um}um budget_fill={cfg.budget_fill}")

t0 = time.time()
tr = predict_dataset(TRAIN / name, cfg, verbose=True)
dt = time.time() - t0
print(f"\n{name}: {tr.n_nodes:,} nodes, {tr.n_edges:,} edges in {dt:.1f}s "
      f"({dt/max(1,cfg.max_frames):.2f}s per frame)")

import zarr
T_full = zarr.open_group(str(TRAIN / f"{name}.zarr"), mode="r")["0"].shape[0]
per_ds = dt / max(1, cfg.max_frames) * T_full
print(f"\nfull dataset ({T_full} frames) ~= {per_ds/60:.1f} min")
print(f"all {len(train_names)} train datasets ~= {per_ds*len(train_names)/60:.1f} min per sweep arm")
print(f"all {len(test_names)} test datasets  ~= {per_ds*len(test_names)/60:.1f} min per submission")
print("\nIf that is too slow: raise downsample, cap max_frames for the sweep, or sweep on "
      "a subset of datasets. Do NOT quietly shrink the eval set between arms - the "
      "comparison has to stay paired.")
""")

code(r"""
# Does the budget cap actually bind, and does scoring work end to end?
from harness import read_geff, read_scale
gt = read_geff(TRAIN / f"{name}.geff")
scale = read_scale(TRAIN / f"{name}.zarr")
n_est = estimated_total_nodes(TRAIN / f"{name}.zarr")

counts = purescore.count_edges(tr.t, tr.zyx, tr.edges, gt.t, gt.zyx, gt.edges, scale=scale)
row = purescore.per_sample(counts, n_est if n_est else float("nan"), gt.n_nodes,
                           n_gt_divisions=gt.n_divisions)
print(f"{name} on {cfg.max_frames} frames:")
print(f"  TP/FP/FN = {counts.tp}/{counts.fp}/{counts.fn}   edge_J={row['edge_jaccard']:.4f}")
print(f"  node_recall={row['node_recall']:.3f}  (GT nodes in these frames are a subset)")
print(f"  budget {n_est:,.0f} -> cap {n_est/T_full:.0f} detections/frame; "
      f"we emitted {tr.n_nodes/max(1,cfg.max_frames):.0f}/frame")
""")

md("""## 2. Experiment #1 — the detection-threshold sweep

Pre-registered, before looking at any result:

- **Prediction**: score improves as the threshold falls from 0.99, because unmatched
  detections are cheap while missed detections are unrecoverable.
- **What would falsify it**: score flat or worse at lower thresholds. Two mechanisms could
  do that — the node-budget multiplier, and extra distractors making the assignment harder
  (a wrong link costs 2x a missing one).
- **Second prediction, from recon §9**: with `budget_fill=1.0` the low-threshold arms
  should *stop* degrading, because the per-dataset cap holds `N_pred` at the budget no
  matter how permissive the threshold gets. Running one arm with `budget_fill=None`
  measures how much that guard is worth.
- **Decision rule**: adopt a lower threshold only if `gate()` passes, i.e. it improves the
  pooled score *and* regresses no fold. A pooled gain that costs a fold is rejected.

Use a subset of datasets and capped frames if the timing above demands it — but keep the
subset **identical across arms**.
""")

code(r"""
SWEEP_THRESHOLDS = [0.99, 0.70, 0.50, 0.30, 0.15, 0.05]
SWEEP_NAMES = train_names          # shrink if the timing cell says to, but keep it FIXED
SWEEP_MAX_FRAMES = None            # e.g. 20 for a fast first pass

budget = {n: estimated_total_nodes(TRAIN / f"{n}.zarr") for n in SWEEP_NAMES}
budget = {k: v for k, v in budget.items() if v}
h = Harness(data_dir=TRAIN, cache_dir=CACHE, n_total_override=budget)

results = {}
for th in SWEEP_THRESHOLDS:
    cfg = Config(det_threshold=th, max_frames=SWEEP_MAX_FRAMES)
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg, budgets=budget), arm=f"th{th}",
                     names=SWEEP_NAMES, verbose=False)
    results[th] = res
    s = res.summary
    print(f"threshold {th:<5} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"adj={s['adj_edge_jaccard']:.4f}  node_recall={s['node_recall']:.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
""")

code(r"""
print("=== sweep summary ===")
for th, res in sorted(results.items(), reverse=True):
    s = res.summary
    print(f"  {th:<6} score={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"adj={s['adj_edge_jaccard']:.4f}  "
          f"nodes={sum(r['num_pred_nodes'] for r in res.rows.values()):,}")

incumbent = results[max(results)]          # the official baseline's 0.99
best_th = max(results, key=lambda t: results[t].score)
print(f"\nincumbent (0.99): {incumbent.score:.4f}")
print(f"best sweep arm  : {best_th} -> {results[best_th].score:.4f}")
print()
print(gate(incumbent, results[best_th]))
print()
print("PROMOTE means it beat 0.99 pooled AND in every fold. REJECT with a positive pooled "
      "delta means the gain came out of a fold - which is the shape of overfitting, and the "
      "reason this gate exists. If the prediction failed outright, say so in the notes: a "
      "falsified prediction that was written down beforehand is worth more than a vague one "
      "that was not.")
""")

code(r"""
# Is the per-dataset budget cap actually earning its place? One paired arm, cap OFF.
cfg_off = Config(det_threshold=best_th, budget_fill=None, max_frames=SWEEP_MAX_FRAMES)
res_off = h.evaluate(make_predictor(cfg_off), arm=f"th{best_th}_nobudget",
                     names=SWEEP_NAMES, verbose=False)
print("budget cap ON :", f"{results[best_th].score:.4f}")
print("budget cap OFF:", f"{res_off.score:.4f}")
print()
print(gate(res_off, results[best_th]))
print("\nrecon §9 predicts the cap helps most where the true density is LOW. Check the "
      "per-dataset rows for the sparse crops specifically, not just the pooled number.")
""")

code(r"""
# Per-dataset detail, weighted the way the scorer weights it (w = TP+FP+FN).
# recon §9: adj_edge_jaccard is a WEIGHTED MEAN of per-dataset values, so a handful of
# dense datasets can carry the whole number. Look at where the weight actually is.
best = results[best_th]
rows = [(n, r) for n, r in best.rows.items()]
rows.sort(key=lambda kv: -(kv[1]["edge_tp"] + kv[1]["edge_fp"] + kv[1]["edge_fn"]))
total_w = sum(r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for _, r in rows)
print(f"{'dataset':<28} {'weight':>8} {'edge_J':>8} {'adj':>8} {'ratio':>8} {'recall':>7}")
for n, r in rows[:15]:
    w = r["edge_tp"] + r["edge_fp"] + r["edge_fn"]
    print(f"{n:<28} {w/total_w:>7.1%} {r['edge_jaccard']:>8.4f} {r['adj_edge_jaccard']:>8.4f} "
          f"{r['total_node_ratio']:>+8.2f} {r['node_recall']:>7.3f}")
zeroed = [n for n, r in rows if r["adj_edge_jaccard"] == 0]
if zeroed:
    print(f"\n!! {len(zeroed)} dataset(s) scored adj=0 — over budget by 11x or worse: {zeroed[:8]}")
""")

md("""## 3. Submission

Predict on test, validate, write `submission.csv`. The validator catches what the scorer
silently repairs — edges not spanning `t → t+1` (dropped), duplicate pairs (de-duped),
merges (collapsed), out-degree above 2 (truncated), and missing test datasets.
""")

code(r"""
FINAL = Config(det_threshold=best_th)
print(f"predicting {len(test_names)} test datasets at threshold {FINAL.det_threshold}\n")

graphs = {}
for i, n in enumerate(test_names, 1):
    t0 = time.time()
    graphs[n] = predict_dataset(TEST / n, FINAL, verbose=False)
    nb = estimated_total_nodes(TEST / f"{n}.zarr")
    ratio = (graphs[n].n_nodes - nb) / nb if nb else float("nan")
    print(f"  [{i}/{len(test_names)}] {n:<28} {graphs[n].n_nodes:>8,} nodes  "
          f"{graphs[n].n_edges:>8,} edges  budget ratio={ratio:+.2f} "
          f"-> x{max(0, 1 - 0.1*ratio):.3f}  ({time.time()-t0:.0f}s)", flush=True)

csv = build_submission(graphs, "/kaggle/working/submission.csv")
problems = validate_submission(csv, expected_datasets=test_names)
print("\nREADY TO SUBMIT" if not problems else f"\nFIX {len(problems)} PROBLEM(S) FIRST")
""")

code(r"""
# Record the sweep so the next session starts from measurements, not memory.
import json
payload = {
    "sweep": {str(th): {k: (None if isinstance(v, float) and v != v else v)
                        for k, v in res.summary.items()}
              for th, res in results.items()},
    "best_threshold": best_th,
    "budget_cap_off": {k: (None if isinstance(v, float) and v != v else v)
                       for k, v in res_off.summary.items()},
    "config": {"min_separation_um": FINAL.min_separation_um,
               "link_radius_um": FINAL.link_radius_um,
               "budget_fill": FINAL.budget_fill,
               "downsample": list(FINAL.downsample)},
    "sweep_names": SWEEP_NAMES,
    "sweep_max_frames": SWEEP_MAX_FRAMES,
    "test_names": test_names,
    "test_has_geff": test_geffs,
}
Path("/kaggle/working/sweep_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(json.dumps(payload["sweep"], indent=2))
print("\nWrote /kaggle/working/sweep_results.json - commit it back to the repo.")
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")

for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        stripped = "\n".join("pass  # shell" if l.strip().startswith("!") else l
                             for l in src.splitlines())
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} syntax error: {e}\n---\n{src}")
print("all code cells parse as valid Python")
