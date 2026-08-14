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
assignment between consecutive frames. Two jobs:

1. **Put a real number on the leaderboard**, so every later change has something to beat.
2. **Run experiment #1** — the detection-threshold sweep.

`notes/02-metric-findings.md` argues the official baseline's `--det-threshold 0.99` is
tuned on the axis that barely matters: unmatched detections cost nothing but the weak node
budget term, while a missed detection is 2 permanent FN and unrecoverable. If that reading
is right, the score should *improve* as the threshold falls, until the node budget bites.

**That is a prediction, not a result.** The sweep is how we find out, and the honest
outcome is whatever the harness reports.

> Run `01_recon.ipynb` first. Its measurements set `min_separation_um`, `link_radius_um`
> and `max_per_frame`; the defaults here are placeholders from the domain notes.
""")

code(r"""
# --- deps + code -------------------------------------------------------------
!pip install -q tracksdata 2>&1 | tail -2

import subprocess, sys, os
from pathlib import Path

WORK = Path("/kaggle/working")

# 1. the official scorer — our numbers must be the leaderboard's numbers
CELLMOT = WORK / "kaggle-cell-tracking-competition"
if not (CELLMOT / "src").is_dir():
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
                    str(CELLMOT)], check=False)
os.environ["CELLMOT_REPO"] = str(CELLMOT)

# 2. our own harness + pipeline. Easiest is to attach this repo as a Kaggle dataset
#    (Add Input -> Datasets -> upload the zip) or clone it if it is public.
CANDIDATES = [
    WORK / "biohub-cell-tracking",
    Path("/kaggle/input/biohub-cell-tracking"),
    Path("/kaggle/input/biohub-cell-tracking/biohub-cell-tracking"),
]
REPO = next((p for p in CANDIDATES if (p / "harness").is_dir()), None)
if REPO is None:
    raise SystemExit(
        "Could not find our harness/ and pipeline/ code.\n"
        "Upload the project zip as a Kaggle Dataset and add it as an input, or unzip it "
        "into /kaggle/working/biohub-cell-tracking."
    )
sys.path.insert(0, str(REPO))
print("scorer:", CELLMOT, "| project:", REPO)
""")

code(r"""
import time
import numpy as np

from harness import Harness, gate, build_submission, validate_submission
from harness.scorer import read_estimated_nodes
from pipeline.classical import Config, predict_dataset, make_predictor

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():
    alt = Path("/kaggle/input/biohub-cell-tracking-during-development")
    COMP = alt if alt.exists() else COMP
TRAIN, TEST = COMP / "train", COMP / "test"

train_names = sorted(p.stem for p in TRAIN.glob("*.zarr"))
test_names = sorted(p.stem for p in TEST.glob("*.zarr"))
print(f"{len(train_names)} train / {len(test_names)} test datasets")

CACHE = Path("/kaggle/working/cache")
CACHE.mkdir(exist_ok=True, parents=True)
""")

md("""## 1. Smoke test and timing

Before committing to a full run, check the pipeline works on **one dataset, a few frames**,
and measure how long a frame takes. A sweep that would need 40 hours is worth knowing about
now rather than at hour six.
""")

code(r"""
name = train_names[0]
cfg = Config(det_threshold=0.30, min_separation_um=5.0, link_radius_um=8.0,
             downsample=(1, 4, 4), max_frames=5)

t0 = time.time()
g = predict_dataset(TRAIN / name, cfg, verbose=True)
dt = time.time() - t0

print(f"\n{name}: {g.num_nodes():,} nodes, {g.num_edges():,} edges in {dt:.1f}s "
      f"({dt/max(1,cfg.max_frames):.2f}s per frame)")

# Extrapolate honestly before launching anything long.
import zarr
T_full = zarr.open_group(str(TRAIN / f"{name}.zarr"), mode="r")["0"].shape[0]
per_ds = dt / max(1, cfg.max_frames) * T_full
print(f"\nfull dataset ({T_full} frames) ~= {per_ds/60:.1f} min")
print(f"all {len(train_names)} train datasets   ~= {per_ds*len(train_names)/60:.1f} min per sweep arm")
print(f"all {len(test_names)} test datasets    ~= {per_ds*len(test_names)/60:.1f} min per submission")
print("\nIf that is too slow: raise downsample, cap max_frames for the sweep, or sweep on "
      "a subset of datasets. Do NOT quietly shrink the eval set between arms - the "
      "comparison has to stay paired.")
""")

code(r"""
# Sanity-check the detection density against the node budget before sweeping.
n_est = read_estimated_nodes(TRAIN / f"{name}.geff")
if n_est == n_est:  # not NaN
    per_frame_pred = g.num_nodes() / max(1, cfg.max_frames)
    projected = per_frame_pred * T_full
    ratio = (projected - n_est) / n_est
    print(f"projected nodes for {name}: {projected:,.0f} vs budget {n_est:,.0f}")
    print(f"  node ratio = {ratio:+.2f}  ->  adj multiplier = {max(0, 1 - 0.1*ratio):.3f}")
    print("\nRemember the multiplier is weak and two-sided: 2x the budget costs 10%, and "
          "UNDER-predicting pays a bonus up to 1.1x. It should not dominate the threshold "
          "choice unless we are wildly over-detecting.")
else:
    print("No estimated_number_of_nodes in the geff metadata - the adjusted metric cannot "
          "be reproduced locally. Sweep on raw edge_jaccard and treat adj as unknown.")
""")

md("""## 2. Experiment #1 — the detection-threshold sweep

Pre-registered, before looking at any result:

- **Prediction**: score improves as the threshold falls from 0.99, because unmatched
  detections are nearly free while missed detections are unrecoverable.
- **What would falsify it**: score flat or worse at lower thresholds. Two mechanisms could
  do that — the node-budget multiplier, and extra distractors making the assignment harder
  (a wrong link costs 2x a missing one).
- **Decision rule**: adopt a lower threshold only if `gate()` passes, i.e. it improves the
  pooled score *and* regresses no fold. A pooled gain that costs a fold is rejected.

Use a subset of datasets and capped frames if the timing above demands it — but keep the
subset **identical across arms**.
""")

code(r"""
SWEEP_THRESHOLDS = [0.99, 0.70, 0.50, 0.30, 0.15, 0.05]
SWEEP_NAMES = train_names          # shrink if the timing cell says to, but keep it FIXED
SWEEP_MAX_FRAMES = None            # e.g. 20 for a fast first pass

budget = {n: read_estimated_nodes(TRAIN / f"{n}.geff") for n in SWEEP_NAMES}
h = Harness(data_dir=TRAIN, cache_dir=CACHE, n_total_override=budget)

results = {}
for th in SWEEP_THRESHOLDS:
    cfg = Config(det_threshold=th, min_separation_um=5.0, link_radius_um=8.0,
                 downsample=(1, 4, 4), max_frames=SWEEP_MAX_FRAMES)
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg), arm=f"th{th}", names=SWEEP_NAMES, verbose=False)
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
          f"adj={s['adj_edge_jaccard']:.4f}  nodes={sum(r['num_pred_nodes'] for r in res.rows.values()):,}")

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

md("""## 3. Submission

Predict on test, validate, write `submission.csv`. The validator catches what the scorer
silently repairs — edges not spanning `t → t+1` (dropped), duplicate pairs (de-duped),
merges (collapsed), out-degree above 2 (truncated), and missing test datasets.
""")

code(r"""
FINAL = Config(det_threshold=best_th, min_separation_um=5.0, link_radius_um=8.0,
               downsample=(1, 4, 4))
print(f"predicting {len(test_names)} test datasets at threshold {FINAL.det_threshold}\n")

graphs = {}
for i, n in enumerate(test_names, 1):
    t0 = time.time()
    graphs[n] = predict_dataset(TEST / n, FINAL, verbose=False)
    print(f"  [{i}/{len(test_names)}] {n:<28} {graphs[n].num_nodes():>8,} nodes  "
          f"{graphs[n].num_edges():>8,} edges  ({time.time()-t0:.0f}s)", flush=True)

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
    "config": {"min_separation_um": FINAL.min_separation_um,
               "link_radius_um": FINAL.link_radius_um,
               "downsample": list(FINAL.downsample)},
    "sweep_names": SWEEP_NAMES,
    "sweep_max_frames": SWEEP_MAX_FRAMES,
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
