"""Build notebooks/03_linking.ipynb."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "03_linking.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Experiment #2 — the linking gap

`notes/05-first-sweep.md` §3 found the thing that redirects this project.

At the best arm of the threshold sweep: `node_recall = 0.845`, `edge_jaccard = 0.528`.
If linking were as good as recon's measured ceiling, edge Jaccard would be about
`0.845² × 0.9915 = 0.707`. **The shortfall is 0.179** — roughly 42,000 wrong links
against 38,000 missing ones, an edge precision of ~0.68.

Recon §7 concluded "linking is solved, detection is the whole contest". That was measured
with **perfect detections**: 133k ground-truth nodes, ~6.6 per frame, 25 µm apart.
Nearest-neighbour is trivially right at that spacing. The real detection field is **~210
nodes per frame at ~8 µm spacing with 15 % of true successors missing** — and when a
cell's real successor was never detected, a 9 µm search radius does not decline to link.
It links to a neighbour, turning one false negative into a false negative *and* a false
positive.

So: sweep the two knobs that set the density and reach of the linker, as a **grid**,
because they interact — a denser detection field makes linking harder.

- `link_radius_um` — currently 9.0, from a p99 measured on the sparse annotated set.
- `min_separation_um` — currently 6.0. This is what caps recall now that the threshold has
  saturated: 0.15 → 0.05 bought 146k nodes and **+0.0014** recall.

**Pre-registered predictions**, written before any of this runs:

1. **Score improves as `link_radius_um` falls below 9.** We are FP-heavy, and the metric's
   own rule — link when `p > J/(1+J)`, which is **0.35** at `J = 0.53` — says a candidate
   must be right about a third of the time to be worth linking. At 9 µm in a field with
   8 µm spacing, many are not. *Falsified if* the score is flat or falls monotonically as
   the radius tightens, which would mean the missing links cost more than the wrong ones.
2. **Smaller `min_separation_um` raises `node_recall`.** *Falsified if* recall is flat —
   which would mean the missing 15 % is invisible in this channel, not suppressed, and no
   amount of detector tuning will reach it.
3. **The two interact negatively**: the best radius at `min_sep = 3.5` is smaller than the
   best radius at `min_sep = 6.0`. *Falsified if* the best radius is the same at both.

Every arm is scored by the same harness on the same fixed subset, and promotion still
requires a pooled gain with no fold regression.
""")

code(r"""
# --- deps -------------------------------------------------------------------
# geff reads the ground truth, zarr reads the images. NOT tracksdata: it needs numpy>2,
# Kaggle pins numpy<2, and installing it rewrites numpy under the running kernel.
import subprocess, sys

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
# Self-contained: works after a kernel restart without re-running the install cell.
import sys, os, time, json, itertools
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

from harness import (Harness, build_submission, gate, purescore, read_geff,
                     read_scale, validate_submission)
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

train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
test_names = sorted(p.stem for p in TEST.glob("*.zarr"))
print("project:", REPO)
print(f"{len(train_names)} train / {len(test_names)} test")
""")

md("""## 1. A fixed subset — chosen once, then never changed

A full arm over 199 datasets took **~40 minutes**. A 4x3 grid would be 8 hours, which does
not fit alongside anything else. So the grid runs on a fixed subset.

The subset is chosen **deterministically and before any scoring**, stratified so that both
embryos and the full range of annotation density are represented. Two rules:

- **It never changes between arms.** A comparison whose eval set moved is not a comparison.
- **It is chosen by name hash, not by score.** Picking datasets we already do well on is
  how a sweep produces a number that does not survive contact with the leaderboard.
""")

code(r"""
import hashlib

SUBSET_SIZE = 60

def stable_key(n):
    return int(hashlib.sha1(n.encode()).hexdigest(), 16)

# stratify by prefix so both embryos keep their share of the subset
by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)

SUBSET = []
for pfx, names in sorted(by_prefix.items()):
    k = round(SUBSET_SIZE * len(names) / len(train_names))
    SUBSET += sorted(names, key=stable_key)[:k]
SUBSET = sorted(SUBSET)
print(f"subset: {len(SUBSET)} datasets")
for pfx, names in sorted(by_prefix.items()):
    n_in = sum(1 for s in SUBSET if s.startswith(pfx))
    print(f"  {pfx}: {n_in}/{len(names)} ({n_in/len(SUBSET):.0%} of subset, "
          f"{len(names)/len(train_names):.0%} of train)")

h = Harness(data_dir=TRAIN, cache_dir=CACHE)
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
print(f"\nfolds: " + ", ".join(f"{f}:{len(v)}" for f, v in sorted(folds.items())))
if min(len(v) for v in folds.values()) < 3:
    print("!! a fold has under 3 datasets — the no-regression gate is weak here")
""")

md("""## 2. Reproduce the incumbent on the subset

Before sweeping, confirm the subset gives roughly the same number as the full 199. If the
subset score is wildly different, it is not representative and every comparison run on it
is measuring the subset, not the change.

Full-set reference from `notes/05`: threshold 0.15, `budget_fill=None` -> **0.5552**.
""")

code(r"""
BASE = Config(det_threshold=0.15, min_separation_um=6.0, link_radius_um=9.0,
              budget_fill=None)
t0 = time.time()
base = h.evaluate(make_predictor(BASE), arm="base_r9.0_s6.0", names=SUBSET, verbose=False)
s = base.summary
print(f"incumbent on subset: SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
      f"node_recall={s['node_recall']:.3f}  ({time.time()-t0:.0f}s)")
print(f"full-set reference : 0.5552")
print(f"difference         : {s['score'] - 0.5552:+.4f}")
print("\nA gap under ~0.03 is fine — the subset is smaller and weighted differently. "
      "A large gap means the subset is not representative; widen it before trusting anything below.")
ARM_SECONDS = time.time() - t0
""")

md("""## 3. The grid

`link_radius_um` x `min_separation_um`. Budget the run from the timing above before
starting: 12 arms at the measured per-arm cost.
""")

code(r"""
RADII = [9.0, 7.0, 5.0, 4.0]
SEPS = [6.0, 4.5, 3.5]

n_arms = len(RADII) * len(SEPS)
print(f"{n_arms} arms x ~{ARM_SECONDS/60:.1f} min = ~{n_arms*ARM_SECONDS/3600:.1f} hours")
print("Kaggle CPU sessions run 12h. If that is too long, cut SEPS to [6.0, 4.0] first — "
      "prediction 1 (the radius) is the one with a measured 0.179 behind it.\n")

grid = {}
for sep, rad in itertools.product(SEPS, RADII):
    cfg = Config(det_threshold=0.15, min_separation_um=sep, link_radius_um=rad,
                 budget_fill=None)
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg), arm=f"r{rad}_s{sep}", names=SUBSET, verbose=False)
    grid[(sep, rad)] = res
    st = res.summary
    print(f"  sep={sep:<4} radius={rad:<4} SCORE={st['score']:.4f}  "
          f"edge_J={st['edge_jaccard']:.4f}  recall={st['node_recall']:.3f}  "
          f"nodes={sum(r['num_pred_nodes'] for r in res.rows.values()):>9,}  "
          f"({time.time()-t0:.0f}s)", flush=True)
""")

code(r"""
print("=== SCORE grid (rows = min_separation_um, cols = link_radius_um) ===")
print("sep / rad ".ljust(10) + "".join(f"{r:>10}" for r in RADII))
for sep in SEPS:
    row = "".join(f"{grid[(sep, r)].score:>10.4f}" if (sep, r) in grid else f"{'-':>10}"
                  for r in RADII)
    print(f"{sep:<10}{row}")

print("\n=== node_recall grid (detection only — the radius must not change this) ===")
print("sep / rad ".ljust(10) + "".join(f"{r:>10}" for r in RADII))
for sep in SEPS:
    row = "".join(f"{grid[(sep, r)].summary['node_recall']:>10.3f}" if (sep, r) in grid
                  else f"{'-':>10}" for r in RADII)
    print(f"{sep:<10}{row}")

best_key = max(grid, key=lambda k: grid[k].score)
print(f"\nbest: min_sep={best_key[0]}  link_radius={best_key[1]}  "
      f"-> {grid[best_key].score:.4f}")
print(f"incumbent (6.0, 9.0)                 -> {grid[(6.0, 9.0)].score:.4f}")
print()
print(gate(grid[(6.0, 9.0)], grid[best_key]))
""")

code(r"""
print("=== the three pre-registered predictions ===\n")

# 1. score improves as the radius falls
r_at_base = {r: grid[(6.0, r)].score for r in RADII if (6.0, r) in grid}
best_r = max(r_at_base, key=r_at_base.get)
p1 = best_r < 9.0
print(f"1. tighter radius helps:  best radius at sep=6.0 is {best_r} "
      f"({r_at_base[best_r]:.4f} vs {r_at_base[9.0]:.4f} at 9.0)")
print(f"   -> {'CONFIRMED' if p1 else 'FALSIFIED'}")
print(f"   {r_at_base}")

# 2. smaller separation raises recall
rec = {s: grid[(s, RADII[0])].summary['node_recall'] for s in SEPS if (s, RADII[0]) in grid}
p2 = rec[min(SEPS)] > rec[max(SEPS)] + 0.005
print(f"\n2. smaller min_sep raises recall: {rec}")
print(f"   -> {'CONFIRMED' if p2 else 'FALSIFIED'}")
if not p2:
    print("   The missing GT nodes are NOT suppressed by non-max — they are invisible in")
    print("   this channel. Detector tuning cannot reach them; stop trying.")

# 3. the two interact
best_r_per_sep = {s: max(RADII, key=lambda r: grid[(s, r)].score)
                  for s in SEPS if all((s, r) in grid for r in RADII)}
p3 = len(set(best_r_per_sep.values())) > 1
print(f"\n3. radius and separation interact: best radius per sep = {best_r_per_sep}")
print(f"   -> {'CONFIRMED' if p3 else 'FALSIFIED'}")
""")

md("""## 4. What the leaderboard will say — before spending a submission

We hold ground truth for all four test datasets (they also appear in `train/`). So the
test score is computable locally. Two things this buys:

1. **An expected leaderboard value.** If the submitted score comes back close to this, the
   visible test set is the scored set — which is the leak in `notes/05` §0, and the thing
   to report to the organisers.
2. **A check on the subset.** The four test datasets are two `44b6_` with ~50 GT edges each
   and two `6bba_` carrying ~95 % of the weight. A config tuned on a 60-dataset subset
   might not be the best config for those four specifically.
""")

code(r"""
SEP, RAD = best_key
FINAL = Config(det_threshold=0.15, min_separation_um=SEP, link_radius_um=RAD,
               budget_fill=None)
print(f"scoring the 4 test datasets locally with min_sep={SEP}, radius={RAD}\n")

rows = []
for n in test_names:
    t0 = time.time()
    pred = predict_dataset(TEST / n, FINAL, verbose=False)
    gt = read_geff(TRAIN / f"{n}.geff")          # available: test names appear in train
    scale = read_scale(TEST / f"{n}.zarr")
    n_est = estimated_total_nodes(TRAIN / f"{n}.zarr")   # budget from the TRAIN copy
    c = purescore.count_edges(pred.t, pred.zyx, pred.edges, gt.t, gt.zyx, gt.edges,
                              scale=scale)
    row = purescore.per_sample(c, n_est if n_est else float("nan"), gt.n_nodes,
                               n_gt_divisions=gt.n_divisions)
    rows.append(row)
    w = c.tp + c.fp + c.fn
    print(f"  {n:<28} J={row['edge_jaccard']:.4f} adj={row['adj_edge_jaccard']:.4f} "
          f"TP/FP/FN={c.tp}/{c.fp}/{c.fn} recall={row['node_recall']:.3f} "
          f"ratio={row['total_node_ratio']:+.2f} ({time.time()-t0:.0f}s)", flush=True)

summ = purescore.summarise(rows)
print(f"\nPREDICTED LEADERBOARD SCORE = {summ['score']:.4f}")
print(f"  edge_jaccard={summ['edge_jaccard']:.4f}  adj={summ['adj_edge_jaccard']:.4f}")
print(f"  division term = {summ['division_jaccard']} (we predict no divisions, so 0 or n/a)")
print("\nSubmit the honest prediction and compare. A close match means the visible test")
print("set is the scored set — report that to the organisers, and do not read the")
print("leaderboard position as signal about the real problem.")
""")

code(r"""
graphs = {n: predict_dataset(TEST / n, FINAL, verbose=False) for n in test_names}
csv = build_submission(graphs, WORK / "submission.csv")
problems = validate_submission(csv, expected_datasets=test_names)
print("\nREADY TO SUBMIT" if not problems else f"\nFIX {len(problems)} PROBLEM(S) FIRST")

payload = {
    "subset": SUBSET,
    "subset_size": len(SUBSET),
    "incumbent_on_subset": base.score,
    "grid": {f"sep{s}_rad{r}": {k: (None if isinstance(v, float) and v != v else v)
                                for k, v in g.summary.items()}
             for (s, r), g in grid.items()},
    "best": {"min_separation_um": SEP, "link_radius_um": RAD, "score": grid[best_key].score},
    "predicted_leaderboard": {k: (None if isinstance(v, float) and v != v else v)
                              for k, v in summ.items()},
    "test_rows": [{k: (None if isinstance(v, float) and v != v else v) for k, v in r.items()}
                  for r in rows],
}
(WORK / "linking_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/linking_results.json - send it back with the log.")
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
