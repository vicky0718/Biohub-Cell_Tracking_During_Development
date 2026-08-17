"""Build notebooks/05_density.ipynb."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "05_density.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Experiment #4 — spend the node budget, don't overrun it

`04` promoted the DoG detector: **0.5790 → 0.6760**, +0.0970 pooled, positive in every
fold. `notes/10` then found ~0.05 sitting unclaimed in the budget term.

| arm | edge_J | ratio | multiplier | score |
|---|---|---|---|---|
| `dog_sep4.5` | **0.7195** | +0.528 | 0.947 | 0.6671 |
| `dog_matched` (sep 6.0) | 0.7053 | +0.239 | 0.976 | **0.6760** |
| `dog_sep7.5` | 0.6623 | −0.018 | 1.002 | 0.6511 |

**`dog_sep4.5` has the best detection+linking quality we have measured — edge Jaccard
0.7195 — and loses 5.3 % of it to over-detection.** Brought to budget it would score about
**0.72**, past the 0.682 the public rule-based pipeline reached with plain DoG.

`notes/05` §2 measured `budget_fill` as *harmful* and I turned it off — with the caveat
that any change pushing node counts up must re-check the ratio. DoG is that change: the
ratio went from −0.111 to +0.24…+0.53. So the cap comes back for testing, for exactly the
reason it was dropped.

`notes/10` §4: `dog_rel_threshold` is a weak density knob (16× change moved density 15 %).
`min_separation_um` is the strong one, and the DoG ball footprint is continuous, unlike the
four-valued box footprint `03` tripped over.

## Pre-registered

1. **Capping to the budget beats not capping**, on the same detector config. *Falsified if*
   `budget_fill=1.0` scores no better than `None` — which would mean the cap costs more
   real detections than the multiplier saves, as it did in `notes/05` §2 when we were
   *under* budget.
2. **The score peaks at a separation between 4.5 and 6.0 µm.** Quality rises as separation
   falls (0.6623 → 0.7053 → 0.7195) while the budget penalty also rises; the product should
   have an interior maximum. *Falsified if* the score is monotonic across the sweep.
3. **Three scales beats two at matched density.** In `04` the 3-scale arm tied the winner
   on edge Jaccard (0.7027 vs 0.7053) while emitting twice the nodes, because I reused the
   2-scale calibration. *Falsified if* it does not beat 2 scales once its own density is
   matched.
""")

code(r"""
# --- deps -------------------------------------------------------------------
import subprocess, sys

def pip_install(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode

print("installing geff + zarr ...")
pip_install(["geff", "zarr"])
""")

code(r"""
import sys, os, time, json, hashlib
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")

def find_dir(is_match, roots, max_depth=5):
    # Breadth-first over the Kaggle mounts; never descends into .zarr/.geff.
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
    raise SystemExit("Could not find harness/ and pipeline/. Add the project dataset as an input.")
sys.path.insert(0, str(REPO))

from harness import Harness, gate
from pipeline.classical import Config, estimated_total_nodes, make_predictor

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data.")
TRAIN, TEST = COMP / "train", COMP / "test"

CACHE = WORK / "cache"; CACHE.mkdir(exist_ok=True, parents=True)
train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
print("project:", REPO, "| train:", len(train_names))

# 04 printed "leave-one-embryo-out" as hardcoded text while actually running the five-way
# hash split, because the uploaded snapshot predated fold_by (notes/10 section 2). Never
# claim the setup — derive it, and fail loudly.
import inspect
HAS_FOLD_BY = "fold_by" in inspect.signature(Harness.__init__).parameters
print("harness supports fold_by:", HAS_FOLD_BY)
if not HAS_FOLD_BY:
    raise SystemExit(
        "This snapshot of harness/ predates fold_by, so folds would be the five-way hash\n"
        "split, not leave-one-embryo-out. Re-upload the current repo as the Kaggle dataset\n"
        "before running — otherwise this notebook measures the wrong generalisation."
    )
""")

code(r"""
SUBSET_SIZE = 60

def stable_key(n):
    return int(hashlib.sha1(n.encode()).hexdigest(), 16)

by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)

SUBSET = []
for pfx, names in sorted(by_prefix.items()):
    k = round(SUBSET_SIZE * len(names) / len(train_names))
    SUBSET += sorted(names, key=stable_key)[:k]
SUBSET = sorted(SUBSET)
assert len(SUBSET) == 60, f"subset drifted from 03/04 ({len(SUBSET)}) — comparison invalid"

h = Harness(data_dir=TRAIN, cache_dir=CACHE)      # fold_by="embryo"
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
print("folds:", {f: len(v) for f, v in sorted(folds.items())})

# derive, don't claim
prefixes_per_fold = {f: {n.split("_")[0] for n in v} for f, v in folds.items()}
is_embryo_split = all(len(p) == 1 for p in prefixes_per_fold.values())
print("each fold is a single embryo:", is_embryo_split, prefixes_per_fold)
assert is_embryo_split, "folds are NOT leave-one-embryo-out — check the uploaded harness"

# the per-dataset budgets, needed for budget_fill and for reporting the ratio
BUDGETS = {n: estimated_total_nodes(TRAIN / f"{n}.zarr") for n in SUBSET}
BUDGETS = {k: v for k, v in BUDGETS.items() if v}
print(f"budgets read for {len(BUDGETS)}/{len(SUBSET)} datasets; "
      f"subset total {sum(BUDGETS.values()):,.0f} nodes")

BEST_04 = 0.6760   # dog_matched, sep 6.0, 2 scales
SCALES2 = [(1.5, 4.0), (2.5, 6.0)]
SCALES3 = [(1.0, 3.0), (1.5, 4.0), (2.5, 6.0)]
""")

md("""## 1. Does capping to the budget pay now that we are over it?

The paired test of prediction 1, on the arm with the best measured quality (`sep 4.5`,
edge Jaccard 0.7195, ratio +0.528) and on the current champion (`sep 6.0`).
""")

code(r"""
def run(name, cfg):
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg, budgets=BUDGETS), arm=name, names=SUBSET,
                     verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    print(f"{name:<26} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  nodes={n:>10,}  ratio={np.mean(ratios):+.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return res

results = {}
for sep in (4.5, 6.0):
    for fill in (None, 1.0):
        tag = f"sep{sep}_{'cap' if fill else 'nocap'}"
        results[tag] = run(tag, Config(detector="dog", min_separation_um=sep,
                                       dog_rel_threshold=0.005, dog_scales=SCALES2,
                                       budget_fill=fill))
""")

code(r"""
print("=== prediction 1: does the cap pay? ===")
for sep in (4.5, 6.0):
    a, b = results[f"sep{sep}_nocap"], results[f"sep{sep}_cap"]
    print(f"\n-- min_separation {sep} um --")
    print(f"   no cap {a.score:.4f}   cap {b.score:.4f}   delta {b.score-a.score:+.4f}")
    print(gate(a, b))
""")

md("""## 2. Where is the separation optimum?

Prediction 2. The ball footprint is continuous, so unlike `03` these are genuinely
distinct settings. Run with whichever cap setting §1 favoured.
""")

code(r"""
USE_CAP = 1.0 if results["sep4.5_cap"].score > results["sep4.5_nocap"].score else None
print(f"using budget_fill={USE_CAP} for the sweep (from section 1)\n")

for sep in (3.5, 5.0, 5.5, 6.5):
    tag = f"sweep_sep{sep}"
    results[tag] = run(tag, Config(detector="dog", min_separation_um=sep,
                                   dog_rel_threshold=0.005, dog_scales=SCALES2,
                                   budget_fill=USE_CAP))
""")

code(r"""
sweep = {}
for sep in (3.5, 4.5, 5.0, 5.5, 6.0, 6.5):
    tag = f"sweep_sep{sep}"
    if tag not in results:
        tag = f"sep{sep}_{'cap' if USE_CAP else 'nocap'}"
    if tag in results:
        sweep[sep] = results[tag].score

print("=== prediction 2: separation sweep ===")
for sep, sc in sorted(sweep.items()):
    print(f"  min_separation {sep:<5} -> {sc:.4f}")
vals = [sweep[s] for s in sorted(sweep)]
interior = vals.index(max(vals)) not in (0, len(vals) - 1)
print(f"\n  peak is interior: {interior}  -> "
      f"{'CONFIRMED' if interior else 'FALSIFIED (monotonic — extend the sweep)'}")
BEST_SEP = max(sweep, key=sweep.get)
print(f"  best separation: {BEST_SEP} at {sweep[BEST_SEP]:.4f}")
""")

md("""## 3. Three scales, with its own density calibration

Prediction 3. In `04` the 3-scale arm reused the 2-scale threshold and emitted twice the
nodes; its edge Jaccard tied the winner. Here it gets its own separation, chosen so its
node count lands near the 2-scale champion's.
""")

code(r"""
import zarr
from pipeline.classical import detect_frame_dog
SCALE_UM = (1.625, 0.40625, 0.40625)

def counts_per_frame(name, cfg, frames=(0, 33, 66)):
    grp = zarr.open_group(str(TRAIN / f"{name}.zarr"), mode="r")
    arr = grp["0"]; attrs = dict(grp.attrs)
    q = attrs.get("image_statistics", {}).get("quantiles", {})
    lo, hi = float(q.get("0.001", 0.0)), float(q.get("0.999", 1.0))
    dz, dy, dx = cfg.downsample
    vox = (SCALE_UM[0]*dz, SCALE_UM[1]*dy, SCALE_UM[2]*dx)
    out = []
    for t in frames:
        vol = np.asarray(arr[t, ::dz, ::dy, ::dx]).astype(np.float32)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, None)
        out.append(len(detect_frame_dog(vol, vox, cfg)[0]))
    return out

CAL = SUBSET[:3]
ref = Config(detector="dog", min_separation_um=BEST_SEP, dog_rel_threshold=0.005,
             dog_scales=SCALES2)
target = float(np.mean([c for n in CAL for c in counts_per_frame(n, ref)]))
print(f"2-scale at sep {BEST_SEP}: {target:.0f} detections/frame — the target\n")

best_sep3, best_gap = None, None
for sep in (BEST_SEP, BEST_SEP + 1.0, BEST_SEP + 2.0, BEST_SEP + 3.0):
    cfg = Config(detector="dog", min_separation_um=sep, dog_rel_threshold=0.005,
                 dog_scales=SCALES3)
    m = float(np.mean([c for n in CAL for c in counts_per_frame(n, cfg)]))
    print(f"  3-scale at sep {sep:<5} -> {m:>6.0f}/frame ({m/target:.2f}x)")
    if best_gap is None or abs(m - target) < best_gap:
        best_sep3, best_gap = sep, abs(m - target)
print(f"\n-> 3-scale density-matched at min_separation {best_sep3}")

results["scales3_matched"] = run("scales3_matched",
    Config(detector="dog", min_separation_um=best_sep3, dog_rel_threshold=0.005,
           dog_scales=SCALES3, budget_fill=USE_CAP))
""")

code(r"""
two = results[f"sweep_sep{BEST_SEP}"] if f"sweep_sep{BEST_SEP}" in results \
      else results[f"sep{BEST_SEP}_{'cap' if USE_CAP else 'nocap'}"]
three = results["scales3_matched"]
print("=== prediction 3: 3 scales vs 2, at matched density ===")
print(f"  2 scales {two.score:.4f}   3 scales {three.score:.4f}   "
      f"delta {three.score-two.score:+.4f}")
print(f"  -> {'CONFIRMED' if three.score > two.score else 'FALSIFIED'}")
print()
print(gate(two, three))

print("\n" + "=" * 66)
best_name = max(results, key=lambda k: results[k].score)
print(f"BEST THIS RUN: {best_name} -> {results[best_name].score:.4f}")
print(f"  04 champion (dog_matched):        {BEST_04:.4f}")
print(f"  03 incumbent (intensity):         0.5790")
print(f"  public rule-based, plain DoG:     0.682")
print(f"  public rule-based, multi-scale:   0.824")

payload = {"subset": SUBSET, "best": best_name,
           "arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                        for kk, vv in v.summary.items()} for k, v in results.items()},
           "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
                     for k, v in results.items()},
           "best_separation": BEST_SEP, "use_cap": USE_CAP, "sep3_matched": best_sep3}
(WORK / "density_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/density_results.json — send it back with the log.")
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
