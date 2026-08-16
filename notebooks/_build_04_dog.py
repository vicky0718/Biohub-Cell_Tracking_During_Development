"""Build notebooks/04_dog_detector.ipynb."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "04_dog_detector.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Experiment #3 — the detector

`notes/09` §2 is the reason this notebook exists. Tightening the detection window took
node recall from 0.895 to 0.976 and the **score from 0.5790 to 0.3449**. The arithmetic:

- extra detections spent: **1,864,575**
- extra ground-truth nodes found: **~3,264**
- → **571 spurious detections per additional GT node found**

Our intensity detector can reach any recall you like, but only by spraying, and the metric
charges for that twice — the linker drowns in distractors (−0.169 edge Jaccard) and the
node budget starts biting (×0.826 at 2.7× over budget).

**So the quantity that matters is recall per detection spent, and that is a property of the
detector, not of any threshold.** `notes/08` §2 has the measured target: the public
rule-based pipeline scored **CV 0.682 with plain DoG + Hungarian** and **0.824 multi-scale**,
where we score 0.5790 on this subset. Same linker, same match radius, same metric.

## Pre-registered, before anything runs

1. **DoG beats intensity at a matched node count.** This is the claim, and the matched
   count is what makes it a claim about the *detector* rather than about density.
   *Falsified if* DoG scores no better than intensity when both emit ~the same number of
   nodes.
2. **Multi-scale beats single-scale.** Their measurement: +0.040 LB. *Falsified if* the
   scale-space maximum is within noise of the best single pair.
3. **The optimum sits near the node budget, not above it.** `notes/09` §4 found this subset
   runs +0.121 over budget already. *Falsified if* the best arm is far over budget and
   still wins.

Promotion still requires a pooled gain with **no fold regression**, and folds are now
**leave-one-embryo-out** (`notes/07` §3) — the shift the leaderboard actually applies.

> No `tracksdata`, and note for later: a *scored submission* cannot `pip install` at all
> (`notes/07` §10). This notebook is research and runs with internet on; the eventual
> submission notebook must install nothing.
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
import sys, os, time, json, hashlib, itertools
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")

def find_dir(is_match, roots, max_depth=5):
    # Breadth-first search of the Kaggle mounts. Never descends into .zarr/.geff,
    # which hold thousands of chunk files.
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
from pipeline.classical import Config, make_predictor, predict_dataset

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data.")
TRAIN, TEST = COMP / "train", COMP / "test"

CACHE = WORK / "cache"; CACHE.mkdir(exist_ok=True, parents=True)
train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
print("project:", REPO, "| train:", len(train_names))
""")

md("""## 1. The same fixed subset as `03`

Identical selection rule, so the incumbent number carries over directly: **0.5790**.
Changing the eval set between experiments would make the comparison meaningless.

Folds are now **leave-one-embryo-out**. That gives two folds, which is thin — but `03` ran
under the old five-way hash split and its one regressing fold measured within-embryo
variance, which is not the shift we are graded on.
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

h = Harness(data_dir=TRAIN, cache_dir=CACHE)          # fold_by="embryo" by default
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
print(f"subset: {len(SUBSET)} datasets;  folds (leave-one-embryo-out): "
      + ", ".join(f"{f}:{len(v)}" for f, v in sorted(folds.items())))
assert len(SUBSET) == 60, "subset drifted from 03 — the comparison would be invalid"

INCUMBENT_SUBSET_SCORE = 0.5790   # 03: intensity, thr 0.15, sep 6.0, radius 9.0
""")

md("""## 2. Calibrate DoG to a matched node count

`notes/09` §2 showed that comparing two detectors at different densities measures density,
not detector quality. So before scoring anything, find the `dog_rel_threshold` that makes
DoG emit roughly the **same number of nodes** as the incumbent on a few datasets. Only then
is a score difference attributable to the detector.

This runs on 3 datasets and a handful of frames — it is calibration, not measurement.
""")

code(r"""
import zarr
from pipeline.classical import detect_frame, detect_frame_dog

CAL = SUBSET[:3]
FRAMES = [0, 25, 50, 75]
SCALE_UM = (1.625, 0.40625, 0.40625)

def frame_counts(name, cfg, det):
    grp = zarr.open_group(str(TRAIN / f"{name}.zarr"), mode="r")
    arr = grp["0"]
    attrs = dict(grp.attrs)
    q = attrs.get("image_statistics", {}).get("quantiles", {})
    lo, hi = float(q.get("0.001", 0.0)), float(q.get("0.999", 1.0))
    dz, dy, dx = cfg.downsample
    vox = (SCALE_UM[0]*dz, SCALE_UM[1]*dy, SCALE_UM[2]*dx)
    out = []
    for t in FRAMES:
        vol = np.asarray(arr[t, ::dz, ::dy, ::dx]).astype(np.float32)
        vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0.0, None)
        out.append(len(det(vol, vox, cfg)[0]))
    return out

base_cfg = Config(det_threshold=0.15, min_separation_um=6.0)
base_counts = [c for n in CAL for c in frame_counts(n, base_cfg, detect_frame)]
target = float(np.mean(base_counts))
print(f"incumbent detections/frame: mean {target:.0f}  {base_counts}")

print("\ncalibrating dog_rel_threshold to match that count:")
rows = []
for thr in (0.005, 0.01, 0.02, 0.04, 0.08):
    cfg = Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=thr,
                 dog_scales=[(1.5, 4.0), (2.5, 6.0)])
    cnt = [c for n in CAL for c in frame_counts(n, cfg, detect_frame_dog)]
    rows.append((thr, float(np.mean(cnt))))
    print(f"  rel_threshold={thr:<6} mean {np.mean(cnt):>7.0f}/frame  "
          f"({np.mean(cnt)/target:.2f}x incumbent)")

MATCHED_THR = min(rows, key=lambda r: abs(r[1] - target))[0]
print(f"\n-> matched threshold = {MATCHED_THR} "
      f"({dict(rows)[MATCHED_THR]/target:.2f}x the incumbent node count)")
""")

md("""## 3. The arms

Every arm is scored by the same harness on the same 60 datasets. The first two are the
paired test of prediction 1; the rest explore scales once the density question is settled.
""")

code(r"""
ARMS = {
    "intensity_incumbent": Config(det_threshold=0.15, min_separation_um=6.0),
    "dog_matched":         Config(detector="dog", min_separation_um=6.0,
                                  dog_rel_threshold=MATCHED_THR,
                                  dog_scales=[(1.5, 4.0), (2.5, 6.0)]),
    "dog_singlescale":     Config(detector="dog", min_separation_um=6.0,
                                  dog_rel_threshold=MATCHED_THR, dog_scales=None),
    "dog_multiscale3":     Config(detector="dog", min_separation_um=6.0,
                                  dog_rel_threshold=MATCHED_THR,
                                  dog_scales=[(1.0, 3.0), (1.5, 4.0), (2.5, 6.0)]),
    # the ball footprint is continuous, unlike the box one 03 tripped over
    "dog_sep4.5":          Config(detector="dog", min_separation_um=4.5,
                                  dog_rel_threshold=MATCHED_THR,
                                  dog_scales=[(1.5, 4.0), (2.5, 6.0)]),
    "dog_sep7.5":          Config(detector="dog", min_separation_um=7.5,
                                  dog_rel_threshold=MATCHED_THR,
                                  dog_scales=[(1.5, 4.0), (2.5, 6.0)]),
}

results = {}
for name, cfg in ARMS.items():
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg), arm=name, names=SUBSET, verbose=False)
    results[name] = res
    s = res.summary
    n_nodes = sum(r["num_pred_nodes"] for r in res.rows.values())
    print(f"{name:<22} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  nodes={n_nodes:>10,}  "
          f"recall/Mnode={1e6*s['node_recall']/n_nodes:.3f}  ({time.time()-t0:.0f}s)",
          flush=True)
""")

code(r"""
print("=== summary, sorted by score ===")
print(f"{'arm':<22} {'SCORE':>8} {'edge_J':>8} {'recall':>7} {'nodes':>11} {'vs budget':>10}")
for name, res in sorted(results.items(), key=lambda kv: -kv[1].score):
    s = res.summary
    n_nodes = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    print(f"{name:<22} {s['score']:>8.4f} {s['edge_jaccard']:>8.4f} "
          f"{s['node_recall']:>7.3f} {n_nodes:>11,} {np.mean(ratios):>+10.3f}")

print(f"\nincumbent on this subset from 03: {INCUMBENT_SUBSET_SCORE:.4f}")
print(f"the measured target (public rule-based, plain DoG + Hungarian): 0.682\n")

base = results["intensity_incumbent"]
best_name = max(results, key=lambda k: results[k].score)
print(gate(base, results[best_name]))
print(f"\nbest arm: {best_name}")
""")

code(r"""
print("=== the three pre-registered predictions ===\n")

d_score = results["dog_matched"].score - results["intensity_incumbent"].score
n_i = sum(r["num_pred_nodes"] for r in results["intensity_incumbent"].rows.values())
n_d = sum(r["num_pred_nodes"] for r in results["dog_matched"].rows.values())
p1 = d_score > 0
print(f"1. DoG beats intensity at a matched node count: "
      f"{results['dog_matched'].score:.4f} vs {results['intensity_incumbent'].score:.4f} "
      f"({d_score:+.4f})")
print(f"   node counts {n_d:,} vs {n_i:,} ({n_d/n_i:.2f}x — matched if near 1.0)")
print(f"   -> {'CONFIRMED' if p1 else 'FALSIFIED'}")
if abs(n_d/n_i - 1) > 0.25:
    print("   !! counts are NOT matched; this arm measures density as well as detector.")

ms, ss = results["dog_multiscale3"].score, results["dog_singlescale"].score
p2 = ms > ss
print(f"\n2. multi-scale beats single-scale: {ms:.4f} vs {ss:.4f} ({ms-ss:+.4f})")
print(f"   -> {'CONFIRMED' if p2 else 'FALSIFIED'}")

best = results[best_name]
br = np.mean([r["total_node_ratio"] for r in best.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]])
p3 = br < 0.5
print(f"\n3. the optimum sits near the budget: best arm mean node ratio {br:+.3f}")
print(f"   -> {'CONFIRMED' if p3 else 'FALSIFIED'}")
if not p3:
    print("   The best arm is well over budget and still wins — the multiplier is weaker")
    print("   than notes/09 §2 implied, and density deserves its own sweep.")

payload = {"subset": SUBSET, "matched_threshold": MATCHED_THR,
           "incumbent_subset_score": INCUMBENT_SUBSET_SCORE,
           "arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                        for kk, vv in v.summary.items()} for k, v in results.items()},
           "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
                     for k, v in results.items()},
           "best": best_name}
(WORK / "dog_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/dog_results.json — send it back with the log.")
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
