"""Build notebooks/claude_divdata.ipynb — how much division training data actually exists.

`notes/42` §3: the score is `edge_J + 0.1 * div_J`, we sit at `div_J = 0.0645`, so
divisions contribute 0.0065 of an available 0.1. **Headroom 0.094 against a 0.027 gap to
bronze** — the only quantity in this project with room bigger than the target.

Nothing has ever been trained on divisions. They fall out of the ILP's three global
weights, which were swept for total score and never for `div_J`. Meanwhile the published
edge model is at `test_acc 0.99988` / `test_recall 0.9755` after 400 epochs on all 199
datasets — its own metadata, read from the checkpoint — so the edge term is close to
saturated and more edge training has little to bite on.

So the training data worth having is the 199 datasets' **division events**, and this
notebook asks whether there are enough of them to train on before a GPU is spent finding
out. It is CPU-only and reads ground truth alone: no model, no prediction, no scoring.

It also mounts `kkunizaw/biohub-zh001r`, the one external ZebraHub mirror still reachable
(`biohub-zmnscrops` returns 403), and reports what is actually inside it — Stage 0 of the
plan in `keen-munching-lerdorf`, answered with facts rather than assumption.

The failure that killed every prior learned arm was **temporal incoherence** (`notes/21`):
a per-frame detector finds different-but-plausible cells frame to frame, so edges between
them fail to match. A division classifier does not have that failure mode. It does not
replace detection; it puts a per-node prior on top of detections that already exist and are
already coherent, and feeds the ILP's `division_weight` per node instead of globally.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_divdata.ipynb")
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Is there enough division data to train on?

```
0.901  submitted        0.926 bronze     0.944 gold
score = adjusted_edge_jaccard + 0.1 * division_jaccard      (max 1.1)
```

`notes/42` measured where the room is:

```
edge_J 0.9449   div_J 0.0645
divisions contribute  0.0065  of an available  0.1000
                          headroom             0.0935
```

**We recover about 6% of divisions**, and the gap to bronze is 0.027. Divisions hold three
and a half times that. Nothing in this project has ever been *trained* on them — they fall
out of the ILP's three global weights, swept for total score and never once for `div_J`.

The edge model, by contrast, is close to saturated. Read straight out of the published
checkpoint's own metadata:

```
test_acc 0.99988   test_recall 0.9755   best_score 0.9835
epoch 402,  train_datasets 199   (i.e. all of them)
```

More edge training has little left to bite on, and the one external corpus still reachable
is a single embryo. So the training data worth having is the **199 datasets' own division
events** — which we have, in full, and have never touched.

## What this notebook does

CPU only. Ground truth only. No model, no prediction, no scoring. It reads all 199 `.geff`
graphs and answers the questions that decide whether a division model is worth building and
how it must be built:

* how many division events exist in total, and per dataset
* the class balance a classifier would face
* whether divisions are uniform in time, or concentrated
* how far the two embryos differ, since every stratification mistake in this project has
  come from assuming they do not

It also mounts `kkunizaw/biohub-zh001r` — the only external ZebraHub mirror still reachable,
`biohub-zmnscrops` now returns 403 — and prints what is actually in it.

## Pre-registered predictions

Thresholds are set where the *decision* changes, not at the noise floor. `notes/42`: a
prediction whose threshold sits below the measurement's resolution is a coin flip with a
paper trail.

1. **More than 10,000 division events across the 199 datasets.** Below that a per-node
   classifier is not trainable and the direction dies here, cheaply.
2. **The positive rate is under 5%.** If divisions were common the ILP's global weight
   would already be near-optimal; a strong imbalance is what makes a per-node prior worth
   more than one number.
3. **Division counts vary more than 3x across datasets.** If they do, a single global
   `division_weight` cannot be right everywhere and per-dataset adaptivity is on the table.
4. **Divisions are concentrated in time, not uniform** — more than 60% of them falling in
   fewer than half the frames. Development slows; a model that knows the frame has a free
   feature.
5. **`zh001r` carries both a volume and per-node labels**, i.e. Stage 0 of the external-data
   plan passes on inspection rather than on assumption.
""")

code(r"""
import os, subprocess, sys, time, json
from pathlib import Path

T0 = time.time()
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

def find_dir(is_match, roots, max_depth=6):
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

# The pack is identified by its wheels, not by repo/+weights/ — claude_secondary v1
# resolved a second model dataset as the pack because find_dir walks alphabetically.
PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir()
                and (p / "wheels").is_dir() and "seed314159" not in str(p),
                ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
ZH = find_dir(lambda p: "zh001r" in p.name.lower(), ["/kaggle/input"], max_depth=3)
for label, val in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                   ("zebrahub", ZH)):
    print(f"  {label:<14} {val}")
missing = [l for l, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP))
           if v is None]
if missing:
    raise SystemExit(f"not mounted: {missing}")
TRAIN = COMP / "train"

# geff needs only networkx + zarr; no torch, no GPU, nothing from the wheelhouse.
r = sh(sys.executable, "-m", "pip", "install", "-q", "--no-index",
       f"--find-links={PACK/'wheels'}",
       *[str(p) for p in sorted((PACK / "wheels").glob("*.whl"))])
print("pack wheels", "ok" if r.returncode == 0 else "FAILED", f"({time.time()-T0:.0f}s)")
if r.returncode != 0:
    print(r.stdout[-1500:]); print(r.stderr[-1500:])

sys.path.insert(0, str(REPO))
import numpy as np

# Load harness/tracks.py DIRECTLY, bypassing harness/__init__.py.
#
# v1 died here. The pack wheels replace numpy underneath an interpreter that has already
# imported it, so the image's scipy no longer matches -- and `import harness.tracks` runs
# `harness/__init__.py`, which imports `purescore`, which imports `scipy.sparse`. Every
# other notebook in this project dodges that by doing its work in a subprocess with a
# fresh interpreter. This one does not need to: it never scores anything, so it never
# needs scipy. Loading the one module by path is the smaller, more precise fix, and it
# keeps the notebook free of the .format() brace hazard a worker template carries.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_tracks", Path(REPO) / "harness" / "tracks.py")
_tracks = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tracks)
read_geff, read_estimated_nodes = _tracks.read_geff, _tracks.read_estimated_nodes
print("ready", flush=True)
""")

md("""## 1. Read all 199 ground-truth graphs""")

code(r"""
names = sorted(p.stem for p in TRAIN.glob("*.zarr")
               if (TRAIN / (p.stem + ".geff")).exists())
n44 = sum(n.startswith("44b6") for n in names)
print(f"{len(names)} datasets: {n44} x 44b6, {len(names)-n44} x 6bba", flush=True)

STATS = {}
FRAME_HIST = np.zeros(128, np.int64)      # divisions per frame index, pooled
NODE_HIST = np.zeros(128, np.int64)       # nodes per frame index, pooled
t0 = time.time()
for i, name in enumerate(names):
    gt = read_geff(TRAIN / (name + ".geff"))
    out_deg = np.bincount(gt.edges[:, 0], minlength=len(gt.t)) if len(gt.edges) \
        else np.zeros(len(gt.t), np.int64)
    div_idx = np.flatnonzero(out_deg >= 2)
    dt = gt.t[div_idx]
    STATS[name] = {
        "nodes": int(len(gt.t)), "edges": int(len(gt.edges)),
        "divisions": int(len(div_idx)),
        "frames": int(gt.t.max() + 1) if len(gt.t) else 0,
        "div_per_frame": [int(x) for x in np.bincount(dt, minlength=100)[:100]],
        "n_total": float(read_estimated_nodes(TRAIN / (name + ".geff"))),
    }
    FRAME_HIST[:100] += np.bincount(dt, minlength=100)[:100]
    NODE_HIST[:100] += np.bincount(gt.t, minlength=100)[:100]
    if (i + 1) % 25 == 0 or i + 1 == len(names):
        print(f"  {i+1:>3}/{len(names)}  {int(time.time()-t0)}s", flush=True)

(WORK / "divdata.json").write_text(json.dumps(
    {"stats": STATS, "frame_hist": FRAME_HIST[:100].tolist(),
     "node_hist": NODE_HIST[:100].tolist(), "names": names}, default=float))
print("wrote divdata.json", flush=True)
""")

md("""## 2. What is inside the one reachable external dataset""")

code(r"""
if ZH is None:
    print("zh001r not mounted — prediction 5 cannot be graded")
    ZH_FILES = []
else:
    ZH_FILES = sorted(p for p in Path(ZH).rglob("*") if p.is_file())[:400]
    tot = sum(p.stat().st_size for p in ZH_FILES)
    print(f"{ZH}\n  {len(ZH_FILES)} files, {tot/1e9:.2f} GB")
    # Group by extension so the shape of the thing is visible in a few lines.
    from collections import Counter
    for ext, k in Counter(p.suffix or "<none>" for p in ZH_FILES).most_common(12):
        ex = next(p for p in ZH_FILES if (p.suffix or "<none>") == ext)
        print(f"  {ext:<10} x{k:<5} e.g. {ex.relative_to(ZH)}  {ex.stat().st_size/1e6:.1f} MB")
    for d in sorted({p.parent for p in ZH_FILES})[:12]:
        print(f"  dir: {d.relative_to(ZH)}")
    # A .zarr directory is a directory, not a file, so look for those separately.
    zarrs = [p for p in Path(ZH).rglob("*") if p.is_dir() and p.suffix == ".zarr"]
    print(f"  {len(zarrs)} .zarr volumes" + (f": {zarrs[0].relative_to(ZH)}" if zarrs else ""))
    for p in ZH_FILES:
        if p.suffix in (".json", ".csv", ".txt") and p.stat().st_size < 200_000:
            print(f"\n--- {p.relative_to(ZH)} (first 400 chars) ---")
            print(p.read_text(errors="replace")[:400])
            break
""")

md("""## 3. The five predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "divdata.json").read_text())
S, names = D["stats"], D["names"]
FH = np.array(D["frame_hist"], float); NH = np.array(D["node_hist"], float)

divs = np.array([S[n]["divisions"] for n in names], float)
nodes = np.array([S[n]["nodes"] for n in names], float)
TOTAL_D, TOTAL_N = float(divs.sum()), float(nodes.sum())
rate = TOTAL_D / max(TOTAL_N, 1)

print(f"{len(names)} datasets")
print(f"  nodes      {TOTAL_N:>12,.0f}")
print(f"  divisions  {TOTAL_D:>12,.0f}   ({rate:.2%} of nodes)")
print(f"  per dataset: min {divs.min():.0f}  median {np.median(divs):.0f}  "
      f"max {divs.max():.0f}  mean {divs.mean():.1f}")

a = np.array([S[n]["divisions"] for n in names if n.startswith("44b6")], float)
b = np.array([S[n]["divisions"] for n in names if not n.startswith("44b6")], float)
print(f"  44b6 median {np.median(a):.0f} (n={len(a)})   "
      f"6bba median {np.median(b):.0f} (n={len(b)})")

print("\ndivisions by frame (pooled over all datasets, 10-frame bins)")
for lo in range(0, 100, 10):
    d10, n10 = FH[lo:lo+10].sum(), NH[lo:lo+10].sum()
    bar = "#" * int(60 * d10 / max(FH.max() * 10, 1))
    print(f"  t {lo:>3}-{lo+9:<3} {d10:>7,.0f} div  {d10/max(n10,1):>7.2%} of nodes  {bar}")

print("\n" + "=" * 84)
print("PREDICTION GRADING")
print("=" * 84)

print("\n1. more than 10,000 division events across the 199 datasets")
ok1 = TOTAL_D > 10_000
print(f"   {TOTAL_D:,.0f}  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   Not enough to train a per-node classifier. The direction dies here, for")
    print("   the price of one CPU notebook, which is the point of asking first.")

print("\n2. the positive rate is under 5%")
ok2 = rate < 0.05
print(f"   {rate:.2%}  ->  {'PASS' if ok2 else 'FAIL'}")
print("   " + ("Strong imbalance: the loss needs explicit weighting, and a per-node prior"
               " is worth more than one global number."
               if ok2 else
               "Divisions are common enough that the ILP's global weight may already be"
               " near-optimal, which weakens the case for a model."))

print("\n3. division counts vary more than 3x across datasets")
lo, hi = np.percentile(divs, 10), np.percentile(divs, 90)
ratio = hi / max(lo, 1e-9)
ok3 = ratio > 3.0
print(f"   p10 {lo:.0f}  p90 {hi:.0f}  ratio {ratio:.1f}x  ->  {'PASS' if ok3 else 'FAIL'}")
if ok3:
    print("   One global division_weight cannot be right everywhere.")

print("\n4. divisions are concentrated in time (>60% inside fewer than half the frames)")
order = np.argsort(-FH)
cum = np.cumsum(FH[order]) / max(FH.sum(), 1)
k = int(np.searchsorted(cum, 0.60)) + 1
ok4 = k < 50
print(f"   60% of divisions fall in {k} of 100 frames  ->  {'PASS' if ok4 else 'FAIL'}")
if ok4:
    print("   The frame index is a free feature, and a time-varying prior beats a constant.")

print("\n5. zh001r carries both a volume and per-node labels")
ok5 = bool(ZH_FILES) and (
    any(p.suffix in (".csv", ".json", ".parquet", ".npy", ".npz") for p in ZH_FILES)
    and (any(p.suffix in (".zarr", ".tif", ".tiff", ".npy", ".h5", ".n5") for p in ZH_FILES)
         or any(p.is_dir() and p.suffix == ".zarr" for p in Path(ZH).rglob("*"))))
print(f"   {'PASS' if ok5 else 'FAIL'}"
      + ("" if ZH_FILES else " — not mounted"))
print("   One embryo either way (biohub-zmnscrops is 403), against 199 the published")
print("   model has already trained on. Useful as a domain-shift check, not as scale.")

print("\n" + "=" * 84)
print(f"{TOTAL_D:,.0f} division events, {rate:.2%} positive rate, "
      f"{ratio:.1f}x spread across datasets")
if TOTAL_D > 10_000:
    print("A per-node division prior is trainable on data we already have and have")
    print("never used. That is where the 0.094 of headroom is.")
print("=" * 84)
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
