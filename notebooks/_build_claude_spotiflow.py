"""Build notebooks/claude_spotiflow.ipynb — does r35's detector find the ANNOTATED cells?

`notes/45`: `J_adj = J · (1 − 0.1·(N_pred − N_total)/N_total)`, we sit at a multiplier of
1.0012 against a ceiling of 1.1, and — the part that makes it exploitable — a predicted
edge is counted only if an endpoint matched a tracked ground-truth node. Everything else
is excluded, not penalised. So over-prediction buys nothing and costs the budget.

`altervation/biohub-r35-spotiflow` (MIT) is a complete solution from a team plausibly at
rank 35. Reading it corrected one thing I had wrong and sharpened another:

* It does **not** detect 1–25 cells per frame. Its own docstring: *"By default no
  density-model cap: Spotiflow emits hundreds of candidates per frame, and TinyUNet sparse
  hybrid (6–10/frame) ranked by intensity yields 0 recall."* The 1–25 figure I quoted came
  off a config dataclass's defaults, not the pipeline's behaviour.
* The budget comes from `predict_total_node_budget`, and only on the sparse embryo:

      budget = cells_per_frame · n_frames · 1.26,  clipped to [50, n_frames · 7]

  where `cells_per_frame` estimates the **annotation** density
  (`gt.n_nodes / n_unique_frames`), not the true cell count. On 100 frames that is
  50–700 nodes, against `N_total ≈ 24,000`. Multiplier ≈ 1.097.

Our own diagnostics agree that this is not absurd: 44b6 datasets carry 50–170 ground-truth
nodes over 100 frames — 0.5 to 1.7 annotated cells per frame against ~240 real ones.

## What this run asks

One question, and not the outcome question: **is Spotiflow selective, or merely sparse?**
A detector that finds few cells is worthless if the few are the wrong ones. So it measures
detections and recall against ground truth, on the same datasets, in the same run, against
our current detector — because three of this project's measurement errors were a control
compared with something it was not comparable to.

No linking, no ILP, no score. Detection only. If Spotiflow is not more selective per node
than what we already run, nothing downstream can rescue it and this closes in one cheap run.

    r35 artefacts are MIT-licensed; the licence and attribution travel with any use.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_spotiflow.ipynb")
N_EVAL = 12
PROB_THRESH = 0.3      # r35's README: spotiflow_domain_r35 @ prob_thresh 0.3
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_EVAL__", str(N_EVAL))
              .replace("__PROB_THRESH__", repr(PROB_THRESH)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Is r35's detector selective, or just sparse?

```
0.901  submitted        0.926 bronze     0.944 gold
measurable > 0.0015      worth a slot > 0.01        (notes/44)
```

`notes/45`, from the scorer itself:

```python
J_adj = max(0, J · (1 − 0.1 · (N_pred − N_total) / N_total))     # ceiling 1.1, ours 1.0012
pred_valid = out_valid | in_valid                                 # both from the MATCHED GT node
```

A predicted edge is counted — as TP **or** FP — only when an endpoint matched a tracked
ground-truth node. Predictions elsewhere are excluded, not penalised. **Over-prediction
buys nothing and costs the budget**, and of our ~24,000 nodes about 670 match anything.

## What reading r35 actually said

I told you earlier they detect 1–25 cells per frame. That was wrong — I read it off a
config dataclass's defaults. Their detector docstring says the opposite:

> *"By default **no** density-model cap: Spotiflow emits hundreds of candidates per frame,
> and TinyUNet sparse hybrid (6–10/frame) ranked by intensity yields **0 recall** even with
> correct axes."*

The budget is applied afterwards, and only on the sparse embryo:

```python
budget = cells_per_frame · n_frames · 1.26,  clipped to [50, n_frames · 7]
# cells_per_frame estimates the ANNOTATION density: gt.n_nodes / n_unique_frames
```

50–700 nodes against `N_total ≈ 24,000`. Our own logs make that credible: `44b6` datasets
carry **50–170** ground-truth nodes across 100 frames — 0.5 to 1.7 annotated cells per
frame, against roughly 240 real ones.

So the pair is: a detector that finds the annotated cells, and a trim that collects the
multiplier. Neither works alone. **A trim without a selective detector just deletes recall.**

## What this run measures — and what it does not

Detection only. No linking, no ILP, no score. One question:

> Per node kept, does Spotiflow recover more ground truth than the detector we run today?

Both detectors run **in the same notebook on the same datasets and the same frames**. Three
of this project's measurement errors were a control compared with something it was not
comparable to; quoting our own numbers from an earlier note would have been the fourth.

## Pre-registered predictions

1. **Spotiflow installs from r35's wheels and the fine-tune loads** on the P100 torch. A
   gate, not a result — if it fails, the direction needs a different route and we know in
   ten minutes.
2. **It detects far fewer nodes** — under half of ours on the same frames. Its docstring
   says "hundreds per frame" against our ~240, so this is genuinely uncertain and worth
   asking rather than assuming.
3. **Node recall stays above 0.60** against ground truth. Below that it is sparse, not
   selective, and no budget trim can rescue it.
4. **Recall per thousand nodes is at least 3× ours.** The selectivity claim, and the one
   that decides this. Predictions 2 and 3 can both pass while this fails, which is exactly
   the case where a trim would look attractive and be worthless.
5. **`total_node_ratio` below −0.5** at r35's own threshold — it reaches the regime the
   multiplier pays for without any extra trimming.

*Prediction 4 is the crux. `notes/42`: a prediction whose threshold sits below the
measurement's resolution is a coin flip with a paper trail, so these are set where the
decision changes.*

---
*`altervation/biohub-r35-spotiflow` is MIT-licensed. The licence and attribution travel
with any use of its weights or approach.*
""")

code(r"""
import os, subprocess, sys, time, json
from pathlib import Path

T_START = time.time()
WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-2500:]); print(r.stderr[-2500:])
    return r.returncode == 0

print(sh("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader").stdout.strip()
      or "no GPU")

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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir()
                and (p / "wheels").is_dir() and "seed314159" not in str(p),
                ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
R35 = find_dir(lambda p: (p / "models").is_dir() and (p / "wheels").is_dir()
               and any(p.glob("README.md")), ["/kaggle/input"])
SPOT_DIR = find_dir(lambda p: any(p.glob("best.pt")) and any(p.glob("config.yaml")),
                    [R35] if R35 else [], max_depth=4)
TORCH_WH = find_dir(
    lambda p: p.name == "wheels" and any(x.name.startswith("torch-") for x in p.iterdir()),
    ["/kaggle/input"])
for label, val in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                   ("r35", R35), ("spotiflow model", SPOT_DIR), ("torch wheels", TORCH_WH)):
    print(f"  {label:<16} {val}")
missing = [l for l, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                          ("r35", R35), ("spotiflow model", SPOT_DIR)) if v is None]
if missing:
    raise SystemExit(f"not mounted: {missing}")
TRAIN = COMP / "train"

t0 = time.time()
ok1 = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                  extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok1 else 'FAILED'} ({time.time()-t0:.0f}s)")

# The P100 is sm_60 and the image torch ships sm_70+ kernels. r35's wheel set carries no
# torch at all, so ours must land FIRST and spotiflow must resolve against it rather than
# pulling its own.
if TORCH_WH is None:
    raise SystemExit("no torch wheelhouse — the P100 cannot run the image torch")
t0 = time.time()
ok2 = pip_install(["torch==2.5.1"], extra=("--no-index", f"--find-links={TORCH_WH}"))
print(f"torch wheels {'ok' if ok2 else 'FAILED'} ({time.time()-t0:.0f}s)")

t0 = time.time()
R35_WH = R35 / "wheels"
# v1 asked pip to RESOLVE spotiflow and got ResolutionImpossible: the image ships
# torchvision 0.25.0+cu128 and spotiflow pins an incompatible one. Falling back to
# --no-deps on spotiflow ALONE then installed it and left `lightning` missing, so the
# import died one line later.
#
# r35's wheel set is a complete offline closure — 45 wheels — so pass every one
# explicitly with --no-deps and let pip INSTALL rather than resolve. Checked against the
# pack's stack: zarr is the only overlap (no numpy, scipy, torch or torchvision), and it
# is excluded because the pack's copy already works and every zarr read here uses the
# pack's convention.
_wheels = [str(p) for p in sorted(R35_WH.glob("*.whl")) if not p.name.startswith("zarr-")]
print(f"installing {len(_wheels)} r35 wheels with --no-deps (zarr excluded)")
ok3 = pip_install(_wheels, extra=("--no-index", "--no-deps", f"--find-links={R35_WH}"))
print(f"spotiflow stack {'ok' if ok3 else 'FAILED'} ({time.time()-t0:.0f}s)")

# v2 got the wheels in and then died importing them. The image ships torchvision
# 0.25.0+cu128, built against a different torch than the 2.5.1+cu121 the P100 needs, so
# its C extension will not load:
#
#   AttributeError: partially initialized module 'torchvision' has no attribute
#   'extension' (most likely due to a circular import)
#
# and the chain that hits it is spotiflow -> lightning -> torchmetrics ->
# functional.image.arniqa -> torchvision. Nothing in either wheel set carries a matching
# torchvision, so install the one that PAIRS with torch 2.5.1. Internet is on for this
# probe; a submission notebook would need this wheel published to a wheelhouse first,
# exactly as torch itself was.
t0 = time.time()
ok4 = pip_install(["torchvision==0.20.1"],
                  extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
print(f"torchvision 0.20.1 {'ok' if ok4 else 'FAILED'} ({time.time()-t0:.0f}s)")
_tv = sh(sys.executable, "-c",
         "import torch, torchvision; from torchvision import transforms; "
         "print('torch', torch.__version__, '| torchvision', torchvision.__version__)")
print(_tv.stdout.strip() or _tv.stderr.strip()[-800:])
if _tv.returncode != 0:
    print("!! torchvision still broken — spotiflow cannot import, prediction 1 fails")

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, scipy; ok=False\n"
           "import spotiflow; from spotiflow.model import Spotiflow\n"
           "if torch.cuda.is_available():\n"
           "    try:\n"
           "        w=torch.nn.Conv3d(1,4,3,padding=1).cuda()\n"
           "        _=w(torch.randn(2,1,8,8,8,device='cuda')).sum().item()\n"
           "        torch.cuda.synchronize(); ok=True\n"
           "    except Exception as e: print('GPU BROKEN:', type(e).__name__, str(e)[:120])\n"
           "print('numpy', numpy.__version__, '| torch', torch.__version__,\n"
           "      '| spotiflow', getattr(spotiflow,'__version__','?'), '| gpu_ok', ok)")
print(probe.stdout.strip() or probe.stderr.strip()[-2000:])
SPOT_OK = probe.returncode == 0 and "gpu_ok True" in probe.stdout
print(f"\nprediction 1 (spotiflow imports and the GPU works): "
      f"{'PASS' if SPOT_OK else 'FAIL'}")
if not SPOT_OK:
    raise SystemExit("spotiflow unavailable — prediction 1 failed, which is the answer")
""")

md("""
## 1. Both detectors, same datasets, same frames

In a subprocess: the pack wheels replace numpy under this kernel, so `scipy` here no longer
matches it and `pipeline/detector.py` needs scipy. `claude_divdata` v1 died on exactly that.
""")

WORKER_BODY = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np
import torch

T0 = time.time()
sys.path.insert(0, "{repo}")
PACK = Path("{pack}"); TRAIN = Path("{train}"); WORK = Path("{work}")
SPOT_DIR = Path("{spot}")

import zarr
from harness.tracks import read_geff, read_scale, read_estimated_nodes
from pipeline.detector import recall_at_budget
from pipeline.spotiflow import load as spot_load, detect_volume, node_budget

# BOTH paths. `repo/src` holds the `biohub_tracking` package that the entry script
# imports; `repo/scripts` holds the script itself. v3 added only the second and died on
# `ModuleNotFoundError: No module named 'biohub_tracking'` seven seconds in, because I
# wrote this block from memory instead of copying the one that already works in
# _build_claude_widecv.py.
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))
import types
# The pack's entry point is a SCRIPT, not a package, and it imports a `dataspec` module
# that exists only in its authors' environment, so a synthetic one has to be injected.
#
# This block is copied VERBATIM from _build_claude_widecv.py, which works. Three launches
# of this notebook died here because I retyped it from memory instead: first the missing
# `repo/src` path, then WEIGHTS_PATH/DATASET_PATH without USERNAME and INTERACTIVE. The
# working copy even carries a comment saying to copy rather than retype. Fixing a
# known-good preamble one attribute per launch is the expensive way to rediscover it.
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"; _ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"; _ds.DATASET_PATH = TRAIN
_ds.PREDICTIONS_PATH = WORK / "predictions"
sys.modules["dataspec"] = _ds
import predict_unet_transformer as P

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("worker numpy " + np.__version__ + " torch " + torch.__version__ + " on "
      + str(DEV), flush=True)

N_EVAL = __N_EVAL__
PROB_THRESH = __PROB_THRESH__
COMP_UM = (1.625, 0.40625, 0.40625)

WPATH = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WPATH, DEV)
print("pack model params " + str(sum(p.numel() for p in model.parameters()))
      + " window " + str(window_size) + " downsample " + str(downsample), flush=True)

spot = spot_load(SPOT_DIR, device=str(DEV))
_np = sum(p.numel() for p in spot.parameters()) if hasattr(spot, "parameters") else -1
print("spotiflow loaded from " + str(SPOT_DIR) + " params " + str(_np), flush=True)

names = sorted(p.stem for p in TRAIN.glob("*.zarr")
               if (TRAIN / (p.stem + ".geff")).exists())
a = [n for n in names if n.startswith("44b6")]
b = [n for n in names if not n.startswith("44b6")]
k = max(1, round(N_EVAL * len(a) / max(len(a) + len(b), 1)))
names = a[:k] + b[:N_EVAL - k]
print(str(len(names)) + " datasets: " + str(sum(1 for n in names if n.startswith("44b6")))
      + " x 44b6", flush=True)

def recall_of(pt, pz, gt, sc):
    # recall_at_budget wraps purescore.match_nodes, which is the verified reimplementation
    # of the official node matching -- so this is the recall the leaderboard would compute,
    # not a proxy. Written by hand first, wrongly: match_nodes takes the WHOLE arrays and
    # handles frames itself, returning one matched GT index per PREDICTED node. Reusing the
    # tested helper is how not to make that mistake a fourth time.
    if len(pt) == 0:
        return 0.0
    return float(recall_at_budget(pt, pz, gt.t, gt.zyx, sc))

ROWS = []
for name in names:
    t0 = time.time()
    arr = zarr.open_group(str(TRAIN / (name + ".zarr")), mode="r")["0"]
    gt = read_geff(TRAIN / (name + ".geff"))
    sc = read_scale(TRAIN / (name + ".zarr"))
    n_total = float(read_estimated_nodes(TRAIN / (name + ".geff")))
    T = int(arr.shape[0])

    st, sz, ss = detect_volume(spot, arr, prob_thresh=PROB_THRESH, remap_zxy=False)
    cfg = P.PredictConfig(det_threshold=0.975, use_ilp=False)
    coords, edges = P.predict_video(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg,
                                    window_size=window_size, unet_batch_size=8,
                                    downsample=downsample)
    coords = np.asarray(coords, dtype=float)
    pt, pz = coords[:, 0].astype(np.int64), coords[:, 1:4]

    row = dict(name=name, n_frames=T, n_gt=int(len(gt.t)), n_total=n_total,
               gt_cpf=len(gt.t) / max(len(np.unique(gt.t)), 1),
               n_spot=int(len(st)), n_pack=int(len(pt)),
               spot_recall=recall_of(st, sz, gt, sc),
               pack_recall=recall_of(pt, pz, gt, sc),
               budget=node_budget(len(gt.t) / max(len(np.unique(gt.t)), 1), T))
    for tag in ("spot", "pack"):
        n = row["n_" + tag]
        row[tag + "_ratio"] = (n - n_total) / n_total if n_total > 0 else float("nan")
        row[tag + "_per_k"] = 1000.0 * row[tag + "_recall"] / max(n, 1)
    ROWS.append(row)
    print("  " + name + "  gt " + str(row["n_gt"])
          + " (" + format(row["gt_cpf"], ".2f") + "/frame)"
          + " | spot n=" + str(row["n_spot"]) + " rec=" + format(row["spot_recall"], ".3f")
          + " | pack n=" + str(row["n_pack"]) + " rec=" + format(row["pack_recall"], ".3f")
          + "  " + str(int(time.time() - t0)) + "s", flush=True)
    (WORK / "spotiflow.json").write_text(json.dumps(dict(rows=ROWS), default=float))

print("worker done in " + str(int(time.time() - T0)) + " s", flush=True)
'''

code('''
import subprocess, sys, time
WORKER = WORK / "run_spot.py"
WORKER.write_text(""" + BODY + """.format(
    pack=str(PACK), repo=str(REPO), train=str(TRAIN), work=str(WORK), spot=str(SPOT_DIR)))

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print("worker exited", rc, "after", int(time.time() - t0), "s")
if rc != 0:
    raise SystemExit("worker failed (" + str(rc) + ")")
'''.replace('""" + BODY + """', Q3 + WORKER_BODY + Q3))

md("""## 2. The five predictions""")

code(r"""
import json, numpy as np
D = json.loads((WORK / "spotiflow.json").read_text())
R = D["rows"]

def col(k):
    v = [r[k] for r in R if k in r and r[k] == r[k]]
    return float(np.mean(v)) if v else float("nan")

print(f"{len(R)} datasets\n")
print(f"{'dataset':<18}{'gt':>6}{'cpf':>7}{'spot n':>9}{'rec':>7}"
      f"{'pack n':>9}{'rec':>7}{'budget':>8}")
print("-" * 74)
for r in R:
    print(f"{r['name']:<18}{r['n_gt']:>6}{r['gt_cpf']:>7.2f}{r['n_spot']:>9,}"
          f"{r['spot_recall']:>7.3f}{r['n_pack']:>9,}{r['pack_recall']:>7.3f}"
          f"{r['budget']:>8,}")

SN, PN = col("n_spot"), col("n_pack")
SR, PR = col("spot_recall"), col("pack_recall")
SK, PK = col("spot_per_k"), col("pack_per_k")
print(f"\n{'':<18}{'spotiflow':>14}{'pack':>14}")
for lbl, s, p in (("nodes", SN, PN), ("node recall", SR, PR),
                  ("recall / 1k nodes", SK, PK),
                  ("total_node_ratio", col("spot_ratio"), col("pack_ratio"))):
    print(f"{lbl:<18}{s:>14.4f}{p:>14.4f}")

print("\n" + "=" * 84)
print("PREDICTION GRADING")
print("=" * 84)
print("\n1. spotiflow installs and the fine-tune loads")
print("   PASS — the setup cell raises otherwise, so reaching here is the verdict")

print("\n2. it detects under half as many nodes as ours on the same frames")
ok2 = SN == SN and PN == PN and SN < 0.5 * PN
print(f"   spotiflow {SN:,.0f} vs pack {PN:,.0f}"
      f"  ({SN/max(PN,1):.2f}x)  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   Its docstring says 'hundreds per frame', so this was genuinely open. If it")
    print("   emits as many as ours, the trim has to do all the work and the detector is")
    print("   not the differentiator.")

print("\n3. node recall stays above 0.60")
ok3 = SR == SR and SR > 0.60
print(f"   {SR:.4f}  ->  {'PASS' if ok3 else 'FAIL'}")
if not ok3:
    print("   Sparse, not selective. No budget trim rescues a detector that has already")
    print("   lost the annotated cells.")

print("\n4. recall per 1,000 nodes is at least 3x ours  <- the crux")
ok4 = SK == SK and PK == PK and PK > 0 and SK > 3.0 * PK
print(f"   spotiflow {SK:.4f} vs pack {PK:.4f}  ({SK/max(PK,1e-9):.1f}x)"
      f"  ->  {'PASS' if ok4 else 'FAIL'}")
if not ok4:
    print("   Predictions 2 and 3 can both pass while this fails, and that is precisely")
    print("   the case where a trim looks attractive and is worthless: fewer nodes, but")
    print("   no better at picking the annotated ones.")

print("\n5. total_node_ratio below -0.5 at r35's own threshold")
SRAT = col("spot_ratio")
ok5 = SRAT == SRAT and SRAT < -0.5
print(f"   {SRAT:+.3f}  (multiplier {1 - 0.1 * SRAT:.4f})  ->  {'PASS' if ok5 else 'FAIL'}")

print("\n" + "=" * 84)
print(f"spotiflow {SN:,.0f} nodes @ recall {SR:.3f}  |  pack {PN:,.0f} @ {PR:.3f}"
      f"  |  selectivity {SK/max(PK,1e-9):.1f}x")
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
