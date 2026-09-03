"""Build notebooks/claude_union.ipynb — do the two detectors find DIFFERENT cells?

notes/47 closed spotiflow on standalone quality (0.547 recall against the pack's
0.996) and showed the pack dominating the recall-per-node curve at every matched
count. Neither settles set membership, and the union was never computed.

Live again because notes/52 measured ratio = -0.129: we run 12.9% UNDER budget, so
adding nodes is affordable, and the premise under which spotiflow was judged (that
over-prediction costs budget) is false at our actual operating point. Ceiling from
notes/51: the detector's own share of fn_detect is ~238 edges, 1.72% of GT.

Derived from the spotiflow builder, which already runs both detectors on the same
volumes; this adds the union matching and the rescue count.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_union.ipynb")
N_EVAL = 12
# r35's README uses 0.3 and its thresholds.yaml records 0.3 (best) / 0.38 (last).
# One point cannot compare two detectors that both have a knob, so sweep it well
# below their value too: low thresholds trade nodes for recall, which is exactly
# the axis the budget term prices.
PROB_GRID = [0.02, 0.05, 0.1, 0.2, 0.3]
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_EVAL__", str(N_EVAL))
              .replace("__PROB_GRID__", repr(PROB_GRID)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Do the two detectors find DIFFERENT cells?

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
pack detector   node recall 0.996     spotiflow   0.547        (notes/47)
```

`notes/47` closed spotiflow, and the reasoning was sound for what it tested: 0.547 recall
against 0.996, and the pack dominating the **recall-per-node curve** at every matched node
count. **Neither of those is a statement about set membership.** Two detectors can sit on
opposite sides of a quality curve and still find different cells, and the union was never
computed — `claude_spotiflow` recorded `pack_recall` and `spot_recall` in separate columns
and never put the two detection sets together.

## Why the question is live again

Two things changed after `notes/47`.

**The budget premise it was judged under is false.** `notes/52` measured `ratio = -0.129`:
we run **12.9% under** the node budget with the multiplier already paying 1.013. Spotiflow
was evaluated when the working assumption was that over-prediction costs budget. At our
actual operating point there is headroom to *add* nodes, which is the one thing an ensemble
must do.

**The ceiling is now measured.** `notes/51` put `fn_detect` at 583 edges, and
`claude_divsweep`'s arms separate it — `ctl/raw` 238 vs `inc/raw` 608 — so ~370 is imposed
by our own ILP weights and **the detector's own share is ~238 edges, 1.72% of GT**. That is
the hard ceiling on any detection-stage ensemble. Above `notes/44`'s 0.0015 floor, roughly
half the bronze gap, and not a guaranteed win.

## What has already been tried, and why this is neither

```
claude_secondary       blended a second EDGE model (temporal-unet3d), adaptive weight
                       from top-2 margin.  +0.0026, t=0.63 -- UNRESOLVED, needs n~147
claude_deepcenter_veto a parallel corrector, but applied to GAP-CLOSING candidates.
                       Bounded at ~0.002 because gap-closing is only worth +0.0013
```

Both act on the **graph**. This acts on **detection**, which is where `notes/51` says the
remaining loss now sits.

## The measurement

`purescore.match_nodes` returns, per predicted node, the GT index it matched or −1 — so the
covered GT set is directly readable. Matching is one-to-one **within a frame**, so the union
is computed by running the matcher on the **concatenated** detections, not by set-unioning
two separate runs (which would let one GT node be claimed twice and overcount the rescue).

Two modes, because the full union is not a deployable operating point:

```
union      every spotiflow detection added        upper bound on recall, worst budget
selective  only spotiflow detections >7um from    the ones that could RESCUE a miss
           any pack detection in the same frame   rather than duplicate a hit
```

7 µm is the scorer's own match radius: a spotiflow detection closer than that to a pack
detection cannot rescue anything the pack already has, so it is pure budget cost.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `pack_recall` within 0.005 of `notes/47`'s **0.996** and `spot_recall`
   within 0.02 of **0.547**. Otherwise this is not the same measurement and nothing below
   compares to the record.
2. **The union rescues something.** `union_recall − pack_recall > 0.002`. **This is the
   crux.** If the union recovers essentially nothing, spotiflow's detections are a subset of
   ours, the ensemble cannot work at any weighting, and the direction closes for one run.
3. **The selective union keeps most of the rescue.** `rescued_sel > 0.5 * rescued` — the
   rescues come from spotiflow detections genuinely far from ours, not from re-matching
   noise. If the rescue survives only in the full union, it is a matching artifact.
4. **The rescue is affordable.** Fewer than 20 added nodes per rescued GT node in the
   selective mode. At `ratio = -0.129` there is real headroom, but the multiplier costs 0.1
   per unit of ratio and a rescue that doubles the node count spends more than it earns.
5. **It holds on both embryos.** `notes/07` §3: the test set is a third pair, and a pooled
   result across crops of two says nothing about it.

*If 2 fails, detection-stage ensembling is closed regardless of architecture — parallel
corrector, veto, weighted blend or anything else — because there is nothing to add. That is
the outcome this run is designed to make cheap.*
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
from harness.purescore import match_nodes
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
PROB_GRID = __PROB_GRID__
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

def matched_gt(pt, pz, gt, sc):
    # match_nodes returns, per PREDICTED node, the GT index it matched or -1. The set of
    # non-negative entries is exactly the GT nodes this detection set covers. Matching is
    # one-to-one WITHIN A FRAME, so running it on the concatenated detections is the
    # correct union -- set-unioning two separate runs would let one GT node be claimed
    # twice and overcount the rescue.
    if len(pt) == 0:
        return set()
    m = match_nodes(np.asarray(pt), np.asarray(pz), gt.t, gt.zyx, scale=sc)
    return set(int(v) for v in m if v >= 0)


def far_from(st, sz, pt, pz, sc, radius=7.0):
    # Spotiflow detections further than the scorer's own 7um match radius from every pack
    # detection in the same frame. A closer one cannot rescue a GT node the pack already
    # matched, so it is pure budget cost.
    if len(st) == 0:
        return np.zeros(0, bool)
    s = np.asarray(sc, float)
    keep = np.ones(len(st), bool)
    st = np.asarray(st); sz = np.asarray(sz, float) * s
    pt_a = np.asarray(pt); pz_a = np.asarray(pz, float) * s
    for fr in np.unique(st):
        si = np.flatnonzero(st == fr)
        pi = np.flatnonzero(pt_a == fr)
        if len(pi) == 0:
            continue
        d = np.linalg.norm(sz[si][:, None, :] - pz_a[pi][None, :, :], axis=2)
        keep[si] = d.min(axis=1) > radius
    return keep


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

    # The pack pass is threshold-independent here and runs ONCE; spotiflow runs per
    # threshold. Same volume, same frames, same ground truth for every arm.
    spot_by_thr = dict()
    for _pt in PROB_GRID:
        _st, _sz, _ss = detect_volume(spot, arr, prob_thresh=_pt, remap_zxy=False)
        spot_by_thr[_pt] = (_st, _sz, _ss)
    st, sz, ss = spot_by_thr[PROB_GRID[-1]]
    cfg = P.PredictConfig(det_threshold=0.975, use_ilp=False)
    coords, edges = P.predict_video(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg,
                                    window_size=window_size, unet_batch_size=8,
                                    downsample=downsample)
    coords = np.asarray(coords, dtype=float)
    pt, pz = coords[:, 0].astype(np.int64), coords[:, 1:4]

    # --- the question this run exists for -------------------------------------
    M_pack = matched_gt(pt, pz, gt, sc)
    _best_thr = PROB_GRID[0]              # the loosest spotiflow cut = most candidates
    _bt, _bz, _ = spot_by_thr[_best_thr]
    M_spot = matched_gt(_bt, _bz, gt, sc)
    _ut = np.concatenate([np.asarray(pt), np.asarray(_bt)]) if len(_bt) else np.asarray(pt)
    _uz = np.concatenate([np.asarray(pz, float), np.asarray(_bz, float)]) if len(_bt) \
        else np.asarray(pz, float)
    M_union = matched_gt(_ut, _uz, gt, sc)
    _far = far_from(_bt, _bz, pt, pz, sc)
    _ft = np.asarray(_bt)[_far]; _fz = np.asarray(_bz, float)[_far]
    _sel_t = np.concatenate([np.asarray(pt), _ft]) if len(_ft) else np.asarray(pt)
    _sel_z = np.concatenate([np.asarray(pz, float), _fz]) if len(_ft) \
        else np.asarray(pz, float)
    M_sel = matched_gt(_sel_t, _sel_z, gt, sc)
    _ng = max(len(gt.t), 1)

    row = dict(name=name, n_frames=T, n_gt=int(len(gt.t)), n_total=n_total,
               n_pack_matched=len(M_pack), n_spot_matched=len(M_spot),
               n_union_matched=len(M_union), n_sel_matched=len(M_sel),
               rescued=len(M_union - M_pack), rescued_sel=len(M_sel - M_pack),
               spot_only=len(M_spot - M_pack),
               union_recall=len(M_union) / _ng, sel_recall=len(M_sel) / _ng,
               n_added_full=int(len(_bt)), n_added_sel=int(_far.sum()),
               gt_cpf=len(gt.t) / max(len(np.unique(gt.t)), 1),
               n_spot=int(len(st)), n_pack=int(len(pt)),
               spot_recall=recall_of(st, sz, gt, sc),
               pack_recall=recall_of(pt, pz, gt, sc),
               budget=node_budget(len(gt.t) / max(len(np.unique(gt.t)), 1), T))
    for tag in ("spot", "pack"):
        n = row["n_" + tag]
        row[tag + "_ratio"] = (n - n_total) / n_total if n_total > 0 else float("nan")
        row[tag + "_per_k"] = 1000.0 * row[tag + "_recall"] / max(n, 1)
    # the whole curve, one entry per threshold
    row["curve"] = dict()
    for _pt in PROB_GRID:
        _st, _sz, _ss = spot_by_thr[_pt]
        _n = int(len(_st))
        _r = recall_of(_st, _sz, gt, sc)
        row["curve"][str(_pt)] = dict(
            n=_n, recall=_r, per_k=1000.0 * _r / max(_n, 1),
            ratio=(_n - n_total) / n_total if n_total > 0 else float("nan"))
    ROWS.append(row)
    _c = "  ".join("t" + str(_pt) + " n=" + str(row["curve"][str(_pt)]["n"])
                     + " r=" + format(row["curve"][str(_pt)]["recall"], ".3f")
                     for _pt in PROB_GRID)
    print("    pack matched " + str(row["n_pack_matched"]) + "/" + str(row["n_gt"])
          + "  union " + str(row["n_union_matched"])
          + " (+" + str(row["rescued"]) + ", " + str(row["n_added_full"]) + " nodes)"
          + "  selective " + str(row["n_sel_matched"])
          + " (+" + str(row["rescued_sel"]) + ", " + str(row["n_added_sel"]) + " nodes)",
          flush=True)
    print("  " + name + "  gt " + str(row["n_gt"])
          + " (" + format(row["gt_cpf"], ".2f") + "/frame) | " + _c
          + " | pack n=" + str(row["n_pack"]) + " r=" + format(row["pack_recall"], ".3f")
          + "  " + str(int(time.time() - t0)) + "s", flush=True)
    (WORK / "union.json").write_text(json.dumps(
        dict(rows=ROWS, prob_grid=PROB_GRID), default=float))

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

md("""## 2. Grading — is there anything to ensemble?""")

code(r"""
import json, numpy as np
D = json.loads((WORK / "union.json").read_text())
R = D["rows"]
print(f"{len(R)} datasets")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(r["name"]) for r in R})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(r['name']) == e for r in R)}" for e in EMB))

# Pooled over GT nodes, not a mean of per-dataset recalls: the denominators span two
# orders of magnitude and averaging ratios across them is notes/47 §2's error.
def pooled(num, den="n_gt", rows=None):
    rows = rows if rows is not None else R
    d = sum(r[den] for r in rows)
    return sum(r[num] for r in rows) / d if d else float("nan")

tot_gt = sum(r["n_gt"] for r in R)
pk, sp = pooled("n_pack_matched"), pooled("n_spot_matched")
un, se = pooled("n_union_matched"), pooled("n_sel_matched")
resc, resc_s = sum(r["rescued"] for r in R), sum(r["rescued_sel"] for r in R)
add_f, add_s = sum(r["n_added_full"] for r in R), sum(r["n_added_sel"] for r in R)
n_pack = sum(r["n_pack"] for r in R)

print(f"\n{'set':<14}{'GT matched':>12}{'recall':>9}{'vs pack':>10}{'nodes added':>13}")
print("-" * 58)
print(f"{'pack':<14}{sum(r['n_pack_matched'] for r in R):>12,}{pk:>9.4f}{'':>10}{'':>13}")
print(f"{'spotiflow':<14}{sum(r['n_spot_matched'] for r in R):>12,}{sp:>9.4f}{'':>10}{'':>13}")
print(f"{'union':<14}{sum(r['n_union_matched'] for r in R):>12,}{un:>9.4f}"
      f"{un - pk:>+10.4f}{add_f:>13,}")
print(f"{'selective':<14}{sum(r['n_sel_matched'] for r in R):>12,}{se:>9.4f}"
      f"{se - pk:>+10.4f}{add_s:>13,}")
print(f"\nGT nodes: {tot_gt:,}   pack detections: {n_pack:,}")
print(f"rescued by union {resc:,}   by selective {resc_s:,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

print("\n1. reproduction: pack 0.996 and spotiflow 0.547 (notes/47)")
ok1 = abs(pk - 0.996) < 0.005 and abs(sp - 0.547) < 0.02
print(f"   pack {pk:.4f} (want 0.996 +-0.005)   spotiflow {sp:.4f} (want 0.547 +-0.02)"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   Not the same measurement as notes/47. Nothing below compares to the record.")

print("\n2. THE CRUX — the union rescues GT nodes the pack misses (>0.002 recall)")
ok2 = (un - pk) > 0.002
print(f"   union {un:.4f} vs pack {pk:.4f}   {un - pk:+.4f}   ({resc:,} GT nodes)"
      f"  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   Spotiflow's detections are effectively a SUBSET of ours. No weighting,")
    print("   veto, corrector or blend can extract what is not there, so detection-stage")
    print("   ensembling with this model is closed regardless of architecture.")

print("\n3. the rescue survives the 7um selectivity filter (>half of it)")
ok3 = resc > 0 and resc_s > 0.5 * resc
print(f"   selective rescues {resc_s:,} of {resc:,}"
      f"   ({resc_s / max(resc, 1):.1%})  ->  {'PASS' if ok3 else 'FAIL'}")
if not ok3:
    print("   The rescue only appears when EVERY spotiflow detection is added, i.e. it")
    print("   comes from re-matching within 7um rather than from finding new cells.")
    print("   That is a matching artifact, not complementarity.")

print("\n4. the rescue is affordable (<20 added nodes per rescued GT node)")
cost = add_s / max(resc_s, 1)
ok4 = resc_s > 0 and cost < 20
print(f"   selective: {add_s:,} nodes added for {resc_s:,} rescues = {cost:.1f} per rescue"
      f"  ->  {'PASS' if ok4 else 'FAIL'}")
# what it does to the budget we actually sit at (notes/52: ratio -0.129)
if n_pack:
    d_ratio = add_s / max(sum(r["n_total"] for r in R), 1)
    print(f"   node count +{add_s / n_pack:.1%}, moving ratio -0.129 -> {-0.129 + d_ratio:+.3f}"
          f"   multiplier {1 - 0.1 * (-0.129):.4f} -> {1 - 0.1 * (-0.129 + d_ratio):.4f}")
if not ok4:
    print("   The nodes cost more multiplier than the rescued edges can repay.")

print("\n5. it holds on BOTH embryos (notes/49 — the test set is a third pair)")
per = {}
for e in EMB:
    rows = [r for r in R if emb(r["name"]) == e]
    per[e] = (pooled("n_union_matched", rows=rows) - pooled("n_pack_matched", rows=rows),
              sum(r["rescued_sel"] for r in rows), len(rows))
print(f"   {'embryo':<8}{'n':>4}{'union - pack':>15}{'sel rescues':>13}")
for e, (d, rs, n) in per.items():
    print(f"   {e:<8}{n:>4}{d:>+15.4f}{rs:>13,}")
vals = [d for d, _, _ in per.values()]
ok5 = len(vals) > 1 and all(v > 0 for v in vals)
print(f"   both positive  ->  {'PASS' if ok5 else 'FAIL'}")
if not ok5:
    print("   The complementarity is embryo-specific, so it does not transfer to a third.")

print("\n" + "=" * 78)
n_ok = sum([ok1, ok2, ok3, ok4, ok5])
print(f"{n_ok}/5 predictions passed")
if not ok1:
    print("NOT COMPARABLE: reproduction failed; fix that before reading anything else.")
elif not ok2:
    print("CLOSED: nothing to ensemble. Spotiflow finds a subset of what we find, and no")
    print("architecture recovers information that is absent. notes/51's remaining")
    print("detection loss needs a better detector, not a second opinion from a worse one.")
elif ok2 and ok3 and ok4 and ok5:
    ceil = resc_s / max(tot_gt, 1)
    print(f"WORTH BUILDING: {resc_s:,} affordable rescues, {ceil:.2%} of GT nodes.")
    print(f"Ceiling from notes/51 is ~1.72% of GT edges; size the build against that.")
else:
    print("PARTIAL: there is complementarity but it fails an affordability or transfer")
    print("test. Read predictions 3-5 before designing anything.")
print("=" * 78)
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
