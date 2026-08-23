"""Build notebooks/claude_cluster_repro.ipynb — score the public pipeline on OUR harness.

The point is a number that is directly comparable to our champion's CV 0.7070. Their
`--evaluate` flag would produce a number in their metric on their splits; ours is what
every decision in this project has been made against, and mixing the two would make the
comparison unreadable.

`predict_video()` returns `(coords, edges)` with coords `(N, 4)` `(t, z, y, x)` in ORIGINAL
voxel space and edges as `(src_idx, tgt_idx, prob, dist)` — which is exactly the shape our
`pipeline.classical.build_graph` consumes. So their model plugs into our `Harness` with no
file I/O and none of their splits machinery.

Run 1 is deliberately SMALL. The question it answers is "does this run at all, and how fast",
not "what does it score". Scoring 60 datasets before knowing the per-dataset cost is how you
discover a 12 h ceiling at hour eleven.

Rules position: Competition Rules §6.b permits external models unless the Host specifically
prohibits them, and §6.a is satisfied by a public CC0 dataset. The Competition-Specific
Rules section is still unconfirmed, so this notebook MEASURES ONLY — it writes no
submission and is not a competition entry.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_cluster_repro.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Reproduce the cluster, scored on our own harness

`claude_cluster_probe` established what the public pack contains. This runs its model and
scores it with **our** `Harness`, so the result sits on the same scale as everything else
in this project:

| | CV | LB |
|---|---|---|
| champion (`adaptive_predicted`) | **0.7070** | **0.752** |
| best learned arm we trained | 0.6490 | 0.633 |
| public cluster (this notebook) | **?** | ~0.913–0.916 reported |

Their own `--evaluate` would report a number in their metric on their splits. That is not
comparable to 0.7070, and a number that cannot be compared is not worth having.

## How it plugs in

`predict_video()` returns `(coords, edges)` — coords `(N, 4)` as `(t, z, y, x)` in
**original voxel space**, edges as `(src_idx, tgt_idx, prob, dist)`. Our
`build_graph(coords_tzyx, edges)` takes precisely that. So the model is wrapped as a
predictor function and handed to `Harness.evaluate`, with no file I/O and none of their
splits machinery.

`dataspec` is **shimmed** rather than used: it supplies `USERNAME`, `WEIGHTS_PATH` and
friends from the author's local environment, and none of those paths exist here. Replacing
it in `sys.modules` before the import is cleaner than hoping it degrades gracefully.

## This run is small on purpose

`N_DATASETS` starts at **6**. The question is "does it run, and at what cost per dataset" —
scoring 60 before knowing the per-dataset cost is how a 12 h ceiling gets discovered at
hour eleven. The projection is printed so the next run can be sized rather than guessed.

**No submission.** The Competition-Specific Rules section is still unconfirmed, so this
measures only.
""")

code(r"""
import os, subprocess, sys, time
from pathlib import Path

def sh(*a, **kw):
    try:
        return subprocess.run(a, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(a, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return r.returncode == 0

gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").stdout.strip()
print(f"accelerator: {gpu or 'NONE'}")
# P100 is sm_60 and the image torch ships sm_70+ only: CUDA reports available and every
# launch dies (notes/17 §4). Replace torch FIRST -- their requirements pull packages that
# pin against whatever torch is present.
if "P100" in gpu:
    print("P100 -> installing torch with sm_60 kernels ...")
    t0 = time.time()
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")

def numpy_version():
    r = sh(sys.executable, "-c", "import numpy; print(numpy.__version__)")
    return (r.stdout or r.stderr).strip()

print(f"numpy before deps: {numpy_version()}")

# Install from the PACK'S OWN WHEELS, not PyPI.
#
# Installing these from PyPI produced:
#   ImportError: cannot import name '_center' from 'numpy._core.umath'
# `_center` exists only in numpy >= 2.3, so a package compiled against a NEW numpy landed
# next to the image's OLD one. Resolving that by hand means guessing which of blosc2 /
# imagecodecs / numcodecs forced it. The pack ships a wheel set its author actually ran --
# including numpy itself -- so installing that set wholesale is coherent by construction,
# and it is exactly what their offline submission notebook does.
PACK_GUESS = None
for root in (Path("/kaggle/input"),):
    stack = [(root, 0)]
    while stack and PACK_GUESS is None:
        d, depth = stack.pop(0)
        try:
            kids = list(os.scandir(d)) if d.is_dir() else []
        except (PermissionError, OSError):
            continue
        for e in kids:
            if e.is_dir() and e.name == "wheels":
                PACK_GUESS = Path(e.path); break
            if e.is_dir() and depth < 5 and not e.name.endswith((".zarr", ".geff")):
                stack.append((Path(e.path), depth + 1))

t0 = time.time()
if PACK_GUESS is not None:
    n_whl = len([p for p in PACK_GUESS.iterdir() if p.name.endswith(".whl")])
    print(f"installing {n_whl} wheels from {PACK_GUESS} (--no-index)")
    ok = pip_install([str(p) for p in sorted(PACK_GUESS.glob("*.whl"))],
                     extra=("--no-index", f"--find-links={PACK_GUESS}"))
    print(f"  pack wheels {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
else:
    print("!! no wheels/ directory found in the pack; falling back to PyPI")
    ok = pip_install(["tracksdata", "zarr>=3.0.10,<4", "pyscipopt", "geff", "ilpy",
                      "polars", "blosc2", "dask", "imagecodecs", "pyarrow", "rustworkx",
                      "sqlalchemy", "numpy>=2.3"])
    print(f"  PyPI requirements {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")

print(f"numpy after deps:  {numpy_version()}")

# The official scorer. Their model PREDICTS DIVISIONS -- run 1 saw 564 forking nodes in
# one dataset -- and our purescore's division term is exact only for fork-free
# predictions, so harness.score_graph routes any forking prediction to the organisers'
# code and refuses to guess. That code needs this repo plus tracksdata (installed above).
#
# This also means the number below INCLUDES the division term, which is worth 0.1 of the
# 1.1 maximum and which every arm we have ever run scored 0.000 on.
CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    t0 = time.time()
    r = sh("git", "clone", "--depth", "1",
           "https://github.com/royerlab/kaggle-cell-tracking-competition", str(CELLMOT))
    print(f"official scorer clone rc={r.returncode} ({time.time()-t0:.0f}s)")
    if r.returncode != 0:
        print(r.stdout[-800:]); print(r.stderr[-800:])
os.environ["CELLMOT_REPO"] = str(CELLMOT)
print(f"CELLMOT_REPO={CELLMOT}  present={(CELLMOT / 'src' / 'tracking_cellmot').is_dir()}")

# Prove the stack actually imports in a FRESH interpreter before the notebook commits to
# it. An ImportError here costs seconds; the same one after model load costs the run.
probe = sh(sys.executable, "-c",
           "import numpy, zarr, polars, tracksdata, torch; "
           "import numpy._core.umath as u; "
           "print('numpy', numpy.__version__, '| torch', torch.__version__, "
           "'| zarr', zarr.__version__, '| tracksdata ok')")
print(probe.stdout.strip() or probe.stderr.strip()[-1200:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import cleanly -- see above. Fix the "
                     "install before running the model; a partial numpy split will "
                     "surface as an unrelated-looking error deeper in.")
""")

code(r"""
import os, gc, json, types, hashlib
from pathlib import Path
import numpy as np
import torch

WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
T_START = time.time()
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device {DEV}")
if DEV.type == "cuda":
    cc = torch.cuda.get_device_capability(0)
    print(f"  sm_{cc[0]}{cc[1]}  built for {torch.cuda.get_arch_list()}")
    try:
        _w = torch.nn.Conv3d(1, 4, 3, padding=1).to(DEV)
        _ = _w(torch.randn(2, 1, 8, 8, 8, device=DEV)).sum().item()
        torch.cuda.synchronize(); print("  GPU smoke test passed")
    except Exception as e:
        raise SystemExit(f"GPU present but unusable: {type(e).__name__}: {str(e)[:200]}")
else:
    print("  !! no GPU — a 2.1M-parameter attention model on CPU will be very slow")

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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(),
                ["/kaggle/input"])
if PACK is None:
    raise SystemExit("Support pack not mounted (need repo/ and weights/).")
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
if REPO is None:
    raise SystemExit("Our project snapshot not mounted (need harness/ and pipeline/).")
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Competition data not mounted.")
TRAIN = COMP / "train"
print(f"pack: {PACK}\nours: {REPO}\ndata: {TRAIN}")

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
if not WEIGHTS.exists():
    raise SystemExit(f"weights not found at {WEIGHTS}")
print(f"weights: {WEIGHTS.name} ({WEIGHTS.stat().st_size/1e6:.1f} MB)")

# --- run configuration, baked into the worker script below -------------------
# notes/15 §3 transcribed DET_THRESHOLD = 0.985 from the public notebook; their script's
# own default is 0.99. Run both, so the operating point is measured here rather than
# inherited from a five-day-old reading of a notebook that has since 404'd.
DET_THRESHOLDS = (0.985, 0.99)
N_DATASETS = 6        # run 1 is a plumbing + timing check, not a score
UNET_BATCH = 4
print(f"config: det_thresholds={DET_THRESHOLDS}  n_datasets={N_DATASETS}  "
      f"unet_batch={UNET_BATCH}")
""")

md("""## 1. Run everything in a fresh interpreter

The pack's wheels upgrade numpy **2.0.2 -> 2.4.6**, and this notebook's process already
imported numpy 2.0.2's compiled extensions before that happened. In-process the result is a
hybrid — old `.so` files already resident, new pure-Python files on disk — which surfaces as

```
AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'
```

A fresh interpreter imports the same stack cleanly (the probe above proves it), so the
whole job — their model, our `Harness`, the scoring — runs in a **subprocess** and returns
JSON. That is also how their code is meant to be invoked: `predict_unet_transformer.py` has
a `main()` and a CLI.

Doing only *part* of the work in the subprocess would not help: our harness reads ground
truth through `zarr`, which the same upgrade touched.
""")

code(r"""
WORKER = WORK / "run_pack.py"
WORKER.write_text(f'''
import json, sys, time, types, hashlib
from pathlib import Path
import numpy as np

PACK = Path({str(PACK)!r})
REPO = Path({str(REPO)!r})
TRAIN = Path({str(TRAIN)!r})
WEIGHTS = Path({str(WEIGHTS)!r})
OUT = Path({str(WORK / "pack_result.json")!r})
N_DATASETS = {N_DATASETS}
DET_THRESHOLDS = {DET_THRESHOLDS!r}
UNET_BATCH = {UNET_BATCH}

import os
os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}

import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("worker numpy", np.__version__, "torch", torch.__version__, "device", DEV, flush=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))

# Shim their dataspec: it carries the author local paths, and predict_unet_transformer
# imports three of its names at module scope.
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"; _ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"; _ds.DATASET_PATH = TRAIN
_ds.PREDICTIONS_PATH = Path("/kaggle/working/predictions")
sys.modules["dataspec"] = _ds

import predict_unet_transformer as P
from harness import Harness
from pipeline.classical import build_graph as our_build_graph

model, window_size, downsample = P.load_model(WEIGHTS, DEV)
n_par = sum(p.numel() for p in model.parameters())
print(f"model {{n_par:,}} params, window_size={{window_size}}, downsample={{downsample}}",
      flush=True)

names = sorted({{p.stem for p in TRAIN.glob("*.zarr")}}
               & {{p.stem for p in TRAIN.glob("*.geff")}})
def key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
by = {{}}
for n in names:
    by.setdefault(n.split("_")[0], []).append(n)
SUBSET = []
for pfx, ns in sorted(by.items()):
    SUBSET += sorted(ns, key=key)[:max(1, round(N_DATASETS * len(ns) / len(names)))]
SUBSET = sorted(SUBSET)[:N_DATASETS]
print("subset:", SUBSET, flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
out = {{"subset": SUBSET, "n_params": int(n_par), "window_size": int(window_size),
       "downsample": [int(x) for x in downsample], "device": str(DEV),
       "scores": {{}}, "summaries": {{}}, "sec_per_dataset": {{}}}}

for th in DET_THRESHOLDS:
    cfg = P.PredictConfig(det_threshold=th, use_ilp=False)
    times = []
    def fn(name, data_dir, _cfg=cfg, _times=times):
        t0 = time.time()
        coords, edges = P.predict_video(
            model, Path(data_dir) / f"{{name}}.zarr", DEV, cfg=_cfg,
            window_size=window_size, unet_batch_size=UNET_BATCH, downsample=downsample)
        idx = [(int(s), int(t)) for s, t, _p, _d in edges]
        g = our_build_graph(np.asarray(coords, float), idx)
        _times.append(time.time() - t0)
        print(f"    {{name:<22}} {{g.n_nodes:>7,}} nodes {{g.n_edges:>7,}} edges "
              f"({{time.time()-t0:.0f}}s)", flush=True)
        return g
    print(f"=== det_threshold={{th}} ===", flush=True)
    res = h.evaluate(fn, arm=f"pack_th{{th}}", names=SUBSET, verbose=False)
    s = res.summary
    out["scores"][str(th)] = float(res.score)
    out["summaries"][str(th)] = {{k: float(v) for k, v in s.items()
                                 if isinstance(v, (int, float))}}
    out["sec_per_dataset"][str(th)] = float(np.mean(times)) if times else None
    print(f"  SCORE={{s['score']:.4f}} edge_J={{s['edge_jaccard']:.4f}} "
          f"recall={{s['node_recall']:.3f}}", flush=True)

OUT.write_text(json.dumps(out, indent=2))
print("worker wrote", OUT, flush=True)
''')
print(f"wrote worker ({len(WORKER.read_text()):,} chars)")

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"\nworker exited {rc} after {time.time()-t0:.0f}s")
if rc != 0:
    raise SystemExit(f"worker failed (exit {rc}) — see its output above")
""")

code(r"""
import json
RES = json.loads((WORK / "pack_result.json").read_text())
CHAMPION_CV, OURS_BEST_LEARNED = 0.7070, 0.6490

print(f"{'arm':<20} {'SCORE':>8} {'edge_J':>8} {'recall':>7} {'s/dataset':>10}")
print("-" * 58)
for th, sc in RES["scores"].items():
    s = RES["summaries"][th]
    t = RES["sec_per_dataset"].get(th) or float("nan")
    print(f"{'pack th='+th:<20} {sc:>8.4f} {s.get('edge_jaccard', float('nan')):>8.4f} "
          f"{s.get('node_recall', float('nan')):>7.3f} {t:>10.1f}")
print(f"{'champion (ours)':<20} {CHAMPION_CV:>8.4f} {0.7128:>8.4f} {0.866:>7.3f} {'~30':>10}")
print(f"{'best learned (ours)':<20} {OURS_BEST_LEARNED:>8.4f} {0.6556:>8.4f} {0.885:>7.3f} {'-':>10}")

best_th = max(RES["scores"], key=lambda k: RES["scores"][k])
best = RES["scores"][best_th]
print(f"\nbest pack arm: {best:.4f} at det_threshold={best_th}")
print(f"  vs our champion {CHAMPION_CV:.4f}:  {best - CHAMPION_CV:+.4f}")
print(f"\nmodel: {RES['n_params']:,} params, window_size={RES['window_size']}, "
      f"downsample={RES['downsample']}")

per = [v for v in RES["sec_per_dataset"].values() if v]
if per:
    p = sum(per) / len(per)
    print(f"\ncost {p:.1f} s/dataset on {RES['device']}")
    print(f"  60-dataset CV run: {p*60/3600:.2f} h")
    print(f"  ~200-dataset scored rerun: {p*200/3600:.2f} h against a 12 h ceiling")
    if p * 200 > 10.5 * 3600:
        print("  !! does not fit a scored rerun as-is — batching or a faster path is "
              "required before any submission is possible")
print(f"\nNOTE: this is a MEASUREMENT ONLY. No submission is written; the "
      f"Competition-Specific Rules section is still unconfirmed.")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try: ast.parse("\n".join("pass" if l.strip().startswith("!") else l for l in src.splitlines()))
        except SyntaxError as e: raise SystemExit(f"cell {i}: {e}\n{src}")
