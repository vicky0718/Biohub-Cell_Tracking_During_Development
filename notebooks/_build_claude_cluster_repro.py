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
import subprocess, sys, time

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

# From the pack's own README ("Kaggle dependency input command"). ILP needs pyscipopt+ilpy.
t0 = time.time()
ok = pip_install(["tracksdata", "zarr>=3.0.10,<4", "pyscipopt", "geff", "ilpy",
                  "polars", "blosc2", "dask", "imagecodecs", "pyarrow", "rustworkx",
                  "sqlalchemy"])
print(f"pack requirements {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
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
""")

md("""## 1. Import their inference code

`dataspec` is replaced with a shim before the import. It exists to carry the author's local
paths (`USERNAME`, `WEIGHTS_PATH`, `DATASET_PATH`, `PREDICTIONS_PATH`), none of which exist
here — and `predict_unet_transformer` imports three of them at module scope, so a missing
or assertive `dataspec` would break the import rather than one function.
""")

code(r"""
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))

# Shim BEFORE importing their module. Their dataspec carries local paths; we supply the
# three names imported at module scope plus the two used inside functions we do not call.
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"
_ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"
_ds.DATASET_PATH = TRAIN
_ds.PREDICTIONS_PATH = WORK / "predictions"
sys.modules["dataspec"] = _ds

import predict_unet_transformer as P
print("imported their predict module:",
      [n for n in ("predict_video", "load_model", "PredictConfig") if hasattr(P, n)])

model, window_size, downsample = P.load_model(WEIGHTS, DEV)
n_par = sum(p.numel() for p in model.parameters())
print(f"model loaded: {n_par:,} parameters, window_size={window_size}, "
      f"downsample={downsample}")
print(f"  our pipeline uses downsample=(1, 4, 4) — "
      f"{'MATCH' if tuple(downsample) == (1, 4, 4) else 'MISMATCH, investigate'}")
""")

md("""## 2. Wrap it as a predictor and score on our harness

`notes/15` §3 recorded `DET_THRESHOLD = 0.985` and `USE_ILP = 1` from the public notebook;
their script's own default is 0.99. Both are run so the choice is measured rather than
inherited from a transcription.
""")

code(r"""
from harness import Harness
from pipeline.classical import build_graph as our_build_graph

DET_THRESHOLDS = (0.985, 0.99)
USE_ILP = True
N_DATASETS = 6            # run 1 is a plumbing + timing check, not a score
UNET_BATCH = 4

train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")}
                     & {p.stem for p in TRAIN.glob("*.geff")})
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)
# Same stable, embryo-balanced selection our other notebooks use, so the subset is not a
# different population from the one the champion was measured on.
SUBSET = []
for pfx, ns in sorted(by_prefix.items()):
    SUBSET += sorted(ns, key=stable_key)[:max(1, round(N_DATASETS * len(ns) / len(train_names)))]
SUBSET = sorted(SUBSET)[:N_DATASETS]
print(f"{len(SUBSET)} datasets: {SUBSET}")

CACHE = WORK / "cache"; CACHE.mkdir(parents=True, exist_ok=True)
h = Harness(data_dir=TRAIN, cache_dir=None)   # no cache: arms differ by threshold

TIMES = {}

def make_predictor(det_threshold):
    cfg = P.PredictConfig(det_threshold=det_threshold, use_ilp=False)
    def _fn(name, data_dir):
        t0 = time.time()
        coords, edges = P.predict_video(
            model, Path(data_dir) / f"{name}.zarr", DEV, cfg=cfg,
            window_size=window_size, unet_batch_size=UNET_BATCH,
            downsample=downsample,
        )
        # Their coords are ORIGINAL voxel space already (predict_video scales by ds_arr
        # before returning), which is the space our Tracks live in -- no conversion.
        idx = [(int(s), int(t)) for s, t, _p, _d in edges]
        g = our_build_graph(np.asarray(coords, float), idx)
        TIMES.setdefault(det_threshold, []).append(time.time() - t0)
        print(f"    {name:<22} {g.n_nodes:>7,} nodes {g.n_edges:>7,} edges "
              f"({time.time()-t0:.0f}s)", flush=True)
        return g
    return _fn

results = {}
for th in DET_THRESHOLDS:
    print(f"\n=== det_threshold={th} (ILP off for run 1) ===", flush=True)
    t0 = time.time()
    res = h.evaluate(make_predictor(th), arm=f"pack_th{th}", names=SUBSET, verbose=False)
    s = res.summary
    results[th] = res
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    print(f"  SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  nodes={n:,}  ({time.time()-t0:.0f}s)", flush=True)
""")

code(r"""
CHAMPION_CV = 0.7070      # notes/14, and the 0.752 LB submission
OURS_BEST_LEARNED = 0.6490

print(f"{'arm':<18} {'SCORE':>8} {'edge_J':>8} {'recall':>7} {'s/dataset':>10}")
print("-" * 56)
for th, res in results.items():
    s = res.summary
    t = np.mean(TIMES.get(th, [0])) if TIMES.get(th) else float("nan")
    print(f"{'pack th='+str(th):<18} {s['score']:>8.4f} {s['edge_jaccard']:>8.4f} "
          f"{s['node_recall']:>7.3f} {t:>10.1f}")
print(f"{'champion (ours)':<18} {CHAMPION_CV:>8.4f} {'0.7128':>8} {'0.866':>7} {'~30':>10}")
print(f"{'best learned (ours)':<18} {OURS_BEST_LEARNED:>8.4f} {'0.6556':>8} {'0.885':>7} {'—':>10}")

best_th = max(results, key=lambda k: results[k].score) if results else None
if best_th is not None:
    best = results[best_th].score
    print(f"\nbest pack arm: {best:.4f} at det_threshold={best_th}")
    print(f"  vs our champion {CHAMPION_CV:.4f}: {best - CHAMPION_CV:+.4f}")
    per_ds = np.mean([t for ts in TIMES.values() for t in ts])
    print(f"\ncost: {per_ds:.1f} s/dataset on {DEV}")
    print(f"  60-dataset CV run would take {per_ds*60/3600:.2f} h")
    print(f"  a ~200-dataset hidden test set would take {per_ds*200/3600:.2f} h "
          f"(12 h submission ceiling)")
    if per_ds * 200 > 10.5 * 3600:
        print("  !! that does not fit a scored rerun — batching or a faster path is "
              "needed before any submission is possible")

(WORK / "claude_cluster_repro.json").write_text(json.dumps({
    "scores": {str(k): v.score for k, v in results.items()},
    "summaries": {str(k): dict(v.summary) for k, v in results.items()},
    "sec_per_dataset": {str(k): float(np.mean(v)) for k, v in TIMES.items()},
    "n_datasets": len(SUBSET), "subset": SUBSET,
    "champion_cv": CHAMPION_CV, "window_size": int(window_size),
    "downsample": list(map(int, downsample)), "n_params": int(n_par),
    "device": str(DEV), "hours": (time.time() - T_START) / 3600,
}, indent=2, default=float))
print("\nwrote claude_cluster_repro.json")
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
