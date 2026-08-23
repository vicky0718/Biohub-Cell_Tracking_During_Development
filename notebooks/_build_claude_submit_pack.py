"""Build notebooks/claude_submit_pack.ipynb — the cluster reproduction, as a submission.

Everything here is offline. A scored rerun has no network, so every dependency comes from
an attached artefact:

  * the pack's own wheels  -> tracksdata, zarr, polars, ilpy, numpy 2.4.6, ...
  * claude_torch_wheelhouse -> torch 2.5.1+cu121, because the free GPU is a P100 (sm_60)
    and the image torch builds sm_70+ only. Verified to resolve with --no-index.

`notes/24` established the pipeline runs at 34.5 s/dataset on a P100, so a ~200-dataset
rerun projects to ~1.9 h against a 12 h ceiling.

The number this produces is the FIRST honest measurement of the pack's model for us:
`notes/24` §2 showed no CV of it is obtainable, because the splits naming its training
data were never published.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_submit_pack.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Submission — the public pipeline, reproduced

Rules position: **§6.b** — *"The use of external data and models is acceptable unless
specifically prohibited by the Host"* — with **§6.a** satisfied by a public CC0 dataset
that ~514 teams already use. The Competition-Specific Rules section was checked by the
owner and does not restrict external models.

## Why this submission exists

`notes/24` §2: the pack's `dataset_splits.json` — which names the datasets `split_0`
trained on — was never published, is absent from the competition data, and is only *read*
by their training script. So **every one of the 199 training datasets is potentially
contaminated**, and no honest cross-validation of these weights is obtainable. The
leaderboard is the only clean measurement.

Our banked score is **0.752**. The cluster running these weights reports **0.913–0.916**.

## Everything is offline

A scored rerun has no network, and two separate things would otherwise need one:

| need | source |
|---|---|
| tracksdata, zarr, polars, ilpy, numpy 2.4.6 | the pack's own `wheels/` |
| **torch 2.5.1+cu121** | `claude_torch_wheelhouse` kernel output |

The torch wheels are not optional. Measured on Kaggle: the free GPU is a **Tesla P100
(sm_60)**, the image torch 2.10+cu128 builds **sm_70+**, and it dies with `no kernel image
is available for execution on the device`. `machineShape="nvidiaTeslaT4"` is accepted and
**ignored** — the accelerator cannot be chosen. CPU is not viable either: 2.1 M parameters
with per-voxel temporal attention at a 10–30× CPU penalty puts this at 19–58 h.

## Two things this run must get right

- **The work runs in a subprocess.** The wheels upgrade numpy 2.0.2 → 2.4.6, and this
  notebook's process has already imported 2.0.2's compiled extensions. In-process that is a
  hybrid that fails as `_blas_supports_fpe` (`notes/24` §1).
- **Divisions are kept.** Their model predicts them — hundreds per dataset — and the term
  is worth 0.1 of the 1.1 maximum. `write_submission(..., allow_divisions=True)` so the
  graph check does not report every one of them as an out-degree violation.
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
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(),
                ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "test").glob("*.zarr")), ["/kaggle/input"])
# The torch wheelhouse: a directory of .whl files that is NOT the pack's own.
TORCH_WH = find_dir(
    lambda p: p.name == "wheels" and any(x.name.startswith("torch-") for x in p.iterdir()),
    ["/kaggle/input"])

for label, val in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
                   ("torch wheels", TORCH_WH)):
    print(f"  {label:<14} {val}")
missing = [l for l, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP)) if v is None]
if missing:
    raise SystemExit(f"not mounted: {missing}")
TEST = COMP / "test"

# Offline installs, pack wheels first so numpy lands before anything compiles against it.
t0 = time.time()
ok1 = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                  extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok1 else 'FAILED'} ({time.time()-t0:.0f}s)")

if TORCH_WH is None:
    print("!! no torch wheelhouse attached — the P100 cannot run the image torch, so this "
          "will fall back to CPU and will NOT finish inside 12 h.")
else:
    t0 = time.time()
    ok2 = pip_install(["torch==2.5.1"], extra=("--no-index", f"--find-links={TORCH_WH}"))
    print(f"torch wheels {'ok' if ok2 else 'FAILED'} ({time.time()-t0:.0f}s)")

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, tracksdata; "
           "ok=False\n"
           "if torch.cuda.is_available():\n"
           "    try:\n"
           "        w=torch.nn.Conv3d(1,4,3,padding=1).cuda()\n"
           "        _=w(torch.randn(2,1,8,8,8,device='cuda')).sum().item()\n"
           "        torch.cuda.synchronize(); ok=True\n"
           "    except Exception as e: print('GPU BROKEN:', type(e).__name__, str(e)[:120])\n"
           "print('numpy', numpy.__version__, '| torch', torch.__version__, '| gpu_ok', ok)")
print(probe.stdout.strip() or probe.stderr.strip()[-1500:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter — a scored "
                     "rerun would fail identically with no way to recover.")
if "gpu_ok True" not in probe.stdout:
    print("\n!! GPU is not usable. Continuing, but expect this to exceed the time budget; "
          "the guard below will still emit a valid file.")
""")

md("""## 1. Predict every test dataset and stream the CSV

All of it in a subprocess, for the numpy reason above. `test/` is globbed at runtime —
the rerun swaps it for the hidden set — and never hardcoded.
""")

code(r"""
WORKER = WORK / "run_submit.py"
WORKER.write_text(f'''
import json, sys, time, types
from pathlib import Path
import numpy as np

PACK = Path({str(PACK)!r})
REPO = Path({str(REPO)!r})
TEST = Path({str(TEST)!r})
WORK = Path({str(WORK)!r})
DET_THRESHOLD = {0.99}
UNET_BATCH = {4}
TIME_BUDGET_S = {10.5 * 3600}

t_start = time.time()
import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("worker numpy", np.__version__, "torch", torch.__version__, "device", DEV, flush=True)

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"; _ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"; _ds.DATASET_PATH = TEST
_ds.PREDICTIONS_PATH = WORK / "predictions"
sys.modules["dataspec"] = _ds

import predict_unet_transformer as P
from pipeline.classical import build_graph as our_build_graph
from harness.csvout import write_submission

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)
print(f"model {{sum(p.numel() for p in model.parameters()):,}} params, "
      f"window_size={{window_size}}, downsample={{downsample}}", flush=True)

# GLOBBED: the rerun swaps test/ for the hidden set.
names = sorted(p.stem for p in TEST.glob("*.zarr"))
print(f"{{len(names)}} test datasets; first three {{names[:3]}}", flush=True)
if not names:
    raise SystemExit(f"no .zarr under {{TEST}}")

cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=False)

def gen():
    starved = False
    for i, name in enumerate(names, 1):
        # A partial submission scores 0 on what it skips; one that never finishes scores
        # nothing at all. Past the budget, keep emitting rows and stop paying for models.
        if starved or time.time() - t_start > TIME_BUDGET_S:
            if not starved:
                print(f"!! time budget exhausted at {{i}}/{{len(names)}}; the rest are "
                      f"written empty so the run still produces a valid submission",
                      flush=True)
                starved = True
            yield name, our_build_graph(np.zeros((0, 4)), [])
            continue
        t0 = time.time()
        coords, edges = P.predict_video(model, TEST / f"{{name}}.zarr", DEV, cfg=cfg,
                                        window_size=window_size,
                                        unet_batch_size=UNET_BATCH, downsample=downsample)
        g = our_build_graph(np.asarray(coords, float),
                            [(int(s), int(t)) for s, t, _p, _d in edges])
        el = time.time() - t0
        print(f"[{{i:>3}}/{{len(names)}}] {{name:<24}} {{g.n_nodes:>7,}} nodes "
              f"{{g.n_edges:>7,}} edges ({{el:.0f}}s, projected total "
              f"{{(time.time()-t_start)/i*len(names)/3600:.1f}}h)", flush=True)
        yield name, g

SUB = WORK / "submission.csv"
# allow_divisions=True: their model predicts divisions and the term is worth 0.1 of the
# 1.1 maximum, so a fork is the point rather than a defect.
summary = write_submission(gen(), SUB, verbose=False, allow_divisions=True)
print(f"\\n{{summary['rows']:,}} rows | {{summary['datasets']}} datasets | "
      f"{{summary['nodes']:,}} nodes | {{summary['edges']:,}} edges", flush=True)
if summary["problems"]:
    print(f"!! {{len(summary['problems'])}} problem(s):", flush=True)
    for p in summary["problems"][:20]:
        print("   ", p, flush=True)
else:
    print("no malformed-graph problems", flush=True)

(WORK / "submit_pack_summary.json").write_text(json.dumps(
    {{k: v for k, v in summary.items() if k != "names"}}
    | {{"n_test": len(names), "det_threshold": DET_THRESHOLD,
        "hours": (time.time() - t_start) / 3600}}, indent=2))
print("worker done", flush=True)
''')

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"\nworker exited {rc} after {time.time()-t0:.0f}s")
if rc != 0:
    raise SystemExit(f"worker failed (exit {rc})")
""")

code(r"""
# Read the artefact back. Everything above worked on objects; this checks the file.
SUB = WORK / "submission.csv"
with SUB.open() as fh:
    header = fh.readline().strip().split(",")
    n_rows = sum(1 for _ in fh)
expected = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
            "source_id", "target_id"]
print(f"header: {header}")
print(f"rows: {n_rows:,}   size: {SUB.stat().st_size/1e6:.1f} MB")
assert header == expected, f"column mismatch\n  got      {header}\n  expected {expected}"

summary = json.loads((WORK / "submit_pack_summary.json").read_text())
assert n_rows == summary["rows"], f"row count drifted: {n_rows} vs {summary['rows']}"
print(f"\nOK: {summary['datasets']} datasets, {summary['nodes']:,} nodes, "
      f"{summary['edges']:,} edges in {summary['hours']:.2f} h")

print("\n--- what to expect ---")
print("  banked (classical, ours):        0.752")
print("  cluster running these weights:   0.913-0.916 reported")
print("  notes/24 CV on train data:       0.9588  <- CONTAMINATED, not a prediction")
print("\nThe leaderboard number this produces is the first honest measurement of this "
      "model for us. A result near 0.915 confirms the reproduction is faithful; well "
      "below it means something in this path differs from theirs and the gap is the "
      "thing to find.")
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
