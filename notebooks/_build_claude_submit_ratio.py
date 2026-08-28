"""Build notebooks/claude_submit_ratio.ipynb — the 0.880 submission PLUS the ILP asymmetry.

`claude_submit_repair` scored **0.880** with the pack's inherited ILP weights
(-1.0 / 0.1 / 0.1 / 1.0) and the repair chain. This is that submission with **one number
changed**: `disappearance_weight` 0.1 -> 0.5.

`notes/31` measured `asym0.1_0.5 + repair` at **+0.0152** against the **+0.0115** that
scored 0.880 -- the ILP change adds **+0.0037** on top of the repair chain, on 24
budget-stratified training datasets where the control reproduced 0.8806 exactly.

**Why this weight and not another.** `notes/03` §3 recorded the same lab's own zebrafish
Ultrack config using `appear_weight = -0.002` and `disappear_weight = -0.01` -- a
deliberate **5x asymmetry**, "discouraging track termination more than initiation". The
pack ships a symmetric 0.1/0.1. This keeps appearance at the pack's 0.1 and raises
disappearance 5x, which is the lab's own ratio at the pack's scale.

**And it is asymmetry, not magnitude.** The sweep deliberately paired every asymmetric arm
with a SYMMETRIC arm at matched magnitude so the gain could not be misattributed:
asym0.1_0.5 (+0.0043) beats the best symmetric sym0.25 (+0.0026) by +0.0017.

**Stated risk.** `asym0.1_0.5` is the largest asymmetry in the swept grid, so the optimum
may lie beyond it -- this is a measured point, not a located maximum. And `notes/31` §3
records the mechanism cost: a higher disappearance penalty makes the solver link through
ambiguity it previously abandoned, which fixed 29 mislinks and 18 gaps but stranded 36
more nodes as undetected. Net positive here; that ratio is what would invert if pushed.
"""

_ORIGINAL_HEADER = """Build notebooks/claude_submit_repair.ipynb — the reproduction PLUS the measured repair.

This is `claude_submit_pack` with one change: the post-ILP graph goes through the repair
chain `notes/26` §4 and `notes/27` §3 measured at **+0.0115** on 24 training datasets --
gap closing at 5.75 um, then line-fit smoothing at weight 0.76 / window 2.

**Why submit this rather than go straight to motion relink.** Training data is contaminated
for these weights (`notes/24` §2 -- the splits naming their training set were never
published), so +0.0115 is a delta on data the model may have been fitted to. The
leaderboard is the only honest measure of whether graph repair transfers **at all**. If it
does not, that kills the repair path before days go into relink; if it does, it banks a
better floor. One measurement, one slot.

**One thing is deliberately NOT changed:** `DET_THRESHOLD` stays at 0.99, matching the
0.867 submission, even though the repair was tuned against graphs built at 0.985. Changing
the threshold too would confound the measurement -- the delta on the leaderboard has to be
the repair and nothing else. The cost is that the repair's tuning is being applied at a
slightly different density than it was measured at, which is a real if second-order risk.
"""

_ORIGINAL_HEADER = """Build notebooks/claude_submit_pack.ipynb — the cluster reproduction, as a submission.

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
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_submit_ratio.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Submission — the 0.880 chain, plus one ILP weight

**The one change from the 0.880 submission**: the ILP's `disappearance_weight` goes from
the pack's inherited **0.1 to 0.5**. Everything else — detections, `det_threshold=0.99`,
the `close_gaps` + `linefit_smooth` repair chain — is byte-identical.

`notes/31` measured this at **+0.0152** against the **+0.0115** that scored 0.880, on 24
budget-stratified datasets where the control reproduced 0.8806 exactly. The ILP change
contributes **+0.0037** on top of the repair.

It is the asymmetry that pays, not the magnitude: the sweep paired every asymmetric arm
with a symmetric one at matched scale, and `asym0.1_0.5` (+0.0043) beat the best symmetric
(+0.0026) by +0.0017. The 5× ratio is the same lab's own zebrafish Ultrack setting
(`notes/03` §3), which is the first primary-source constant in this project to transfer.

**Risk, stated:** this is the largest asymmetry swept, so it is a measured point rather
than a located optimum — and `notes/31` §3 records the trade it makes (−29 mislinks,
−18 gaps, **+36 undetected**). Net positive at this setting; that ratio inverts if pushed.

---

**The one change from the 0.867 submission**: the post-ILP graph now goes through
`close_gaps(5.75 µm)` then `linefit_smooth(weight 0.76, window 2)`, the chain measured at
**+0.0115** across 24 training datasets in `notes/26` and `notes/27`.

Training data is contaminated for these weights (`notes/24` §2), so that +0.0115 is a
delta on data the model may have been fitted to. **This submission is the only honest test
of whether graph repair transfers at all** — and that answer gates whether motion relink,
a much larger build, is worth starting.

`DET_THRESHOLD` stays at **0.99** to match the 0.867 run, so the leaderboard delta is the
repair and nothing else. Noted risk: the repair was tuned against graphs built at 0.985.

---

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

## The fix from the 0.843 run

The first submission scored **0.843** — well above our banked 0.752, but 0.07 short of the
cluster's 0.913–0.916. The cause: `predict_video()` returns **candidate** edges with
probabilities, and their `predict()` then runs an **ILP** over them to select a consistent
subset (at most one parent, at most two children, paying appearance / disappearance /
division costs). That run called `predict_video()` and went straight to a graph, keeping
*every* candidate edge.

The `use_ilp` flag was set on the config and did nothing, because the solve lives in
`predict()`, which was bypassed. A flag that looks configured and is inert is worse than an
obviously missing one, so the solve is now written out explicitly in the worker rather than
delegated to a flag, and the log prints how many candidates the ILP kept per dataset.

## Three things this run must get right

- **The work runs in a subprocess.** The wheels upgrade numpy 2.0.2 → 2.4.6, and this
  notebook's process has already imported 2.0.2's compiled extensions. In-process that is a
  hybrid that fails as `_blas_supports_fpe` (`notes/24` §1).
- **Divisions are kept.** Their model predicts them — hundreds per dataset — and the term
  is worth 0.1 of the 1.1 maximum. `write_submission(..., allow_divisions=True)` so the
  graph check does not report every one of them as an out-degree violation.
- **The ILP actually runs.** See above. `ILP kept N/M candidates` in the log is the check:
  if N equals M on every dataset, the solve is not doing anything and the result will
  reproduce 0.843.

## Inputs this notebook needs

`pilkwang/biohub-tracking-support-pack-50ep-v1`, `vigneshnehru/biohub-cell-tracking`
(v21+, for `write_submission`'s `allow_divisions`), the competition data, and the
`claude_torch_wheelhouse` kernel output. Internet **off**.
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
import tracksdata as td
from harness.tracks import Tracks, read_scale
from harness.csvout import write_submission
from pipeline.repair import close_gaps, linefit_smooth

# The measured chain. None of these is a free parameter -- every one was swept:
#   gap 5.75 um  8.0 and 11.0 both score WORSE (notes/27 §2): wider closes more gaps and
#               loses more on node budget and new mislinks than it recovers.
#   w 0.76      the sweep's best solo was w1.0/win3 at +0.0092 vs +0.0086 -- a +0.0006
#   window 2    difference -- but the best measured CHAIN used 0.76/2. Putting the tuned
#               solo setting inside an untested combination would trade a measured
#               +0.0115 for an assumed +0.0121. Not on a submission.
#   order       gap-close BEFORE smooth: +0.0115 vs +0.0109 reversed, because smoothing
#               afterwards also refines the inserted midpoint nodes.
GAP_UM, SMOOTH_W, SMOOTH_WIN, SMOOTH_SHIFT = 5.75, 0.76, 2, 3.2

def repair(g, scale):
    r = close_gaps(g.t, g.zyx, g.edges, scale=scale, max_um=GAP_UM,
                   max_added_frac=0.038, max_added_abs=1650)
    r = linefit_smooth(*r, window=SMOOTH_WIN, weight=SMOOTH_W, scale=scale,
                       max_shift_um=SMOOTH_SHIFT)
    return Tracks(r[0], r[1], r[2])

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)
print(f"model {{sum(p.numel() for p in model.parameters()):,}} params, "
      f"window_size={{window_size}}, downsample={{downsample}}", flush=True)

# GLOBBED: the rerun swaps test/ for the hidden set.
names = sorted(p.stem for p in TEST.glob("*.zarr"))
print(f"{{len(names)}} test datasets; first three {{names[:3]}}", flush=True)
if not names:
    raise SystemExit(f"no .zarr under {{TEST}}")

# ILP ON. This is the fix for the 0.843 submission.
#
# predict_video() returns CANDIDATE edges with probabilities; their predict() then runs an
# ILP over them to select a consistent subset -- at most one parent, at most two children,
# paying appearance / disappearance / division costs. The previous submission called
# predict_video() and went straight to a graph, keeping EVERY candidate edge, and scored
# 0.843 against the cluster's 0.913-0.916.
#
# The use_ilp flag was set on the config even then, and did nothing, because the solve
# lives in predict() which was bypassed. A flag that looks configured and is inert is
# worse than an obviously missing one, so the solve is now written out explicitly here
# rather than delegated to a flag.
#
# Weights are their defaults, which notes/15 §3 also read off the public notebook:
#   edge -1.0 * edge_prob, appearance 0.1, disappearance 0.1, division 1.0
# THE ONE CHANGE from the 0.883 submission: appearance 0.1 -> 0.4, disappearance 0.5 -> 2.0.
# notes/35: ratio0.4_2.0 + repair measured 0.9179 vs the 0.8958 that scored 0.883, on the
# same 24 cached instances -- +0.0221, five times the largest gain measured before it.
#
# The reason is NOT the one the 0.883 submission gave. notes/35 §1 retracts that: sweep2's
# properly matched controls show the symmetric arm BEATING the 1:5 ratio at magnitude 0.5
# (0.8980 vs 0.8849), so the earlier +0.003 was a MAGNITUDE effect, not the lab's ratio
# transferring. What this arm buys, per notes/35 §3, is two things in equal measure:
#   * div_J 0.0000 -> 0.1154. A high termination penalty makes the solver fork rather than
#     end a track, which is what finally scores on the division term (+0.0115).
#   * the node-budget multiplier: ~10% OVER budget -> ~2% under it (+0.0106).
# Raw linking quality is unchanged (edge_J 0.9050 -> 0.9047).
#
# Stated risk: the budget half is a calibration, and 2% under on train could be past the
# optimum on a hidden set where the model drops more nodes anyway. And notes/35 §4 records
# that every axis was STILL climbing at this grid's boundary, so this is a measured point,
# not a located maximum -- claude_ilp_sweep3 is searching past it now.
ILP_EDGE_W, ILP_APP_W, ILP_DIS_W, ILP_DIV_W = -1.0, 0.4, 2.0, 1.0
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      ilp_edge_weight=ILP_EDGE_W,
                      ilp_appearance_weight=ILP_APP_W,
                      ilp_disappearance_weight=ILP_DIS_W,
                      ilp_division_weight=ILP_DIV_W)

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
            yield name, Tracks(np.zeros(0), np.zeros((0, 3)), np.zeros((0, 2), int))
            continue
        t0 = time.time()
        coords, edges = P.predict_video(model, TEST / f"{{name}}.zarr", DEV, cfg=cfg,
                                        window_size=window_size,
                                        unet_batch_size=UNET_BATCH, downsample=downsample)
        # THEIR build_graph, which attaches edge_prob -- the ILP objective reads it.
        g_td = P.build_graph(coords, edges)
        n_cand = g_td.num_edges()
        if g_td.num_edges() > 0:
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APP_W,
                disappearance_weight=ILP_DIS_W,
                division_weight=ILP_DIV_W,
            )
            with P.suppress_output():
                g_td = solver.solve(g_td)
        g = Tracks.from_tracksdata(g_td)
        raw_n, raw_e = g.n_nodes, g.n_edges
        # THE ONE CHANGE from claude_submit_pack. Per-dataset scale, not a constant: the
        # radii are in um and every dataset's voxel size is read from its own zarr.
        g = repair(g, read_scale(TEST / f"{{name}}.zarr"))
        el = time.time() - t0
        print(f"[{{i:>3}}/{{len(names)}}] {{name:<24}} {{g.n_nodes:>7,}} nodes "
              f"{{g.n_edges:>7,}} edges (ILP kept {{raw_e}}/{{n_cand}} candidates, "
              f"repair {{g.n_nodes-raw_n:+,}} nodes {{g.n_edges-raw_e:+,}} edges, "
              f"{{el:.0f}}s, projected total "
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

(WORK / "submit_ilp_summary.json").write_text(json.dumps(
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

summary = json.loads((WORK / "submit_ilp_summary.json").read_text())
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
