"""Build notebooks/claude_config_sweep2.ipynb — the settings the audit found untested.

`notes/39` asked why progress was slow and the answer was a bad diagnosis held too long.
`notes/33` attributed ~0.04 to the two model datasets we do not attach; two mechanisms from
that thesis have now been measured at a ~0.002 ceiling (deepcenter, `notes/34`) and +0.0036
(bidirectional, which needs no second model at all). That is ~10 % of the claimed gap.

Meanwhile an audit of the 0.927 notebook's own config against ours found four settings that
every 0.926+ team uses and we have **never tested**:

    DET_THRESHOLD          theirs 0.96875   ours 0.99      never swept
    GAP_CLOSE_MAX_GAP      theirs 2         ours 1         never tested
    OUTPUT_MIN_TRACK_LEN   theirs 6         ours none      never tested
    GAP_DENSITY_ADAPTIVE   theirs on        ours off       never tested

`DET_THRESHOLD` is the worst of them: `notes/28` froze it at 0.99 deliberately, so one
leaderboard delta would isolate the repair, and it was never unfrozen. `notes/35` §3 then
measured the node budget as worth ~0.01 on its own.

This run stops testing one mechanism per launch. `DET_THRESHOLD` changes detection so it
needs a prediction pass per value; everything else is post-processing on the resulting
graph and is free. So: predict at four thresholds, and sweep the post-processing options
over each cached graph -- four GPU passes and a few hundred cheap re-solves in one run,
instead of four runs.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_budget.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 36
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# notes/40: 0.975 won and is INTERIOR, so refine around it rather than re-scan.
# pool_kernel_um: the NMS radius in the pack's detection line, default 3.0 and
# never swept. Node count falls roughly as its cube, which is what reaches the
# regime the budget multiplier rewards.
POOL_GRID = [3.0, 6.0, 10.0, 15.0, 22.0]
DET_FIXED = 0.975          # notes/44's threshold, held while the kernel varies
# post-processing, free on each cached graph: (min_track_frames, gap_max_frames)
# notes/40 §4: max_gap was on the boundary (only 1 and 2 tried, 2 won everywhere) and
# min_len was coarse (6 beat 3 everywhere it mattered). Both extended; min_len=0 at
# gap=2 stays as the within-run reference for what pruning adds.
POST_GRID = [(6, 2), (0, 1), (6, 1), (0, 2), (8, 2), (6, 3)]
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_DATASETS__", str(N_DATASETS))
              .replace("__BLEND_W__", repr(BLEND_W))
              .replace("__POOL_GRID__", repr(POOL_GRID))
              .replace("__DET_FIXED__", repr(DET_FIXED))
              .replace("__POST_GRID__", repr(POST_GRID)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# The node budget: the term we have never touched

```
0.901  submitted        0.926 bronze     0.944 gold
measurable > 0.0015      worth a slot > 0.01        (notes/44)
```

`notes/45`, read out of the official scorer rather than assumed:

```python
ADJUSTMENT_ALPHA = 0.1
J_adj = max(0, J · (1 − ADJUSTMENT_ALPHA · (N_pred − N_total) / N_total))
```

`N_pred → 0` gives a multiplier of **1.1**. Ours is **1.0012** — we predict almost exactly
`N_total`, the estimated *true* cell count, and collect none of it.

And over-prediction is otherwise free, which is why nobody noticed. `pred_valid =
out_valid | in_valid`, both taken from the GT node a prediction matched to and
`fill_null(False)` otherwise, so a predicted edge counts as TP **or** FP only if an
endpoint matched a tracked ground-truth node. Everything else is excluded, not penalised.
That is how 22,000 predicted edges against ~600 real ones still reads `edge_J = 0.935`.

Of our ~24,000 predicted nodes, roughly **670 match ground truth and ~23,300 are pure
budget cost with zero edge benefit.**

`altervation/biohub-r35-spotiflow` — a complete MIT-licensed solution from a team
plausibly at rank 35 — caps detections at **1–25 cells per frame**. We emit about 220.

## The knob, and why the config sweeps never found it

Detection in the pack is one line:

```python
is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)
```

Every sweep this project ran moved `det_threshold`, and `notes/44` found that surface flat.
It is flat: thresholds from 0.965 to 0.99 change the node count by **5.6%**, so all of it
lived inside the predict-everything regime where the multiplier is pinned near 1.0 by
construction. We measured a plateau at high resolution and never stepped off it.

`pool_kernel_um` is the other half of that line — the non-maximum-suppression radius,
sitting at its default **3.0 µm**, never once swept. It controls how many peaks survive,
and node count falls roughly as its cube. That is the lever that reaches the regime the
multiplier rewards.

## What this run measures

`pool_kernel_um` ∈ {3, 6, 10, 15, 22} µm at the located `det_threshold=0.975`, on 36
datasets, with **the components reported separately**: `edge_jaccard`, `node_recall`,
`total_node_ratio`, the multiplier, and `adj_edge_jaccard`. The whole point is the shape of
the trade, not one number — the multiplier rises as `edge_J` falls, and where they cross
is the answer.

## Pre-registered predictions

1. **The anchor arm reproduces.** `pool 3.0, m6 g2` is exactly what we run today; it must
   land within 0.003 of `claude_widecv`'s 0.9348 or nothing below is comparable.
2. **Node count falls more than 5× across the grid.** `det_threshold` moved it 5.6%; if the
   kernel cannot do better, the lever does not exist and this closes cheaply.
3. **`total_node_ratio` goes below −0.5 at the largest kernel** — we actually enter the
   regime where the multiplier pays, rather than probing another plateau edge.
4. **`adj_edge_jaccard` is non-monotonic in the kernel** — it should rise, then fall, as the
   multiplier gain gives way to lost edges. Monotone down means the trade never pays;
   monotone up means the optimum is past 22 µm and the grid was too timid.
5. **The best arm beats the incumbent by more than 0.01** — `notes/44`'s bar for being worth
   a submission slot, and about three times what n=36 can resolve.

*Prediction 5 is the only outcome test here. 1 to 4 are about whether the experiment is
even looking at the thing it claims to, which is what `notes/42` said pre-registration is
actually for.*
""")

# The submission notebook's setup cell already handles the pack wheels, the torch
# wheelhouse and the gpu_ok probe, all of which this run needs identically. Lifting it
# keeps one copy of that logic rather than a second that can drift.
_s = SETUP.read_text()
_a = _s.index('code(r"""\nimport os, subprocess, sys, time, json')
_b = _s.index('"""', _s.index('if "gpu_ok True" not in probe.stdout:')) + 3
_setup = _s[_a + len('code(r"""'):_b - 3]
# This run scores against ground truth, so it needs train/ not test/, and it wants the
# candidate cache only to read WHICH datasets notes/35 measured.
_setup = _setup.replace('and any((p / "test").glob("*.zarr")), ["/kaggle/input"])',
                        'and any((p / "train").glob("*.zarr")), ["/kaggle/input"])')
_setup = _setup.replace(
    'TEST = COMP / "test"',
    'TRAIN = COMP / "train"\n'
    '# Only to reuse notes/35\'s dataset list so the control is comparable; the candidates\n'
    '# themselves are re-predicted here, not read from it.\n'
    'CACHE = find_dir(lambda p: any(p.glob("cand_*.npz")), ["/kaggle/input"])\n'
    'print(f"  {\'cand cache\':<14} {CACHE}")\n'
    '# The ILP at 0.4/2.0 emits forks BY DESIGN (notes/35: div_J 0.1154), and purescore is\n'
    '# only exact without them, so Harness.score_graph requires the official scorer.\n'
    'CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")\n'
    'if not (CELLMOT / "src" / "tracking_cellmot").is_dir():\n'
    '    _r = sh("git", "clone", "--depth", "1",\n'
    '            "https://github.com/royerlab/kaggle-cell-tracking-competition", str(CELLMOT))\n'
    '    print(f"official scorer clone rc={_r.returncode}")\n'
    'os.environ["CELLMOT_REPO"] = str(CELLMOT)\n'
    'if not (CELLMOT / "src" / "tracking_cellmot").is_dir():\n'
    '    raise SystemExit("official scorer not available; forked predictions cannot be scored")')
code(_setup)

md("""
## 1. Patch `predict_video`, then run the arms

`inspect.getsource` on the pack's own function, `patch_source` to insert the reverse pass,
`exec` into a **copy** of the pack module's namespace. The copy matters: mutating the pack
in place would make the control arm depend on run order.
""")

WORKER_BODY = r'''
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {cellmot!r}
PACK = Path({pack!r}); REPO = Path({repo!r}); TRAIN = Path({train!r})
CACHE = {cache!r}; WORK = Path({work!r})
N_DATASETS = __N_DATASETS__
BLEND_W = __BLEND_W__
POOL_GRID = __POOL_GRID__
DET_FIXED = __DET_FIXED__
POST_GRID = [tuple(g) for g in __POST_GRID__]
T0 = time.time()

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))
# The pack's entry point is a SCRIPT, not a package, and it imports a `dataspec` module
# that only exists in the authors' own environment. claude_submit_ratio injects a synthetic
# one; copied from there rather than retyped, which is how v1 of this notebook came to
# import a module name that does not exist.
import types
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"; _ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"; _ds.DATASET_PATH = TRAIN
_ds.PREDICTIONS_PATH = WORK / "predictions"
sys.modules["dataspec"] = _ds

import inspect
import torch
import tracksdata as td
from harness import Harness
from harness.tracks import Tracks, read_geff, read_scale
from harness.purescore import summarise
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy
from pipeline.repair import close_gaps, linefit_smooth, prune_short_tracks
from pipeline.bidirectional import ANCHOR, harmonic_blend, patch_source
import predict_unet_transformer as P
print("worker numpy", np.__version__, "torch", torch.__version__, flush=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ILP_EDGE_W, ILP_APP_W, ILP_DIS_W, ILP_DIV_W = -1.0, 0.4, 2.0, 1.0   # notes/36: the optimum
DET_THRESHOLD = 0.99

def repair_chain(g, sc):
    # A docstring here would terminate the outer f-string that writes this file.
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038, max_added_abs=1650)
    return linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

ORIG_SRC = inspect.getsource(P.predict_video)
if ANCHOR not in ORIG_SRC:
    # Print the neighbourhood so the anchor can be fixed in ONE round trip rather than by
    # guessing. patch_source would catch it, but only after a GPU has been spent.
    i = ORIG_SRC.find("predict_edges")
    print("!! ANCHOR DOES NOT MATCH. predict_video source near predict_edges:", flush=True)
    print(ORIG_SRC[max(0, i - 500):i + 1000], flush=True)
    raise SystemExit("anchor mismatch -- update pipeline/bidirectional.ANCHOR")
print("anchor matches", ORIG_SRC.count(ANCHOR), "x in", len(ORIG_SRC), "chars", flush=True)

def make_predict(weight):
    # w=0 uses the ORIGINAL function object, so the control cannot differ from the
    # unpatched pipeline by even a rounding step.
    if weight == 0.0:
        return P.predict_video
    ns = dict(P.__dict__)
    exec(compile(patch_source(ORIG_SRC, weight), "<bidir>", "exec"), ns)
    return ns["predict_video"]

WPATH = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WPATH, DEV)
print("model params", sum(p.numel() for p in model.parameters()),
      "window", window_size, "downsample", downsample, flush=True)
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      ilp_edge_weight=ILP_EDGE_W, ilp_appearance_weight=ILP_APP_W,
                      ilp_disappearance_weight=ILP_DIS_W, ilp_division_weight=ILP_DIV_W)

# The same datasets notes/35 measured, so w=0 is comparable to 0.9179. The fallback is
# stratified -- notes/34's lesson: names[:12] sorted alphabetically gave 10 44b6 and 2
# 6bba, inverting a 71/128 population split.
seed_names = []
if CACHE:
    seed_names = sorted(p.stem[5:] for p in Path(CACHE).glob("cand_*.npz"))
    seed_names = [n for n in seed_names if (TRAIN / (n + ".geff")).exists()]
alln = sorted(p.stem for p in TRAIN.glob("*.zarr")
              if (TRAIN / (p.stem + ".geff")).exists())
print("pool", len(alln), "datasets | seed (already measured)", len(seed_names), flush=True)
rest = [n for n in alln if n not in set(seed_names)]
a = [n for n in rest if n.startswith("44b6")]
b = [n for n in rest if not n.startswith("44b6")]
need = max(0, N_DATASETS - len(seed_names))
# Proportional to the 71/128 population split, on the datasets not already taken.
k = min(len(a), max(0, round(need * len(a) / max(len(a) + len(b), 1))))
names = seed_names + a[:k] + b[:need - k]
# Stratify whatever list we ended up with. notes/34 recorded that names[:12] taken
# alphabetically inverted the embryo split; v2 of THIS notebook did it again, because the
# stratified branch only ran when the cache was missing. Slice proportionally, always.
# Deliberately NOT re-sliced here. sweep2 re-stratified after selection, which would
# drop seed datasets and break the superset property this run depends on. The selection
# above is already proportional; assert that rather than silently re-cutting it.
names = names[:N_DATASETS]
n44 = sum(n.startswith("44b6") for n in names)
print(len(names), "datasets:", n44, "x 44b6,", len(names) - n44, "x 6bba", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)

def repair_at(g, sc, min_len, gap_max):
    # The submitted chain, with the two audited knobs exposed. min_len=0 / gap_max=1 is
    # byte-identical to what scored 0.897 -- probes/exec_config.py pins that on 30 random
    # graphs, so the anchor cell really is the current submission.
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=gap_max)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)
    if min_len > 0:
        r = prune_short_tracks(*r, min_frames=min_len, keep_division_components=True)
    return r

LABELS = ["p" + str(d) + "_m" + str(m) + "_g" + str(g)
          for d in POOL_GRID for m, g in POST_GRID]
ROWS = dict((l, []) for l in LABELS); ANAT = dict((l, []) for l in LABELS)
NODES = dict((l, 0) for l in LABELS); EDGES = dict((l, 0) for l in LABELS)
CAND = dict(("p" + str(d), 0) for d in POOL_GRID); PER = {{}}
pv = make_predict(BLEND_W)
print("blend", BLEND_W, "| thresholds", POOL_GRID, "|", len(POST_GRID),
      "post combos =", len(LABELS), "cells", flush=True)

BUDGET_S = 9.5 * 3600
for name in names:
    t0 = time.time()
    done = len(PER)
    if done >= 3:
        per = (time.time() - T0) / done
        if time.time() - T0 + per * 1.3 > BUDGET_S:
            print("stopping at " + str(done) + " datasets: another would cost ~"
                  + str(int(per)) + "s and the budget is " + str(int(BUDGET_S)) + "s",
                  flush=True)
            break
    sc = read_scale(TRAIN / (name + ".zarr"))
    gt = read_geff(TRAIN / (name + ".geff"))
    parts = [name]
    for det in POOL_GRID:
        # det_threshold is FIXED here; `det` is the pool kernel. Sweeping both at
        # once would confound the two halves of the same detection line.
        cfg_d = P.PredictConfig(det_threshold=DET_FIXED, pool_kernel_um=det, use_ilp=True,
                                ilp_edge_weight=ILP_EDGE_W,
                                ilp_appearance_weight=ILP_APP_W,
                                ilp_disappearance_weight=ILP_DIS_W,
                                ilp_division_weight=ILP_DIV_W)
        coords, edges = pv(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg_d,
                           window_size=window_size, unet_batch_size=8,
                           downsample=downsample)
        g_td = P.build_graph(coords, edges)
        CAND["p" + str(det)] += int(g_td.num_edges())
        if g_td.num_edges():
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APP_W, disappearance_weight=ILP_DIS_W,
                division_weight=ILP_DIV_W)
            with P.suppress_output():
                g_td = solver.solve(g_td)
        tr = Tracks.from_tracksdata(g_td)
        base = (tr.t, tr.zyx, tr.edges)
        best_here = None
        for min_len, gap_max in POST_GRID:
            lbl = "p" + str(det) + "_m" + str(min_len) + "_g" + str(gap_max)
            g = repair_at(base, sc, min_len, gap_max)
            ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            NODES[lbl] += int(len(g[0])); EDGES[lbl] += int(len(g[2]))
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[lbl].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(name + "/" + lbl + ": buckets do not sum")
            _r = ROWS[lbl][-1]
            v = float(_r.get("score", float("nan")))
            if v != v:
                v = float(_r.get("adj_edge_jaccard", float("nan")))
            PER.setdefault(name, {{}})[lbl] = v
            if best_here is None or v > best_here[0]:
                best_here = (v, lbl)
        parts.append("p" + str(det) + " " + format(best_here[0], ".4f"))
    print("  " + "  ".join(parts) + "   " + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "pool_grid": POOL_GRID, "det_fixed": DET_FIXED, "post_grid": [list(g) for g in POST_GRID],
           "blend_w": BLEND_W, "datasets": [n for n in names if n in PER],
           "seed_datasets": seed_names,
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "budget.json").write_text(json.dumps(out, indent=2, default=float))

print("worker done in", int(time.time() - T0), "s", flush=True)
'''

code('''
import subprocess, sys, time
WORKER = WORK / "run_bidir.py"
WORKER.write_text(""" + BODY + """.format(
    pack=str(PACK), repo=str(REPO), train=str(TRAIN), cache=str(CACHE or ""),
    work=str(WORK), cellmot=str(CELLMOT)))

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

md('''
## 2. Five predictions, with the components reported separately

The multiplier rises as `edge_jaccard` falls, so a single number hides the mechanism.
`edge_jaccard`, `node_recall`, `total_node_ratio` and `adj_edge_jaccard` are printed apart,
and the trade is read off their shapes.
''')

code(r'''
import numpy as np, json, math
D = json.loads((WORK / "budget.json").read_text())
S, N, E, C = D["summary"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
POOLS, POST = D["pool_grid"], [tuple(g) for g in D["post_grid"]]
ANCHOR = "p3.0_m6_g2"          # the default kernel: exactly what we run today
WIDECV = 0.9348                 # claude_widecv, same chain, n=60
NN = len(PER)
EXACT = ANCHOR in S and S[ANCHOR]["score"] == S[ANCHOR]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{NN} datasets, {len(POOLS)} pool kernels x {len(POST)} post = {len(ARMS)} cells")

def cell(p, m, g):
    return "p" + str(p) + "_m" + str(m) + "_g" + str(g)

m0, g0 = POST[0]
print(f"\n{'pool um':<10}{'score':>10}{'adj_edge':>10}{'edge_J':>9}{'node_rec':>10}"
      f"{'ratio':>9}{'mult':>8}{'nodes':>12}")
print("-" * 78)
ROWS = []
for p in POOLS:
    c = cell(p, m0, g0)
    if c not in S:
        continue
    s = S[c]
    n = N.get(c, 0)
    ratio = s.get("total_node_ratio", float("nan"))
    if ratio != ratio and s.get("edge_jaccard") and s.get("adj_edge_jaccard"):
        # derive it when the column is absent: adj = J * (1 - 0.1 * ratio)
        ratio = (1.0 - s["adj_edge_jaccard"] / s["edge_jaccard"]) / 0.1
    mult = 1.0 - 0.1 * ratio if ratio == ratio else float("nan")
    ROWS.append((p, s, n, ratio, mult))
    print(f"{p:<10}{s.get(key, float('nan')):>10.4f}"
          f"{s.get('adj_edge_jaccard', float('nan')):>10.4f}"
          f"{s.get('edge_jaccard', float('nan')):>9.4f}"
          f"{s.get('node_recall', float('nan')):>10.4f}"
          f"{ratio:>9.3f}{mult:>8.4f}{n:>12,}")

print(f"\nall {len(ARMS)} cells")
hdr = "".join(f"m{m}g{g}".rjust(11) for m, g in POST)
print(f"{'pool':<10}{hdr}")
for p in POOLS:
    row = "".join(f"{S[cell(p,m,g)][key]:>11.4f}" if cell(p, m, g) in S else " " * 11
                  for m, g in POST)
    print(f"{p:<10}{row}")

def paired(a, b):
    d = []
    for nm, r in PER.items():
        x, y = r.get(a), r.get(b)
        if x is not None and y is not None and x == x and y == y:
            d.append(x - y)
    if len(d) < 3:
        return None
    n = len(d); m = sum(d) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in d) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    return m, sd, se, (m / se if se else float("inf")), n

best = max((a for a in ARMS if a in S), key=lambda a: S[a][key])
inc = S.get(ANCHOR, {}).get(key, float("nan"))
print(f"\nbest {best} = {S[best][key]:.4f}   incumbent {ANCHOR} = {inc:.4f}")
print(f"\n{'arm':<16}{'mean d':>10}{'sd':>9}{'SE':>9}{'t':>8}   verdict")
for a in sorted((x for x in ARMS if x in S and x != ANCHOR), key=lambda x: -S[x][key])[:10]:
    r = paired(a, ANCHOR)
    if r is None:
        continue
    m, sd, se, t, n = r
    print(f"{a:<16}{m:>+10.4f}{sd:>9.4f}{se:>9.4f}{t:>8.2f}   "
          f"{'RESOLVED' if abs(t) > 2.0 else 'not resolved'}")

print("\n" + "=" * 92)
print("PREDICTION GRADING")
print("=" * 92)

print("\n1. the anchor arm (pool 3.0, the current default) reproduces widecv's 0.9348")
if not EXACT:
    print("   NOT GRADED — score is NaN")
else:
    ok1 = abs(inc - WIDECV) <= 0.003
    print(f"   {ANCHOR} = {inc:.4f} vs {WIDECV:.4f}  ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   pool_kernel_um=3.0 IS the current default, so this arm should be the")
        print("   chain we already measured. A miss means the grid is not anchored and")
        print("   nothing below compares to anything already scored. (A modest gap is")
        print("   expected from n=36 vs n=60 on a different dataset draw.)")

print("\n2. node count falls more than 5x across the kernel grid")
ns = [n for _, _, n, _, _ in ROWS if n]
if len(ns) < 2:
    print("   NOT GRADED — fewer than two arms produced counts")
else:
    fall = max(ns) / max(min(ns), 1)
    ok2 = fall > 5.0
    print(f"   {max(ns):,} -> {min(ns):,}  = {fall:.1f}x  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   det_threshold moved node count 5.6% and the kernel cannot do much")
        print("   better either. The budget regime is unreachable with this detector,")
        print("   and the lever needs a detector that ranks cells, not a wider NMS.")

print("\n3. total_node_ratio goes below -0.5 at the largest kernel")
rs = [r for _, _, _, r, _ in ROWS if r == r]
if not rs:
    print("   NOT GRADED — ratio unavailable")
else:
    ok3 = min(rs) < -0.5
    print(f"   lowest ratio {min(rs):+.3f} (multiplier {1 - 0.1 * min(rs):.4f})"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   We never entered the regime the multiplier rewards, so this run")
        print("   probed another plateau edge and says nothing about the trade.")

print("\n4. adj_edge_jaccard is non-monotonic in the kernel (rises, then falls)")
adj = [s.get("adj_edge_jaccard", float("nan")) for _, s, _, _, _ in ROWS]
adj = [a for a in adj if a == a]
if len(adj) < 3:
    print("   NOT GRADED — need at least three arms")
else:
    i = int(np.argmax(adj))
    ok4 = 0 < i < len(adj) - 1
    print("   adj by kernel: " + "  ".join(f"{a:.4f}" for a in adj))
    print(f"   peak at index {i} of {len(adj)-1}  ->  {'PASS' if ok4 else 'FAIL'}")
    if i == 0:
        print("   Monotone DOWN: thinning never pays, edges are lost faster than the")
        print("   multiplier gains. The budget direction closes here.")
    elif i == len(adj) - 1:
        print("   Monotone UP: the optimum is past 22 um and the grid was too timid.")
        print("   Rerun with a wider grid before concluding anything about the size.")

print("\n5. the best arm beats the incumbent by more than 0.01 (notes/44's bar)")
r5 = paired(best, ANCHOR) if best != ANCHOR else None
if not EXACT:
    print("   NOT GRADED — score is NaN")
elif r5 is None:
    print(f"   best IS the incumbent ({ANCHOR})  ->  FAIL")
    print("   The default kernel is already optimal and the budget term is not")
    print("   reachable by non-maximum suppression.")
else:
    m, sd, se, t, n = r5
    ok5 = m > 0.01
    print(f"   {best} - {ANCHOR} = {m:+.4f}  SE {se:.4f}  t {t:.2f}  n {n}"
          f"  ->  {'PASS' if ok5 else 'FAIL'}")
    print(f"   (n={n} resolves ~{2 * sd / math.sqrt(n):.4f})")

print("\n" + "=" * 92)
print(f"n={NN}  |  best {best} {S[best][key]:.4f}  |  incumbent {ANCHOR} {inc:.4f}")
print("=" * 92)
''')

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
