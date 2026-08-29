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

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_widecv.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 60
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# notes/40: 0.975 won and is INTERIOR, so refine around it rather than re-scan.
DET_GRID = [0.98, 0.975, 0.97]
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
              .replace("__DET_GRID__", repr(DET_GRID))
              .replace("__POST_GRID__", repr(POST_GRID)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# More data — because at n=12 we cannot see what we are chasing

```
0.899  submitted        rank ~1297/2792
0.926  bronze           0.944  gold
```

Every measurement since `notes/34` has been taken on **12 of the 199** training datasets.
Taking `claude_config_sweep2`'s own per-dataset numbers and running a **paired** test across
those 12 — pairing cancels dataset difficulty, which is enormous here — gives this:

```
comparison              mean Δ       t      verdict
det 0.975 vs 0.98      +0.0021    1.24     not resolved
det 0.975 vs 0.97      +0.0018    1.94     not resolved
det 0.975 vs 0.965     +0.0014    2.32     resolved, barely
```

**At n=12 the smallest effect this design can resolve is 0.0036.** Everything chased since
`notes/34` is +0.001 to +0.002. So `notes/40` and `notes/41` ranked cells that the
measurement cannot distinguish from one another, and called the winner an interior optimum
— twice, on two grids, both of which were reading noise.

That also explains the transfer ratio nobody could pin down. Four measurements gave
implied ratios of (1.04, 1.22), (0.54, 1.08), (0.59, 0.68), (0.28, 0.83) — ranges that do
not overlap. Dividing an unresolved train delta by a rounded leaderboard delta produces
exactly that: noise over noise.

## What n buys

With the paired sd measured at 0.0058:

```
effect    datasets needed
0.0036         11        <- what we have
0.0020         34
0.0015         61
0.0010        136
```

This run takes n to **60**, which resolves ~0.0015 — the size of the effects actually on
the table.

## The sample is a superset, deliberately

The 12 datasets every prior result was measured on are taken **first**, then filled to 60
from the rest of the pool, proportional to the 71/128 embryo split. A fresh independent
draw would confound "more data" with "different data" and settle nothing: prediction 1
below re-computes the incumbent on the 12 seed datasets alone and requires `notes/40`'s
0.9535 back.

## What this does not fix

The published weights record `train_datasets: 199` — the model was fitted on **every**
competition training dataset. So these scores are in-sample, and no amount of extra
training-set data makes them held-out. That is a bias in the level (train 0.95 → LB 0.899),
not in the paired *differences* this run is designed to measure, but it is the reason the
absolute number should never be read as a leaderboard prediction.

A time guard stops the loop while there is still time to print. n=60 is roughly five times
sweep2's runtime and its slowest dataset took 589 s against a 280 s mean; a Kaggle timeout
loses everything, whereas a partial n=45 answers the question nearly as well.
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
DET_GRID = __DET_GRID__
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

LABELS = ["d" + str(d) + "_m" + str(m) + "_g" + str(g)
          for d in DET_GRID for m, g in POST_GRID]
ROWS = dict((l, []) for l in LABELS); ANAT = dict((l, []) for l in LABELS)
NODES = dict((l, 0) for l in LABELS); EDGES = dict((l, 0) for l in LABELS)
CAND = dict(("d" + str(d), 0) for d in DET_GRID); PER = {{}}
pv = make_predict(BLEND_W)
print("blend", BLEND_W, "| thresholds", DET_GRID, "|", len(POST_GRID),
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
    for det in DET_GRID:
        cfg_d = P.PredictConfig(det_threshold=det, use_ilp=True,
                                ilp_edge_weight=ILP_EDGE_W,
                                ilp_appearance_weight=ILP_APP_W,
                                ilp_disappearance_weight=ILP_DIS_W,
                                ilp_division_weight=ILP_DIV_W)
        coords, edges = pv(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg_d,
                           window_size=window_size, unet_batch_size=8,
                           downsample=downsample)
        g_td = P.build_graph(coords, edges)
        CAND["d" + str(det)] += int(g_td.num_edges())
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
            lbl = "d" + str(det) + "_m" + str(min_len) + "_g" + str(gap_max)
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
        parts.append("d" + str(det) + " " + format(best_here[0], ".4f"))
    print("  " + "  ".join(parts) + "   " + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "det_grid": DET_GRID, "post_grid": [list(g) for g in POST_GRID],
           "blend_w": BLEND_W, "datasets": [n for n in names if n in PER],
           "seed_datasets": seed_names,
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "widecv.json").write_text(json.dumps(out, indent=2, default=float))

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
## 2. Five predictions, graded with a paired test

At n=12 the smallest effect this measurement can resolve is **0.0036**, computed from the
paired per-dataset spread in `claude_config_sweep2`'s own log. Everything chased since
`notes/34` is +0.001 to +0.002 — so `notes/40` and `notes/41` ranked cells the measurement
cannot tell apart, and called the winner a located optimum.

Grading below is on the **paired** difference across datasets. Pairing cancels dataset
difficulty, which is enormous here (per-dataset scores span 0.85–1.06, sd 0.064) next to
the effects being tested (0.002). An unpaired comparison at this n cannot see anything.
''')

code(r'''
import numpy as np, json, math
D = json.loads((WORK / "widecv.json").read_text())
S, A, N, E, C = D["summary"], D["anatomy"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
DETS, POST = D["det_grid"], [tuple(g) for g in D["post_grid"]]
ANCHOR = "d0.975_m6_g2"     # notes/40's cell, 0.9535 on the 12 seed datasets
SEED_N12 = 0.9535
EXACT = ANCHOR in S and S[ANCHOR]["score"] == S[ANCHOR]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(DETS)} thresholds x {len(POST)} post = {len(ARMS)} cells")
if not EXACT:
    print("!! score column is NaN — grading on `edge_jaccard`.")

def cell(d, m, g):
    return "d" + str(d) + "_m" + str(m) + "_g" + str(g)

hdr = "".join(f"m{m}g{g}".rjust(10) for m, g in POST)
print(f"\n{'det':<10}{hdr}{'nodes':>11}{'cand':>11}")
print("-" * (10 + 10 * len(POST) + 22))
for d in DETS:
    row = "".join(f"{S[cell(d,m,g)][key]:>10.4f}" if cell(d, m, g) in S else " " * 10
                  for m, g in POST)
    n0 = N.get(cell(d, POST[0][0], POST[0][1]), 0)
    print(f"{d:<10}{row}{n0:>11,}{C.get('d' + str(d), 0):>11,}")

def pairs(a, b, subset=None):
    out = []
    for nm, row in PER.items():
        if subset is not None and nm not in subset:
            continue
        x, y = row.get(a), row.get(b)
        if x is not None and y is not None and x == x and y == y:
            out.append(x - y)
    return out

def paired(a, b, subset=None):
    d = pairs(a, b, subset)
    n = len(d)
    if n < 3:
        return None
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    return m, sd, se, (m / se if se else float("inf")), n

def mre(sd, n):
    return 2.0 * sd / math.sqrt(n) if n else float("nan")

best = max((a for a in ARMS if a in S), key=lambda a: S[a][key])
inc = S.get(ANCHOR, {}).get(key, float("nan"))
print(f"\nbest cell: {best} = {S[best][key]:.4f}   incumbent {ANCHOR} = {inc:.4f}")

print("\nPAIRED vs the incumbent (" + ANCHOR + ")")
print(f"{'arm':<16}{'mean d':>10}{'sd':>9}{'SE':>9}{'t':>8}   verdict")
rank = sorted((a for a in ARMS if a in S and a != ANCHOR), key=lambda a: -S[a][key])
sds, resolved = [], []
for a in rank:
    r = paired(a, ANCHOR)
    if r is None:
        continue
    m, sd, se, t, n = r
    sds.append(sd)
    if abs(t) > 2.0:
        resolved.append(a)
    print(f"{a:<16}{m:>+10.4f}{sd:>9.4f}{se:>9.4f}{t:>8.2f}   "
          f"{'RESOLVED' if abs(t) > 2.0 else 'not resolved'}")
SD_TYP = sorted(sds)[len(sds) // 2] if sds else float("nan")
NN = len(PER)
print(f"\ntypical paired sd {SD_TYP:.4f} -> at n={NN} this run resolves "
      f"{mre(SD_TYP, NN):.4f}; at n=12 it resolved {mre(SD_TYP, 12):.4f}")

print("\n" + "=" * 92)
print("PREDICTION GRADING")
print("=" * 92)

SEED = set(D.get("seed_datasets") or [])
print("\n1. the 12 seed datasets still give notes/40's 0.9535 (+-0.002)")
if not EXACT or not SEED:
    print("   NOT GRADED — " + ("score is NaN" if not EXACT else "seed list absent"))
else:
    vals = [PER[nm][ANCHOR] for nm in PER if nm in SEED and ANCHOR in PER[nm]]
    got = sum(vals) / len(vals) if vals else float("nan")
    ok1 = abs(got - SEED_N12) <= 0.002
    print(f"   {ANCHOR} on the {len(vals)} seed datasets = {got:.4f} vs {SEED_N12:.4f}"
          f"  ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   The superset property is broken: this run is not measuring what the")
        print("   earlier runs measured, and no comparison below is comparable to them.")

print("\n2. the paired sd is close to the n=12 estimate (0.0058), so the power")
print("   calculation that motivated this run was sound")
ok2 = SD_TYP == SD_TYP and SD_TYP < 0.0116
print(f"   typical paired sd {SD_TYP:.4f} vs 0.0058  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   The spread is wider than the pilot implied, so even this n does not resolve")
    print("   what we are chasing. That is the finding, and it says stop tuning config.")

print("\n3. det 0.975 vs 0.98 RESOLVES at this n (n=12 gave t=1.24)")
r3 = paired(cell(0.975, 6, 2), cell(0.98, 6, 2))
if r3 is None:
    print("   NOT GRADED — one arm missing")
else:
    m, sd, se, t, n = r3
    ok3 = abs(t) > 2.0
    print(f"   d0.975_m6_g2 - d0.98_m6_g2 = {m:+.4f}  SE {se:.4f}  t {t:.2f}  n {n}"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   Still unresolved. The threshold optimum claimed in notes/40 and")
        print("   re-claimed in notes/41 is not established, and adding datasets is not")
        print("   the cheap way to establish it.")

print("\n4. the best cell is still " + ANCHOR)
ok4 = best == ANCHOR
print(f"   best = {best}  ->  {'PASS' if ok4 else 'FAIL'}")
if not ok4:
    r = paired(best, ANCHOR)
    if r:
        print(f"   and it beats the incumbent by {r[0]:+.4f}, t {r[3]:.2f} "
              f"({'resolved' if abs(r[3]) > 2 else 'still not resolved'})")
    print("   The n=12 ranking was sample noise. notes/40 and notes/41 both concluded")
    print("   from it and both need amending.")

print("\n5. the extra data BUYS something: at least one comparison unresolved at n=12")
print("   is resolved here")
ok5 = len(resolved) > 0
print(f"   {len(resolved)} of {len(rank)} arms resolve against the incumbent"
      f"  ->  {'PASS' if ok5 else 'FAIL'}")
if resolved:
    print("   " + ", ".join(resolved[:6]))
else:
    print("   Every arm is within noise of every other. The config surface is flat at")
    print("   this resolution and the gap to bronze is not in these knobs.")

print("\n" + "=" * 92)
print(f"n={NN}  resolves {mre(SD_TYP, NN):.4f}  |  best {best} {S[best][key]:.4f}"
      f"  |  incumbent {ANCHOR} {inc:.4f}")
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
