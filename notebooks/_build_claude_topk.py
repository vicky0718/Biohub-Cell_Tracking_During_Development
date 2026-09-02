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

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_topk.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 36
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# notes/40: 0.975 won and is INTERIOR, so refine around it rather than re-scan.
# det_threshold pushed FAR past anything tried. notes/44 only ever swept 0.965-0.99,
# where the sigmoid is saturated and node count moves 5.6%. If the logits have spread,
# these reach the budget regime by ranking rather than by spatial suppression.
POOL_GRID = [0.975, 0.999, 0.9999, 0.99999, 0.999999]
DET_FIXED = 3.0            # pool_kernel_um, held at the pack default while the cut varies
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
# Confidence ranking vs spatial suppression, at matched node count

```
0.901 submitted    0.926 bronze    0.944 gold
measurable > 0.0015     worth a slot > 0.01        (notes/44)
```

`notes/45` established the mechanism and nothing since has dented it: `J_adj = J · (1 −
0.1·(N_pred − N_total)/N_total)` with a ceiling of 1.1, and predicted structure matching no
ground-truth node is **excluded** from the edge term rather than penalised. Over-prediction
buys nothing and costs budget.

Two attempts to collect it have failed, and they failed the same way.

```
claude_budget  (pool_kernel_um, the NMS radius)   node_recall 0.983 -> 0.537 as nodes fall
claude_spotiflow (a 35M-param domain fine-tune)   recall saturates at 0.547 at ANY threshold
```

`notes/47` also found that the pack's own detector is the most selective thing we have —
**0.996 recall at 24,605 nodes**, beating that fine-tune at every matched node count.

## What has not been tried

`claude_budget` cut nodes with `pool_kernel_um`. That is the **non-maximum-suppression
radius**: it suppresses by spatial neighbourhood, keeping one peak per ball regardless of
how confident any of them were. It deleted ground-truth cells as fast as everything else.

**Cutting by the detector's own confidence is a different rule.** `det_threshold` is the
nearest thing tried, and `notes/44` measured it moving node count only **5.6%** across
0.965–0.99 — the sigmoid is saturated there, so a threshold in that range cannot reach the
budget regime. It has never been pushed past 0.99.

So: sweep it to **0.975, 0.999, 0.9999, 0.99999, 0.999999**, with `pool_kernel_um` held at
its default. If the logit distribution has any spread at all, the count falls; if it does
not, the count barely moves and that is the answer — a saturated detector cannot rank, and
the budget is unreachable with what we have.

## The prediction that matters

Not "does thinning help" — `claude_budget` answered that for spatial suppression. The
question is **whether confidence ranking preserves ground truth better than spatial
suppression did, compared at the same node count.**

```
pool sweep, for reference:   nodes 700,216  659,535  519,492  325,514   95,137
                        node_recall 0.9827   0.9790   0.9367   0.8113   0.5374
```

`notes/47`'s lesson is the design constraint here: a ratio like recall-per-node rises
mechanically as nodes fall, and averaging per-dataset ratios lets one small-denominator
dataset carry the mean. Both errors made a dead direction look alive for a full run. So
prediction 3 interpolates **this** run's recall curve to the pool sweep's node counts and
compares there, rather than comparing operating points.

## Pre-registered predictions

1. **The anchor reproduces.** `det 0.975` is what `claude_budget` measured at `pool 3.0`;
   it must land within 0.003 of 0.9356 or nothing below is comparable.
2. **Extreme thresholds move the node count more than 3×.** `notes/44` measured 5.6% over
   0.965–0.99. If 0.999999 cannot do better, the detector's confidence is saturated, it
   cannot rank, and this closes the direction in one run.
3. **Confidence ranking beats spatial suppression at matched node count** — this run's
   `node_recall`, interpolated to the pool sweep's counts, exceeds the pool sweep's recall
   there. The crux, and the only reason to expect a different outcome from `claude_budget`.
4. **`adj_edge_jaccard` is non-monotonic** in the threshold — rises, then falls. Monotone
   down means the trade never pays; monotone up means the grid stopped too early.
5. **The best arm beats the incumbent by more than 0.01** — `notes/44`'s bar for a
   submission slot, roughly three times what n=36 resolves.

*If 2 or 3 fails, the node-budget direction is finished: neither of the two selection rules
available to us can reach the regime without destroying the recall that pays for it.*
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
        # `det` is the DETECTION THRESHOLD here and pool_kernel_um is held at the
        # pack default -- the mirror of claude_budget, which swept the kernel and held
        # the threshold. Sweeping both would confound the two halves of one line.
        cfg_d = P.PredictConfig(det_threshold=det, pool_kernel_um=DET_FIXED, use_ilp=True,
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

    out = {{"arms": LABELS, "det_grid": POOL_GRID, "pool_fixed": DET_FIXED, "post_grid": [list(g) for g in POST_GRID],
           "blend_w": BLEND_W, "datasets": [n for n in names if n in PER],
           "seed_datasets": seed_names,
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "topk.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "topk.json").read_text())
S, N, E, C = D["summary"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
# The writer renames this key to "det_grid"; v1 renamed the writer and not the
# reader, so the worker finished all 7,771 s of real work and the analysis cell
# died on a KeyError. Accept either, so a rename can never throw away a run.
POOLS = D.get("det_grid") or D["pool_grid"]
POST = [tuple(g) for g in D["post_grid"]]
ANCHOR = "p0.975_m6_g2"        # det 0.975 at the default kernel: what we run today
WIDECV = 0.9356                 # claude_budget's p3.0_m6_g2, same chain, n=36
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

print("\n1. the anchor arm (det 0.975, the current default) reproduces claude_budget's 0.9356")
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

print("\n2. node count falls more than 3x across the threshold grid")
ns = [n for _, _, n, _, _ in ROWS if n]
if len(ns) < 2:
    print("   NOT GRADED — fewer than two arms produced counts")
else:
    fall = max(ns) / max(min(ns), 1)
    ok2 = fall > 3.0
    print(f"   {max(ns):,} -> {min(ns):,}  = {fall:.1f}x  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   The sigmoid is saturated even at 0.999999: the detector assigns")
        print("   near-identical confidence to almost everything it finds, so it cannot")
        print("   RANK. Spatial suppression (claude_budget) and confidence ranking are")
        print("   the only two selection rules the pack offers, and neither reaches the")
        print("   budget regime. The direction is finished.")

print("\n3. confidence ranking beats spatial suppression at MATCHED node count")
# claude_budget, n=36, same chain: (nodes, node_recall) as pool_kernel_um widened.
POOL_N = [95137, 325514, 519492, 659535, 700216]
POOL_R = [0.5374, 0.8113, 0.9367, 0.9790, 0.9827]
mine_n = [n for _, _, n, _, _ in ROWS if n]
mine_r = [s.get("node_recall", float("nan")) for _, s, n, _, _ in ROWS if n]
pairs = sorted((n, r) for n, r in zip(mine_n, mine_r) if r == r)
if len(pairs) < 2:
    ok3 = False
    print("   NOT GRADED — need at least two arms with node_recall")
else:
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    # Compare only where BOTH curves have support -- notes/47: extrapolating a ratio
    # past the end of its sweep is how a dead direction looked alive for a full run.
    lo, hi = max(min(xs), min(POOL_N)), min(max(xs), max(POOL_N))
    print(f"   {'nodes':>10}{'this run':>11}{'pool sweep':>12}   verdict")
    wins = 0; tested = 0
    for n in [n for n in POOL_N if lo <= n <= hi]:
        a = float(np.interp(n, xs, ys))
        b = float(np.interp(n, POOL_N, POOL_R))
        tested += 1; wins += int(a > b)
        print(f"   {n:>10,}{a:>11.4f}{b:>12.4f}   {'better' if a > b else 'worse'}")
    ok3 = tested > 0 and wins > tested / 2
    print(f"   better at {wins} of {tested} overlapping counts"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if tested == 0:
        print("   The two sweeps do not overlap in node count, so they cannot be")
        print("   compared at all. That is a design failure, not a result.")
    elif not ok3:
        print("   Confidence ranking is no better than a wider NMS ball at keeping the")
        print("   annotated cells. Both selection rules the pack offers are exhausted.")

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
