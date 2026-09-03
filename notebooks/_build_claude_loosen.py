"""Build notebooks/claude_loosen.ipynb — spend the node budget instead of hoarding it.

notes/52 measured ratio = -0.129: we already predict 12.9% FEWER nodes than
estimated_number_of_nodes, so the multiplier already pays 1.013 and there is no
over-prediction left to remove. notes/51 measured fn_detect = 583 (4.21% of GT edges),
now the largest failure bucket. Together: the chain deleted so many nodes chasing a
bonus it already had that it misses 4.21% of edges through endpoints never detected.

Break-even, from notes/52: giving up the multiplier entirely costs 0.0118 on adj_edge,
and each recovered fn_detect edge is worth ~1/14,343, so ~169 of the 583 (29%) must
come back. This sweep answers whether they do.

Derived from the topk builder, which swept this same axis UPWARD. The lowest
det_threshold ever tried in this project is 0.96875 (notes/40); notes/44 called the
surface flat over [0.965, 0.99] and closed it, and notes/49 found a cliff above it.
Below 0.965 has never been touched.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_loosen.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 36
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# notes/40: 0.975 won and is INTERIOR, so refine around it rather than re-scan.
# det_threshold swept DOWNWARD, below the 0.96875 floor of every previous grid.
# notes/52: we are at ratio -0.129, already under budget, so cutting further only deletes
# real edges. The open question is the other direction -- whether spending the multiplier
# buys back enough of notes/51's 583 undetected endpoints to clear the 0.0118 it costs.
# 0.975 stays as the anchor so the run is comparable to notes/48's 0.9410 on these same
# datasets; the rest is unexplored territory.
POOL_GRID = [0.975, 0.95, 0.90, 0.80, 0.60]
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
# Spend the budget: `det_threshold` below 0.965, where nobody has looked

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
adj = max(0, edge_J * (1 - 0.1 * ratio)),   ratio = (N_pred - N_est) / N_est
```

`notes/52` measured the one column nobody had read: **we sit at `ratio = -0.129`**, already
12.9% *under* the node budget, with the multiplier paying 1.013. Every attempt to collect
that bonus assumed there was over-prediction to remove. There was none, and all three
selection rules failed for that single reason:

```
notes/46     pool_kernel_um, an NMS radius            node_recall 0.983 -> 0.537
notes/48/49  det_threshold pushed UP                  0.901 -> 0.863 on the leaderboard
notes/52     track ranking under a per-dataset cap    monotonically worse
```

Put that beside `notes/51`, and the chain is mispriced in a specific, measurable way:

```
ratio        -0.129     the multiplier bonus we hold          worth ~1.3%
fn_detect       583     4.21% of GT edges, endpoint never matched -- the LARGEST bucket
fn_mislink      226     1.63%
```

**We deleted so many nodes chasing a bonus that we now miss 4.21% of ground-truth edges
through endpoints that were never detected.** That is an over-correction, and it is worth
undoing if — and only if — loosening detection brings enough of those endpoints back.

## The break-even, computed before the run

At the shipped chain `edge_J = 12,962 / 14,343`:

```
give up the multiplier entirely (ratio -0.129 -> 0)      -0.0118 on adj_edge
each recovered fn_detect edge is worth                   ~1 / 14,343
break-even                                               ~169 of 583 edges  (29%)
```

So the question is sharp: **does a ~13% rise in node count return more than 29% of the
undetected-endpoint edges?**

## Why this region is unexplored

Every threshold grid this project has run:

```
notes/40   0.99   0.985   0.975   0.96875
notes/41   0.98   0.975   0.97    0.965
notes/44   0.98   0.975   0.97
notes/48   0.975  0.999   0.9999   0.99999   0.999999
```

**The lowest value ever tried is 0.96875.** `notes/44` called the surface flat across
[0.965, 0.99] — node count moving only 5.6% — and closed the axis. `notes/49` then found a
cliff *above* it. Both statements are about a box whose lower wall was never touched, and
`notes/49`'s lesson was precisely that a sweep growing inward from a frozen value never
learns what is outside it.

Every prior move on this axis **hoarded** budget. This one spends it. That is the difference,
and it is the first time the direction of travel has been argued from a measurement of where
we actually sit rather than from the assumption that less is better.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`) — the check that 0.901 → 0.863 lacked.

1. **The anchor reproduces.** `p0.975_m6_g2` must land within 0.003 of `notes/48`'s
   **0.9410** on these same 36 datasets, or nothing below is comparable to the record.
2. **Node count rises more than 20%** from 0.975 to the lowest threshold. `notes/44`
   measured 5.6% over [0.965, 0.99]; if the sigmoid is just as saturated *below* 0.965 then
   the budget cannot be spent either, and the axis is closed in both directions by one run.
3. **`fn_detect` falls by more than 15%.** The mechanism, stated directly. Extra nodes that
   do not recover missed endpoints are pure cost, and if the count does not move then the
   583 are not threshold-recoverable at all — they are detector-capacity misses, which is
   `notes/51`'s reading and would point at `claude_zhpilot` instead.
4. **Some arm beats the anchor by more than 0.0015** (`notes/44`'s floor). The crux. This is
   where `notes/52`'s break-even arithmetic is either confirmed or refuted.
5. **The best arm holds in sign on BOTH embryos.** The test set is a third pair
   (`notes/07` §3); a pooled win across crops of two is not evidence about a third.

*The `ratio`, `mult`, `nodes` and `fn_detect` columns are printed for every arm regardless,
so a clean failure still yields the shape of the trade — which no run has ever plotted
across the region where the budget is spent rather than saved.*
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
    (WORK / "loosen.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "loosen.json").read_text())
S, N, E, C = D["summary"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
# The writer renames this key to "det_grid"; v1 renamed the writer and not the
# reader, so the worker finished all 7,771 s of real work and the analysis cell
# died on a KeyError. Accept either, so a rename can never throw away a run.
POOLS = D.get("det_grid") or D["pool_grid"]
POST = [tuple(g) for g in D["post_grid"]]
ANCHOR = "p0.975_m6_g2"        # det 0.975, the shipped chain: what we run today
WIDECV = 0.9410                 # notes/48's det 0.975 mean best-cell score, same n=36
NN = len(PER)
EXACT = ANCHOR in S and S[ANCHOR]["score"] == S[ANCHOR]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{NN} datasets, {len(POOLS)} thresholds x {len(POST)} post = {len(ARMS)} cells")

def cell(p, m, g):
    return "p" + str(p) + "_m" + str(m) + "_g" + str(g)

m0, g0 = POST[0]
A = D.get("anatomy", {})
print(f"\n{'det':<10}{'score':>10}{'adj_edge':>10}{'edge_J':>9}{'node_rec':>10}"
      f"{'ratio':>9}{'mult':>8}{'nodes':>12}{'fn_detect':>11}{'fn_mislink':>12}")
print("-" * 101)
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
    _a = A.get(c, {})
    print(f"{p:<10}{s.get(key, float('nan')):>10.4f}"
          f"{s.get('adj_edge_jaccard', float('nan')):>10.4f}"
          f"{s.get('edge_jaccard', float('nan')):>9.4f}"
          f"{s.get('node_recall', float('nan')):>10.4f}"
          f"{ratio:>9.3f}{mult:>8.4f}{n:>12,}"
          f"{_a.get('fn_detect', float('nan')):>11,.0f}"
          f"{_a.get('fn_mislink', float('nan')):>12,.0f}")

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


def by_embryo(a, b):
    """Per-embryo mean differences. THIS is the sample size, not len(PER).

    notes/49: `claude_submit_topk` scored 0.863 against a predicted 0.903-0.911 because
    n=36 was read as 36 independent draws. It is 36 crops of TWO embryos, and the host
    confirmed the test set shares no embryo_id with train (notes/07 §3). A pooled p-value
    over crops answers "does this hold on these two embryos", which is not the question.
    Harness defaults to fold_by="embryo" for this reason; the pooled `paired` above
    bypasses it, so both are printed and the embryo-level pair is the one that decides.
    """
    out = {}
    for nm, r in PER.items():
        x, y = r.get(a), r.get(b)
        if x is not None and y is not None and x == x and y == y:
            out.setdefault(nm.split("_")[0], []).append(x - y)
    return {e: sum(v) / len(v) for e, v in sorted(out.items()) if v}

best = max((a for a in ARMS if a in S), key=lambda a: S[a][key])
inc = S.get(ANCHOR, {}).get(key, float("nan"))
print(f"\nbest {best} = {S[best][key]:.4f}   incumbent {ANCHOR} = {inc:.4f}")
print(f"\n{'arm':<16}{'mean d':>10}{'sd':>9}{'SE':>9}{'t':>8}   verdict"
      f"{'per-embryo means':>34}   holds?")
for a in sorted((x for x in ARMS if x in S and x != ANCHOR), key=lambda x: -S[x][key])[:10]:
    r = paired(a, ANCHOR)
    if r is None:
        continue
    m, sd, se, t, n = r
    em = by_embryo(a, ANCHOR)
    # Agreement in SIGN across embryos is the minimum bar for expecting anything on a
    # third embryo. A pooled t of 2+ with embryos disagreeing is notes/49's failure exactly.
    agree = len(em) > 1 and (all(v > 0 for v in em.values()) or all(v < 0 for v in em.values()))
    print(f"{a:<16}{m:>+10.4f}{sd:>9.4f}{se:>9.4f}{t:>8.2f}   "
          f"{'RESOLVED' if abs(t) > 2.0 else 'not resolved':<14}"
          f"{'  '.join(f'{e} {v:+.4f}' for e, v in em.items()):>32}   "
          f"{'yes' if agree else 'NO -- pooled t is pseudoreplicated'}")

print("\n" + "=" * 92)
print("PREDICTION GRADING")
print("=" * 92)

print("\n1. the anchor arm (det 0.975, the shipped chain) reproduces notes/48's 0.9410")
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

print("\n2. node count RISES more than 20% from the anchor to the loosest threshold")
# ROWS is in POOL_GRID order, so ROWS[0] is the 0.975 anchor and the rest descend.
ns = [(p, n) for p, _, n, _, _ in ROWS if n]
if len(ns) < 2:
    ok2 = False
    print("   NOT GRADED — fewer than two arms produced counts")
else:
    base_n = dict(ns).get(POOLS[0], 0)
    top_n = max(n for _, n in ns)
    rise = top_n / max(base_n, 1)
    ok2 = rise > 1.20
    print(f"   anchor {base_n:,} -> max {top_n:,}   x{rise:.3f}"
          f"  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   notes/44 measured 5.6% over [0.965, 0.99] and called the axis flat. The")
        print("   sigmoid is just as saturated BELOW 0.965, so the budget cannot be spent")
        print("   any more than it could be collected. That closes this axis in both")
        print("   directions in one run, and the 583 undetected endpoints are not")
        print("   threshold-reachable at all.")

print("\n3. fn_detect falls more than 15% — the extra nodes recover missed endpoints")
det = [(p, A.get(cell(p, m0, g0), {}).get("fn_detect", float("nan"))) for p in POOLS]
det = [(p, d) for p, d in det if d == d]
if len(det) < 2:
    ok3 = False
    print("   NOT GRADED — anatomy unavailable")
else:
    base_d = dict(det).get(POOLS[0], float("nan"))
    lo_p, lo_d = min(det, key=lambda x: x[1])
    ok3 = base_d == base_d and base_d > 0 and (base_d - lo_d) / base_d > 0.15
    print(f"   {'det':>10}{'fn_detect':>11}{'vs anchor':>11}")
    for p, d in det:
        print(f"   {p:>10}{d:>11,.0f}{d - base_d:>+11,.0f}")
    print(f"   best {lo_p} at {lo_d:,.0f}, anchor {base_d:,.0f}, "
          f"{(base_d - lo_d) / max(base_d, 1):.1%} lower"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   Loosening does not bring the missed endpoints back. They are detector")
        print("   CAPACITY misses, not threshold misses -- notes/51's reading -- and the")
        print("   direction that addresses them is a retrained detector, not a knob.")

print("\n4. some arm beats the anchor by more than 0.0015 (notes/44's floor)")
if not EXACT:
    ok4 = False
    print("   NOT GRADED — score is NaN")
else:
    gain = S[best][key] - inc
    ok4 = best != ANCHOR and gain > 0.0015
    print(f"   best {best} = {S[best][key]:.4f} vs {ANCHOR} {inc:.4f}"
          f"   {gain:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    r4 = paired(best, ANCHOR) if best != ANCHOR else None
    if r4:
        m, sd, se, t, n = r4
        print(f"   paired over {n} datasets: mean {m:+.4f}  sd {sd:.4f}  t {t:.2f}"
              f"   {'RESOLVED' if abs(t) > 2.0 else 'not resolved'}")
    _bs, _as_ = S[best], S[ANCHOR]
    print(f"   decomposed:  edge_J {_bs.get('edge_jaccard', float('nan')) - _as_.get('edge_jaccard', float('nan')):+.4f}"
          f"   multiplier {(_bs.get('adj_edge_jaccard', 0) / max(_bs.get('edge_jaccard', 1), 1e-9)) - (_as_.get('adj_edge_jaccard', 0) / max(_as_.get('edge_jaccard', 1), 1e-9)):+.4f}"
          f"   nodes {N.get(best, 0) - N.get(ANCHOR, 0):+,}")
    if not ok4:
        print("   notes/52's break-even is refuted: a 13% rise in node count does not")
        print("   return the ~29% of fn_detect edges needed to clear the 0.0118 the")
        print("   multiplier costs. The threshold axis is then closed in BOTH directions")
        print("   and every remaining lever is on the detector itself.")

print(f"\n5. the best arm holds in sign on BOTH embryos (notes/49 — n is 2, not {NN})")
if best == ANCHOR:
    ok5 = False
    print("   NOT GRADED — the anchor is already best, nothing to transfer")
else:
    em = by_embryo(best, ANCHOR)
    print(f"   {'embryo':<8}{'mean delta':>13}")
    for e, v in em.items():
        print(f"   {e:<8}{v:>+13.4f}")
    ok5 = len(em) > 1 and (all(v > 0 for v in em.values())
                           or all(v < 0 for v in em.values()))
    if len(em) > 1 and all(abs(v) < 1e-9 for v in em.values()):
        ok5 = False
        print("   NOT GRADED — the arm is identical to the anchor on both embryos")
    else:
        print(f"   signs agree  ->  {'PASS' if ok5 else 'FAIL'}")
        if not ok5:
            print("   Wins on one embryo and loses on the other. The test set is a THIRD")
            print("   pair (notes/07 §3), so a pooled win across crops of two says nothing")
            print("   about it. This is the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 92)
_oks = [v for v in (globals().get("ok1"), ok2, ok3, ok4, ok5) if v is not None]
print(f"{sum(bool(v) for v in _oks)}/{len(_oks)} predictions passed")
if ok4 and ok5:
    print(f"SUBMITTABLE: {best} gains {S[best][key] - inc:+.4f} and holds on both embryos.")
elif ok2 and ok3 and not ok4:
    print("PRICED, NOT SUBMITTABLE: loosening does recover endpoints, but not enough to")
    print("pay for the multiplier it spends. The trade is now measured in both directions.")
elif not ok2:
    print("CLOSED BOTH WAYS: the threshold cannot move node count downward either.")
    print("Detection capacity is the only remaining lever (notes/51).")
elif ok4 and not ok5:
    print("MEASURED, NOT SUBMITTABLE: real on train, does not agree across embryos.")
else:
    print("SEE ABOVE — read predictions 2 and 3 before 4; they say whether the")
    print("mechanism moved at all.")
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
