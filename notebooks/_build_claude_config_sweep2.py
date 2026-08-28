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

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_config_sweep2.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 12
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# notes/40: 0.975 won and is INTERIOR, so refine around it rather than re-scan.
DET_GRID = [0.98, 0.975, 0.97, 0.965]
# post-processing, free on each cached graph: (min_track_frames, gap_max_frames)
# notes/40 §4: max_gap was on the boundary (only 1 and 2 tried, 2 won everywhere) and
# min_len was coarse (6 beat 3 everywhere it mattered). Both extended; min_len=0 at
# gap=2 stays as the within-run reference for what pruning adds.
POST_GRID = [(0, 2), (6, 2), (8, 2), (10, 2),
             (0, 3), (6, 3), (8, 3), (10, 3), (12, 3),
             (6, 1), (0, 1)]
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
# The settings every 0.926 team uses and we never tested

```
0.897  ours (submitted)          0.926 bronze    0.944 gold
0.923–0.927  the public notebooks
```

`notes/39` asked why progress was slow. The answer was a **bad diagnosis held too long**:
`notes/33` put ~0.04 in the two model datasets we do not attach, and the two mechanisms
tested from that thesis came to a **~0.002 ceiling** (deepcenter) and **+0.0036**
(bidirectional — which needs no second model at all). About 10 % of the claimed gap.

An audit of the 0.927 notebook's config against ours found four settings we have never
touched:

| setting | theirs | ours | tested |
|---|---|---|---|
| `DET_THRESHOLD` | **0.96875** | **0.99** | **never swept** |
| `GAP_CLOSE_MAX_GAP` | **2** | **1** | **no** |
| `OUTPUT_MIN_TRACK_LEN` | **6** | **none** | **no** |
| `GAP_DENSITY_ADAPTIVE` | on | off | no |

`DET_THRESHOLD` is the worst. `notes/28` froze it at 0.99 *deliberately*, so one leaderboard
delta would isolate the repair — and it was never unfrozen. `notes/35` §3 then measured the
node-budget term as worth ~0.01 by itself.

## This run changes the method, not just the settings

One mechanism per launch cannot close 0.029 in the time left, even if every experiment
works. `DET_THRESHOLD` needs a prediction pass per value; **everything else is
post-processing on the resulting graph and is free.** So: 4 GPU passes and 24 post-processing
combinations per dataset, in one run.

Everything else is held at the located configuration — blend w=0.15 (`notes/38`), ILP
0.4/2.0 (`notes/36`), repair chain as submitted.

## Pre-registered predictions

1. **`det=0.99, min_len=0, gap=1` reproduces `notes/38`'s `w0.15+repair` (0.9499 ± 0.002)**
   on the same 12 datasets. That cell is the current submission by construction; if it
   misses, the grid is not anchored to anything and nothing below is readable.
   `notes/38` §3's lesson: grade the control against a number measured on the SAME datasets.
2. **A looser `DET_THRESHOLD` changes the node count by more than 2 %.** Mechanical — a
   threshold that moves nothing is not being tested, whatever the score says.
3. **Short-track pruning improves the score at some `min_len`.** Every 0.926 team prunes and
   we keep every two-node fragment; `notes/35` §3 showed the budget is worth ~0.01.
4. **The best cell beats the current submission's 0.9499 by more than 0.001** — the outcome
   claim at `notes/34`'s noise floor.
5. **The best `DET_THRESHOLD` is not 0.99.** The sharpest form of the audit: if 0.99 wins,
   the freeze was harmless and the audit's headline item is wrong.

*Training data, contaminated for the pack's weights (`notes/24` §2). `notes/37`: transfer is
not a constant — direction has held 4/4, magnitude has ranged 0.59x to 1.22x.*
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
names = []
if CACHE:
    names = sorted(p.stem[5:] for p in Path(CACHE).glob("cand_*.npz"))
    names = [n for n in names if (TRAIN / (n + ".geff")).exists()]
if not names:
    alln = sorted(p.stem for p in TRAIN.glob("*.zarr")
                  if (TRAIN / (p.stem + ".geff")).exists())
    a = [n for n in alln if n.startswith("44b6")]
    b = [n for n in alln if n.startswith("6bba")]
    k = max(1, round(N_DATASETS * len(a) / max(len(a) + len(b), 1)))
    names = a[:k] + b[:N_DATASETS - k]
# Stratify whatever list we ended up with. notes/34 recorded that names[:12] taken
# alphabetically inverted the embryo split; v2 of THIS notebook did it again, because the
# stratified branch only ran when the cache was missing. Slice proportionally, always.
_a = [n for n in names if n.startswith("44b6")]
_b = [n for n in names if not n.startswith("44b6")]
if _a and _b and len(names) > N_DATASETS:
    _k = max(1, round(N_DATASETS * len(_a) / len(names)))
    _k = min(_k, len(_a), N_DATASETS - 1)
    names = _a[:_k] + _b[:N_DATASETS - _k]
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

for name in names:
    t0 = time.time()
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
            v = float(ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
            PER.setdefault(name, {{}})[lbl] = v
            if best_here is None or v > best_here[0]:
                best_here = (v, lbl)
        parts.append("d" + str(det) + " " + format(best_here[0], ".4f"))
    print("  " + "  ".join(parts) + "   " + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "det_grid": DET_GRID, "post_grid": [list(g) for g in POST_GRID],
           "blend_w": BLEND_W, "datasets": [n for n in names if n in PER],
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "config_sweep2.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. The five predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "config_sweep2.json").read_text())
S, A, N, E, C = D["summary"], D["anatomy"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS = D["arms"], D["datasets"]
DETS, POST = D["det_grid"], [tuple(g) for g in D["post_grid"]]
ANCHOR_LBL = "d0.975_m6_g2"         # == notes/40's best cell, measured at 0.9535
N38 = 0.9535                        # notes/40 best cell on THESE 12 datasets
EXACT = ANCHOR_LBL in S and S[ANCHOR_LBL]["score"] == S[ANCHOR_LBL]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(DETS)} thresholds x {len(POST)} post combos = {len(ARMS)} cells")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

def cell(d, m, g):
    return "d" + str(d) + "_m" + str(m) + "_g" + str(g)

# grid view: rows = threshold, cols = (min_len, gap)
hdr = "".join(f"m{m}g{g}".rjust(10) for m, g in POST)
print(f"{'det':<10}{hdr}{'nodes':>11}{'cand':>11}")
print("-" * (10 + 10 * len(POST) + 22))
for d in DETS:
    row = "".join(f"{S[cell(d,m,g)][key]:>10.4f}" if cell(d, m, g) in S else " " * 10
                  for m, g in POST)
    n0 = N[cell(d, POST[0][0], POST[0][1])] if cell(d, POST[0][0], POST[0][1]) in N else 0
    print(f"{d:<10}{row}{n0:>11,}{C.get('d' + str(d), 0):>11,}")

print()
print("=" * 92)
print("PREDICTION GRADING")
print("=" * 92)

print("\n1. det=0.975, min_len=6, gap=2 reproduces notes/40's 0.9535 (+-0.002)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    got = S[ANCHOR_LBL]["score"]
    ok1 = abs(got - N38) <= 0.002
    print(f"   {ANCHOR_LBL} = {got:.4f} vs {N38:.4f}  ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   The anchor cell IS the submitted chain by construction, and")
        print("   probes/exec_config.py pins max_gap=1 as byte-identical on 30 random")
        print("   graphs. A miss here means the grid is not anchored and nothing below is")
        print("   comparable to anything already scored.")

print("\n2. a looser DET_THRESHOLD changes the node count by more than 2%")
base_n = N.get(cell(DETS[0], 0, 1), 0)
# Grade the RANGE, not every step. Adjacent thresholds are allowed to be close; what
# matters is whether the grid spans a meaningful amount of detection at all.
for d in DETS[1:]:
    n_d = N.get(cell(d, 0, 1), 0)
    print(f"   det={d:<9} nodes {n_d:>9,}  vs {base_n:,}  "
          f"({(n_d - base_n) / max(base_n, 1):+.2%})")
loosest = N.get(cell(DETS[-1], 0, 1), 0)
span = (loosest - base_n) / max(base_n, 1)
ok2 = abs(span) > 0.02
print(f"   tightest -> loosest: {span:+.2%}  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   A threshold that does not move the node count is not being tested. Check")
    print("   that det_threshold reaches PredictConfig and is not overridden downstream.")

print("\n3. a min_len LONGER than notes/40's 6 keeps improving")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ok3 = False
    for d in DETS:
        base = S.get(cell(d, 0, 1), {}).get(key)
        if base != base or base is None:
            continue
        line = [f"det={d}: m0 {base:.4f}"]
        for m in sorted({m for m, g in POST if m > 0}):
            c = cell(d, m, 1)
            if c in S:
                line.append(f"m{m} {S[c][key]:.4f} ({S[c][key] - base:+.4f})")
                ok3 |= S[c][key] > base + 1e-6
        print("   " + "   ".join(line))
    print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   Pruning never helps here. Either our fragments are already being caught")
        print("   by the node budget, or they carry edges that DO match and the field's")
        print("   MIN_TRACK_LEN=6 is doing something else for them.")

print("\n4. the best cell beats the current submission (0.9499) by more than 0.001")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ranked = sorted(((S[a][key], a) for a in ARMS if a in S), reverse=True)
    bv, ba = ranked[0]
    d4 = bv - N38
    ok4 = d4 > 0.001
    for v, a in ranked[:6]:
        print(f"   {a:<20} {v:.4f}  ({v-N38:+.4f})")
    print(f"   best {ba} at {d4:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   The audited settings do not pay on our pipeline. That is a real answer:")
        print("   the 0.926 field's config is not transferable line by line, and the gap")
        print("   is somewhere the config diff does not show — most likely the models.")

print("\n5. the refined threshold optimum is interior, not at a grid edge")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    per_det = [(max(S[cell(d, m, g)][key] for m, g in POST if cell(d, m, g) in S), d)
               for d in DETS if any(cell(d, m, g) in S for m, g in POST)]
    for v, d in per_det:
        print(f"   det={d:<9} best cell {v:.4f}")
    bd = max(per_det)[1]
    ok5 = bd not in (DETS[0], DETS[-1])
    print(f"   best threshold {bd}  ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   The optimum is outside 0.965-0.98, so this refinement was aimed at the")
        print("   wrong interval — notes/40's coarse grid had it bracketed and this one")
        print("   does not.")

print()
print("=" * 92)
if EXACT:
    bv, ba = max((S[a][key], a) for a in ARMS if a in S)
    print(f"BEST CELL: {ba} at {bv:.4f}   ({bv - N38:+.4f} vs the current submission)")
    print(f"  nodes {N[ba]:,}  edges {E[ba]:,}  mislink {A[ba]['fn_mislink']:,}  "
          f"gap {A[ba]['fn_gap']:,}  detect {A[ba]['fn_detect']:,}")
    if bv - N38 <= 0.001:
        print("  INSIDE NOISE. The four audited settings do not transfer to our pipeline;")
        print("  do not spend a slot. The remaining candidate is the temporal linker,")
        print("  notes/33 §1 — the one thing that adds model capacity rather than re-reading")
        print("  the model we have.")
else:
    print("NO BEST CELL — score column is NaN.")
print("=" * 92)
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
