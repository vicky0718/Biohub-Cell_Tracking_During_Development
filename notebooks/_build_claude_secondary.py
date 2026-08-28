"""Build notebooks/claude_secondary.ipynb — the settings the audit found untested.

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

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_secondary.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 12
BLEND_W = 0.15          # notes/38/39: the located operating point
# ours is 0.99, theirs 0.96875. Never swept. One prediction pass each.
# the located configuration (notes/40), held fixed: this run varies ONE thing.
DET_GRID = [0.975]
# (secondary edge weight, mode). 0.0 is the control and is bit-identical by identity.
SEC_GRID = [(0.0, 'fixed'), (0.15, 'low_margin'), (0.30, 'low_margin'),
            (0.15, 'fixed'), (0.30, 'fixed'), (0.50, 'fixed')]
# post-processing, free on each cached graph: (min_track_frames, gap_max_frames)
POST_GRID = [(6, 2)]          # notes/40's located post-processing, held fixed
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_DATASETS__", str(N_DATASETS))
              .replace("__BLEND_W__", repr(BLEND_W))
              .replace("__DET_GRID__", repr(DET_GRID))
              .replace("__SEC_GRID__", repr(SEC_GRID))
              .replace("__POST_GRID__", repr(POST_GRID)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# The second model — the last untested lever

```
0.899  submitted        rank ~1297/2792
0.926  bronze           0.944  gold
```

`notes/40` closed the config direction: ILP weights, repair chain, detection threshold, gap
span and short-track pruning are all located, several confirmed on two independent grids.
`notes/34` measured the deepcenter model at a ~0.002 ceiling. `notes/38` measured
bidirectional linking at +0.0036, and it needs no second model at all.

**This is the only remaining lever that adds model capacity** rather than re-reading the one
we have: `pilkwang/biohub-temporal-unet3d-seed314159-v1`, a second edge predictor of the
same architecture from a different seed. The public notebooks run it; we never have.

## How it is wired

The pack ships no ensemble hook, so `pipeline.secondary.patch_source` edits
`predict_video`'s source at two sites — the `model.encode` call and the `predict_edges`
call. **Both anchors were read from the pack's actual script, not guessed**, and
`probes/exec_secondary.py` applies the patch to that real source and compiles it. The
bidirectional build cost three launches to anchor mismatches and missing imports; this one
was checked on the ground truth first.

The secondary's logits are calibrated onto the primary's per-target mean and standard
deviation before mixing — the downstream candidate threshold and the ILP were both tuned
against the primary's scale. Two mixing modes:

* **`fixed`** — constant weight everywhere.
* **`low_margin`** — weight scales with the primary's own top-2 uncertainty and is **zeroed
  where the two models disagree** about the best parent. Where the primary is confident
  there is nothing to gain; where they disagree outright, averaging two contradictory
  answers is worse than either.

Everything else is held at the located configuration: det 0.975, gap 2, min-track 6, ILP
0.4/2.0, bidirectional w=0.15. **This run varies one thing.**

## Pre-registered predictions

1. **`w=0` reproduces `notes/40`'s 0.9535 (± 0.002)** on the same 12 datasets.
   `secondary_blend` returns the primary *by identity* at w=0, so a miss means the patch
   changed something it should not have, and nothing below is readable.
2. **The secondary actually fires** — candidate counts differ from the control by a real
   margin at some weight. A patch that matched but did nothing would present copies of the
   control as a sweep, which is exactly what `notes/38`'s prediction 2 was written to catch.
3. **`low_margin` beats `fixed` at matched weight.** The mode claim. If it does not, the
   gating is complexity for nothing and `fixed` is the honest choice.
4. **The best arm beats the control by more than 0.001** — the outcome claim at `notes/34`'s
   noise floor.
5. **The optimum is not at the largest weight tried.** Asked up front; three ILP sweeps each
   had to ask it again.

*`notes/40`: every mechanism this session has returned +0.002 to +0.004 on the leaderboard,
five for five in direction, none larger. Bronze needs +0.027. This is the last identified
lever, and it would have to be several times larger than anything measured so far to close
that — which is stated here so the result is read against a real expectation.*
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
SEC_GRID = [tuple(g) for g in __SEC_GRID__]
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


SEC_PATH = None
for _root in Path("/kaggle/input").glob("*"):
    for _c in list(_root.rglob("*.pth")) + list(_root.rglob("*.pt")):
        if "seed314159" in str(_c) or "temporal" in str(_c).lower():
            SEC_PATH = _c
            break
    if SEC_PATH:
        break
print("secondary weights:", SEC_PATH, flush=True)
if SEC_PATH is None:
    raise SystemExit("secondary model not mounted — attach "
                     "pilkwang/biohub-temporal-unet3d-seed314159-v1")
SEC_MODEL, sec_w, sec_d = P.load_model(SEC_PATH, DEV)
if (sec_w, sec_d) != (window_size, downsample):
    # Different inference grids means the two models index different feature maps, and the
    # blend would pair unrelated cells while still producing plausible numbers.
    raise SystemExit("primary/secondary inference grids differ: "
                     + str((window_size, downsample)) + " vs " + str((sec_w, sec_d)))
print("secondary loaded, grids match", flush=True)

from pipeline.secondary import patch_source as sec_patch

def make_sec_predict(w, mode):
    # w=0 uses the ORIGINAL bidirectional-only function, so the control cannot differ from
    # the shipped pipeline by even a rounding step.
    base = make_predict(BLEND_W)
    if w == 0.0:
        return base, None
    ns = dict(P.__dict__)
    src = sec_patch(ORIG_SRC, w, mode=mode)
    from pipeline.bidirectional import patch_source as bid_patch
    src = bid_patch(src, BLEND_W)
    exec(compile(src, "<sec>", "exec"), ns)
    ns["_SECONDARY_MODEL"] = SEC_MODEL
    return ns["predict_video"], ns

LABELS = ["w" + str(w) + "_" + m for w, m in SEC_GRID]
ROWS = dict((l, []) for l in LABELS); ANAT = dict((l, []) for l in LABELS)
NODES = dict((l, 0) for l in LABELS); EDGES = dict((l, 0) for l in LABELS)
CAND = dict((l, 0) for l in LABELS); PER = {{}}
DET = DET_GRID[0]
MIN_LEN, GAP_MAX = POST_GRID[0]
cfg_d = P.PredictConfig(det_threshold=DET, use_ilp=True,
                        ilp_edge_weight=ILP_EDGE_W, ilp_appearance_weight=ILP_APP_W,
                        ilp_disappearance_weight=ILP_DIS_W, ilp_division_weight=ILP_DIV_W)
print("det", DET, "| min_len", MIN_LEN, "| gap", GAP_MAX, "|", len(SEC_GRID), "arms", flush=True)

for name in names:
    t0 = time.time()
    sc = read_scale(TRAIN / (name + ".zarr"))
    gt = read_geff(TRAIN / (name + ".geff"))
    parts = [name]
    for w, mode in SEC_GRID:
        lbl = "w" + str(w) + "_" + mode
        pv_s, _ns = make_sec_predict(w, mode)
        coords, edges = pv_s(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg_d,
                             window_size=window_size, unet_batch_size=8,
                             downsample=downsample)
        g_td = P.build_graph(coords, edges)
        CAND[lbl] += int(g_td.num_edges())
        if g_td.num_edges():
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APP_W, disappearance_weight=ILP_DIS_W,
                division_weight=ILP_DIV_W)
            with P.suppress_output():
                g_td = solver.solve(g_td)
        tr = Tracks.from_tracksdata(g_td)
        g = repair_at((tr.t, tr.zyx, tr.edges), sc, MIN_LEN, GAP_MAX)
        ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
        NODES[lbl] += int(len(g[0])); EDGES[lbl] += int(len(g[2]))
        a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
        ANAT[lbl].append(a)
        if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
            raise SystemExit(name + "/" + lbl + ": buckets do not sum")
        PER.setdefault(name, {{}})[lbl] = float(
            ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
        parts.append(lbl + " " + format(PER[name][lbl], ".4f"))
    print("  " + "  ".join(parts) + "   " + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "sec_grid": [list(g) for g in SEC_GRID], "det": DET,
           "post": [MIN_LEN, GAP_MAX], "blend_w": BLEND_W,
           "datasets": [n for n in names if n in PER],
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "secondary.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "secondary.json").read_text())
S, A, N, E, C = D["summary"], D["anatomy"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, GRID = D["arms"], D["datasets"], [tuple(g) for g in D["sec_grid"]]
CTRL = "w0.0_fixed"
N40 = 0.9535                       # notes/40's located cell, on THESE 12 datasets
EXACT = CTRL in S and S[CTRL]["score"] == S[CTRL]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(ARMS)} arms  |  det {D['det']}  post {D['post']}  "
      f"blend {D['blend_w']}")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

print(f"{'arm':<18}{'score':>9}{'vs ctl':>9}{'edge_J':>9}{'div_J':>8}"
      f"{'mislink':>9}{'gap':>7}{'detect':>8}{'cand':>11}")
print("-" * 88)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    dj = st.get("division_jaccard")
    print(f"{a:<18}{st[key]:>9.4f}{st[key]-S[CTRL][key]:>+9.4f}{st['edge_jaccard']:>9.4f}"
          f"{(dj if dj == dj else 0):>8.4f}{an['fn_mislink']:>9,}{an['fn_gap']:>7,}"
          f"{an['fn_detect']:>8,}{C.get(a, 0):>11,}")

print()
print("=" * 88)
print("PREDICTION GRADING")
print("=" * 88)

print("\n1. w=0 reproduces notes/40's 0.9535 (+-0.002)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    got = S[CTRL]["score"]
    ok1 = abs(got - N40) <= 0.002
    print(f"   {CTRL} = {got:.4f} vs {N40:.4f}  ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   secondary_blend returns the primary BY IDENTITY at w=0 and the control")
        print("   uses the unpatched-for-secondary function, so a miss means the patch")
        print("   changed something it should not have. Nothing below is readable.")

print("\n2. the secondary actually fires")
c0 = C.get(CTRL, 0)
ok2 = False
for a in ARMS:
    if a == CTRL:
        continue
    d = (C.get(a, 0) - c0) / max(c0, 1)
    fired = abs(d) > 0.001
    ok2 |= fired
    print(f"   {a:<18} candidates {C.get(a,0):>9,}  vs {c0:,}  ({d:+.2%})"
          f"  {'ok' if fired else '<-- IDENTICAL'}")
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   Every arm is a copy of the control: the patch matched but did nothing.")
    print("   Check that _SECONDARY_MODEL reached the patched function's globals.")

print("\n3. low_margin beats fixed at matched weight")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    pairs = [(w, f"w{w}_low_margin", f"w{w}_fixed") for w, m in GRID
             if m == "low_margin" and f"w{w}_fixed" in S and f"w{w}_low_margin" in S]
    if not pairs:
        print("   NOT GRADED — no matched-weight pair in the grid.")
    else:
        ok3 = True
        for w, lm, fx in pairs:
            d = S[lm][key] - S[fx][key]
            ok3 &= d > 0
            print(f"   w={w:<5} low_margin {S[lm][key]:.4f}  vs fixed {S[fx][key]:.4f}"
                  f"   {d:+.4f}  {'low_margin' if d > 0 else 'FIXED'}")
        print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
        if not ok3:
            print("   The gating is complexity for nothing — report `fixed` as the honest")
            print("   choice rather than keeping a mode that does not earn itself.")

print("\n4. the best arm beats the control by more than 0.001")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    cands = [(S[a][key], a) for a in ARMS if a != CTRL and a in S]
    bv, ba = max(cands)
    d = bv - S[CTRL][key]
    ok4 = d > 0.001
    for v, a in sorted(cands, reverse=True):
        print(f"   {a:<18} {v:.4f}  ({v-S[CTRL][key]:+.4f})")
    print(f"   best {ba} at {d:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   The second model adds nothing on our pipeline. That closes the LAST")
        print("   identified lever (notes/40 §4), and the honest conclusion is that the")
        print("   0.926 field's advantage is not reproducible from what is public —")
        print("   0.899-0.903 is where this approach lands.")

print("\n5. the optimum is not at the largest weight tried")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ws = sorted({w for w, m in GRID if w > 0})
    vals = [(w, max(S[a][key] for a in ARMS
                    if a.startswith("w" + str(w) + "_") and a in S)) for w in ws]
    if len(vals) >= 3:
        bw = max(vals, key=lambda kv: kv[1])[0]
        ok5 = bw != ws[-1]
        print("   " + "  ".join(f"w{w}:{v:.4f}" for w, v in vals))
        print(f"   best w={bw}  ->  {'PASS — interior or at the floor' if ok5 else 'FAIL — still climbing'}")
    else:
        print(f"   NOT GRADED — {len(vals)} weights.")

print()
print("=" * 88)
if EXACT:
    best = max(ARMS, key=lambda a: S[a][key] if a in S else float("-inf"))
    d = S[best][key] - S[CTRL][key]
    print(f"BEST ARM: {best} at {S[best][key]:.4f}  ({d:+.4f} vs control)")
    print(f"  mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
          f"detect {A[best]['fn_detect']:,}  div_J {S[best].get('division_jaccard', 0):.4f}")
    if d <= 0.001:
        print("  INSIDE NOISE. The last identified lever does not pay. Every direction from")
        print("  notes/33's audit is now measured, and the remaining gap to 0.926 is not")
        print("  explained by anything in the public configuration or the public weights.")
    else:
        print(f"  Submittable: secondary at {best}.")
else:
    print("NO BEST ARM — score column is NaN.")
print("=" * 88)
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
