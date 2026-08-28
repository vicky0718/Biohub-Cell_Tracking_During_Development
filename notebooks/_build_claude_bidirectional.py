"""Build notebooks/claude_bidirectional.ipynb — ask every link to hold up read backwards.

`notes/37`: the ILP-weight direction is closed and the remaining gap is **entirely the edge
term** — raw `edge_J` 0.9047 against the public notebooks' 0.923-0.927 total, with `div_J`
0.1154 now competitive with the field.

`notes/33` §2 found the lever the 0.927 notebook uses on that term which costs **no new
weights**: run the pack's linker forward *and* reverse in time and combine the two with a
weighted harmonic mean in probability space. Its own config comments date the feature to
their 0.915 reference run, so it is not a late micro-tweak.

Why a harmonic mean and not an average: if either direction rates a pair near-zero, its
reciprocal blows up and the pair collapses. An average lets a confident forward vote carry
a pair the reverse pass rejects -- precisely the `fn_mislink` failure `notes/26` named, and
a wrong parent is much less likely to survive the question asked backwards.

The pack ships no hook, so `pipeline.bidirectional.patch_source` edits `predict_video`'s
source text and re-execs it, asserting the anchor matches exactly once.
`probes/exec_bidirectional.py` proved the numerics on arrays; this measures them.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_bidirectional.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 12
WEIGHTS_SWEPT = [0.0, 0.15, 0.25, 0.40]
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_DATASETS__", str(N_DATASETS))
              .replace("__WEIGHTS__", repr(WEIGHTS_SWEPT)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Does a link survive being read backwards?

```
0.897  pack + ILP(0.4/2.0) + repair        <- best scored, rank ~1336/2792
0.923–0.927  the public notebooks
0.926  bronze    0.944  gold
```

`notes/37` closed the ILP-weight direction and located the gap precisely:

```
ours:    raw edge_J 0.9047     div_J 0.1154   (competitive — the field reports 0.12)
public:  0.923–0.927 total
```

**Divisions are no longer where we lose. Edges are.** This is the first lever aimed there,
and the cheapest available: no second model, no new weights.

## The mechanism

Run the pack's linker forward *and* reverse in time, then combine in probability space:

```
combined = 1 / ((1 - w) / P_forward  +  w / P_reverse)
```

A **harmonic** mean, not an average. If either direction rates a pair near-zero its
reciprocal blows up and the pair collapses, so a link survives only with *mutual* support.
An average would let a confident forward vote carry a pair the reverse pass rejects, which
is exactly the `fn_mislink` failure. `probes/exec_bidirectional.py` includes a case an
arithmetic mean would fail, so the test cannot pass for a blend that does not do the job.

The blend is re-centred and re-scaled onto the forward logits' own statistics, because
`DET_THRESHOLD` and the ILP's `edge_prob` were both tuned there. The mean is preserved
exactly; the std except where the [0.5, 2.0] clamp binds — 1.2 % of targets at w=0.15,
5.0 % at w=0.5.

## Pre-registered predictions

1. **`w=0` reproduces `notes/35`'s `ratio0.4_2.0+repair` (0.9179)** on the same datasets.
   Load-bearing twice: it proves a fresh prediction pass matches the cached candidates the
   last three sweeps were scored from, AND that the patch left the control untouched.
   `harmonic_blend` returns the forward logits *by identity* at `w=0`, and the control arm
   uses the original unpatched function object, so any difference here is real.
2. **The reverse pass actually changes the candidate set.** Candidate counts must differ
   from the control by a real margin. A patch that matched but did nothing would otherwise
   present four copies of the control as a sweep.
3. **The blend reduces `fn_mislink`** — the mechanism claim, stated separately from the
   outcome because they can come apart, as they did for the deepcenter veto (`notes/34`).
4. **Some weight beats the control by more than 0.001** — the outcome claim, at the noise
   floor `notes/34` forced after a +0.0000 arm was called submittable.
5. **The optimum is not at the largest weight tried.** Three ILP sweeps had to ask this
   three times; asking it up front is cheaper than a fourth grid.

*Training data, contaminated for the pack's weights (`notes/24` §2). `notes/37`: transfer is
not a constant — direction has held four times out of four, magnitude has ranged 0.59x to
1.22x. Read a train gain as roughly face value.*
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
WEIGHTS = __WEIGHTS__
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
from pipeline.repair import close_gaps, linefit_smooth
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
LABELS = ["w" + str(w) for w in WEIGHTS] + ["w" + str(w) + "+repair" for w in WEIGHTS]
ROWS = dict((l, []) for l in LABELS); ANAT = dict((l, []) for l in LABELS)
EDGES = dict((l, 0) for l in LABELS); NODES = dict((l, 0) for l in LABELS)
CAND = dict(("w" + str(w), 0) for w in WEIGHTS); PER = {{}}

for name in names:
    t0 = time.time()
    sc = read_scale(TRAIN / (name + ".zarr"))
    gt = read_geff(TRAIN / (name + ".geff"))
    parts = [name]
    for w in WEIGHTS:
        pv = make_predict(w)
        coords, edges = pv(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg,
                           window_size=window_size, unet_batch_size=8,
                           downsample=downsample)
        g_td = P.build_graph(coords, edges)
        CAND["w" + str(w)] += int(g_td.num_edges())
        if g_td.num_edges():
            solver = td.solvers.ILPSolver(
                edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
                appearance_weight=ILP_APP_W, disappearance_weight=ILP_DIS_W,
                division_weight=ILP_DIV_W)
            with P.suppress_output():
                g_td = solver.solve(g_td)
        tr = Tracks.from_tracksdata(g_td)
        base = (tr.t, tr.zyx, tr.edges)
        for rep in (False, True):
            lbl = "w" + str(w) + ("+repair" if rep else "")
            g = repair_chain(base, sc) if rep else base
            ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            NODES[lbl] += int(len(g[0])); EDGES[lbl] += int(len(g[2]))
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[lbl].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(name + "/" + lbl + ": buckets do not sum")
            PER.setdefault(name, {{}})[lbl] = float(
                ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
        parts.append("w" + str(w) + " " + format(PER[name]["w" + str(w) + "+repair"], ".4f"))
    print("  " + "  ".join(parts) + "   " + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "weights": WEIGHTS,
           "datasets": [n for n in names if n in PER],
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "nodes": NODES, "edges": EDGES, "candidates": CAND, "per_dataset": PER}}
    (WORK / "bidirectional.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "bidirectional.json").read_text())
S, A, N, E, C = D["summary"], D["anatomy"], D["nodes"], D["edges"], D["candidates"]
ARMS, DS, WS = D["arms"], D["datasets"], D["weights"]
CTRL = "w0.0+repair"
EXACT = S[CTRL]["score"] == S[CTRL]["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(ARMS)} arms, weights {WS}")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

print(f"{'arm':<16}{'score':>9}{'vs ctl':>9}{'edge_J':>9}{'div_J':>8}"
      f"{'mislink':>9}{'gap':>7}{'detect':>8}{'cand':>10}{'edges':>10}")
print("-" * 95)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    dj = st.get("division_jaccard")
    base = a.split("+")[0]
    print(f"{a:<16}{st[key]:>9.4f}{st[key]-S[CTRL][key]:>+9.4f}{st['edge_jaccard']:>9.4f}"
          f"{(dj if dj == dj else 0):>8.4f}{an['fn_mislink']:>9,}{an['fn_gap']:>7,}"
          f"{an['fn_detect']:>8,}{C.get(base, 0):>10,}{E[a]:>10,}")

print()
print("=" * 95)
print("PREDICTION GRADING")
print("=" * 95)

# 1 ---------------------------------------------------------------------------------
print("\n1. w=0 reproduces notes/35's ratio0.4_2.0+repair (0.9179 +- 0.002)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    got = S[CTRL]["score"]
    ok1 = abs(got - 0.9179) <= 0.002
    print(f"   w0.0+repair = {got:.4f}  ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   A fresh prediction pass does NOT match the cached candidates the last three")
        print("   sweeps were scored from. Either the cache was stale or the patch touched the")
        print("   control — and w=0 uses the ORIGINAL function object, so suspect the cache.")
        print("   Nothing below is comparable to notes/35 until this is understood.")

# 2 ---------------------------------------------------------------------------------
print("\n2. the reverse pass actually changes the candidate set")
c0 = C.get("w0.0", 0)
ok2 = True
for w in WS:
    if w == 0.0:
        continue
    c = C.get(f"w{w}", 0)
    d = (c - c0) / max(c0, 1)
    moved = abs(d) > 0.001
    ok2 &= moved
    print(f"   w={w:<5} candidates {c:>9,}  vs control {c0:,}  ({d:+.2%})"
          f"  {'ok' if moved else '<-- IDENTICAL, the patch did nothing'}")
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   A patch that matched but changed nothing presents copies of the control as a")
    print("   sweep. Check that _BIDIRECTIONAL_WEIGHT reaches the patched function's globals.")

# 3 ---------------------------------------------------------------------------------
print("\n3. the blend reduces fn_mislink (the MECHANISM claim)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    m0 = A[CTRL]["fn_mislink"]
    best_m, best_l = min((A[f"w{w}+repair"]["fn_mislink"], f"w{w}+repair")
                         for w in WS if f"w{w}+repair" in A)
    ok3 = best_m < m0
    for w in WS:
        l = f"w{w}+repair"
        if l in A:
            print(f"   {l:<16} mislink {A[l]['fn_mislink']:>6,}  ({A[l]['fn_mislink']-m0:+d})"
                  f"   detect {A[l]['fn_detect']:>6,}  ({A[l]['fn_detect']-A[CTRL]['fn_detect']:+d})")
    print(f"   best {best_l}  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   Mutual support does not remove wrong-parent links here. Either the reverse")
        print("   pass agrees with the forward one on the mistakes (the model is confidently")
        print("   wrong both ways) or the blend is too weak to reorder anything.")

# 4 ---------------------------------------------------------------------------------
print("\n4. some weight beats the control by more than 0.001")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    cands = [(S[f"w{w}+repair"][key], w) for w in WS
             if w != 0.0 and f"w{w}+repair" in S]
    bv, bw = max(cands)
    d = bv - S[CTRL][key]
    ok4 = d > 0.001
    for v, w in sorted(cands, reverse=True):
        print(f"   w={w:<5} {v:.4f}  ({v-S[CTRL][key]:+.4f})")
    print(f"   best w={bw} at {d:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   Below the noise floor notes/34 forced. Not submittable, whatever prediction")
        print("   3 said — a working mechanism that does not move the score is still a no.")

# 5 ---------------------------------------------------------------------------------
print("\n5. the optimum is not at the largest weight tried")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    order = sorted(WS)
    vals = [(w, S[f"w{w}+repair"][key]) for w in order if f"w{w}+repair" in S]
    bw = max(vals, key=lambda kv: kv[1])[0]
    ok5 = bw != order[-1]
    print("   " + "  ".join(f"w{w}:{v:.4f}" for w, v in vals))
    print(f"   best w={bw}  ->  {'PASS — interior or at the floor' if ok5 else 'FAIL — still climbing'}")
    if not ok5:
        print("   Extend past the largest weight before submitting. Three ILP sweeps needed")
        print("   three grids for exactly this reason.")

print()
print("=" * 95)
if EXACT:
    best = max((a for a in ARMS if a.endswith("+repair") and a in S),
               key=lambda a: S[a][key])
    d = S[best][key] - S[CTRL][key]
    print(f"BEST ARM: {best} at {S[best][key]:.4f}  ({d:+.4f} vs control)")
    if d <= 0.001:
        print("  INSIDE NOISE (<0.001). Bidirectional linking does not pay here; do not")
        print("  spend a submission slot. The second missing model (the temporal linker")
        print("  blend, notes/33 §1) is untouched by this result — it adds a model rather")
        print("  than re-reading the one we have.")
    else:
        print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
              f"undetected {A[best]['fn_detect']:,}")
else:
    print("NO BEST ARM — score column is NaN.")
print("=" * 95)
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
