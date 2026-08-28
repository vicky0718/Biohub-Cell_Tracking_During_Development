"""Build notebooks/claude_bidir_ilp.ipynb — re-locate the ILP weights with the blend on.

`notes/38`: bidirectional linking improves the **edge** term monotonically (+0.0049 at
w=0.15, +0.0073 at w=0.4) while `division_jaccard` collapses from 0.069 to 0.026. The best
total (+0.0036 at w=0.15) is not the best linker — it is the point where the edge gain
still outruns the division loss.

The two levers pull against each other, and the reason is understood. `notes/35` established
that divisions here come from the ILP's **termination penalty**: a high disappearance cost
makes the solver fork rather than end a track. The blend admits ~2.6 % more candidate edges,
which gives the solver cheaper ways to continue a track without forking, so it forks less.

The ILP weights were located in `notes/36` on **unblended** candidates. This re-locates them
with the blend active. If the division term is recoverable at the new operating point, the
total is worth ~+0.011 rather than +0.0036.

Structure: predict ONCE per dataset at w=0.4 -- the best edge term, and the arm with the
most division headroom to recover -- cache the candidates, then sweep the ILP over that
cache. One expensive prediction pass, many cheap solves, exactly as `claude_relink_sweep`'s
cache enabled three ILP sweeps for the price of one prediction.
"""
import ast
import json
from pathlib import Path

OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_bidir_ilp.ipynb")
SETUP = Path(__file__).with_name("_build_claude_submit_ratio.py")
N_DATASETS = 12
BLEND_W = 0.40          # notes/38: best EDGE term, most division to recover
# (appearance, disappearance) — notes/36's optimum is (0.4, 2.0), located on
# UNBLENDED candidates. The blend suppresses forks, so the termination penalty
# probably has to rise to compensate; the grid brackets 2.0 on both sides.
ILP_GRID = [(0.4, 2.0), (0.4, 3.0), (0.4, 4.0), (0.4, 6.0),
            (0.6, 3.0), (0.8, 4.0), (0.25, 2.0), (0.4, 1.4)]
CELLS = []
Q3 = chr(39) * 3


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    src = (src.replace("__N_DATASETS__", str(N_DATASETS))
              .replace("__BLEND_W__", repr(BLEND_W))
              .replace("__ILP_GRID__", repr(ILP_GRID)))
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# The blend traded divisions for edges. Can the ILP trade them back?

```
0.897  best scored (pack + ILP 0.4/2.0 + repair)
0.926  bronze     0.944  gold
```

`notes/38` measured bidirectional linking and found something the build did not predict:

```
              div_J    adj_edge     total
w0.0+repair   0.0690    0.9393     0.9462   <- control
w0.15+repair  0.0571    0.9442     0.9499   +0.0036  (best total)
w0.25+repair  0.0270    0.9458     0.9485   +0.0022
w0.4+repair   0.0263    0.9466     0.9492   +0.0030  (best EDGE term)
```

The **edge term improves monotonically** with the blend weight — +0.0049, +0.0065, +0.0073
— while `division_jaccard` collapses from 0.069 to 0.026. w=0.15 wins on total only because
it is where the edge gain still outruns the division loss.

## Why they fight, and why that is fixable

`notes/35` established that divisions here come from the ILP's **termination penalty**: a
high disappearance cost makes the solver fork rather than end a track. The blend admits
~2.6 % more candidate edges, giving the solver cheaper ways to *continue* a track without
forking. So it forks less.

**The ILP weights were located on unblended candidates** (`notes/36`, three sweeps, 69
settings). This re-locates them with the blend on. If the division term comes back at the
new operating point, the total is worth ~+0.011 rather than +0.0036.

Structure: predict **once** per dataset at w=0.4, cache the candidates, sweep the ILP over
that cache — one expensive pass, many cheap solves.

## Pre-registered predictions

1. **The cached blended candidates re-solve to `notes/38`'s `w0.4` numbers** at the ILP
   setting `notes/38` used (0.4 / 2.0): `div_J` ≈ 0.026, `edge_J` ≈ 0.944. Load-bearing —
   if the cache does not round-trip, nothing below is readable, exactly as in `notes/31`.
2. **Raising the disappearance penalty raises the fork count** on blended candidates, as it
   did on unblended ones. A mechanical check before any score is read.
3. **Some setting recovers `div_J` above 0.05** — back toward the 0.069 the unblended
   control had. This is the whole question: are the forks recoverable, or did the blend
   remove the *evidence* for them rather than just the incentive?
4. **The best blended arm beats `notes/38`'s best (0.9499) by more than 0.001.** The
   outcome claim, at the noise floor `notes/34` forced.
5. **The optimum is interior to the disappearance grid** — asked up front because three ILP
   sweeps each had to ask it again.

*Same 12 datasets as `notes/38`, stratified 5 x 44b6 / 7 x 6bba. `notes/38` §3: the control
must be graded against a number measured on the SAME datasets — 0.9462 and 0.9499 here, not
`notes/35`'s 24-dataset 0.9179.*
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
ILP_GRID = [tuple(g) for g in __ILP_GRID__]
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
LABELS = ["a" + str(ap) + "_d" + str(dis) for ap, dis in ILP_GRID]
LABELS = LABELS + [l + "+repair" for l in LABELS]
ROWS = dict((l, []) for l in LABELS); ANAT = dict((l, []) for l in LABELS)
EDGES = dict((l, 0) for l in LABELS); FORKS = dict((l, 0) for l in LABELS)
CAND = 0; PER = {{}}
pv = make_predict(BLEND_W)
print("blend weight", BLEND_W, "| ILP settings", len(ILP_GRID), flush=True)

def solve_at(g_td, ap, dis):
    # A docstring here would terminate the outer f-string that writes this file.
    if g_td.num_edges() == 0:
        return g_td
    solver = td.solvers.ILPSolver(
        edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
        appearance_weight=ap, disappearance_weight=dis, division_weight=ILP_DIV_W)
    with P.suppress_output():
        return solver.solve(g_td)

for name in names:
    t0 = time.time()
    sc = read_scale(TRAIN / (name + ".zarr"))
    gt = read_geff(TRAIN / (name + ".geff"))

    # ONE prediction pass. The candidates are then re-solved at every ILP setting, which is
    # what makes the grid cheap -- the same trick claude_relink_sweep's cache enabled for
    # three ILP sweeps.
    coords, edges = pv(model, TRAIN / (name + ".zarr"), DEV, cfg=cfg,
                       window_size=window_size, unet_batch_size=8, downsample=downsample)
    np.savez_compressed(WORK / ("cand_" + name + ".npz"),
                        t=np.asarray(coords)[:, 0].astype(np.int64),
                        zyx=np.asarray(coords)[:, 1:].astype(np.float32),
                        cand=np.asarray(edges, dtype=np.float32))
    t_pred = time.time() - t0
    parts = [name]
    for ap, dis in ILP_GRID:
        g_td = solve_at(P.build_graph(coords, edges), ap, dis)
        if not CAND:
            pass
        tr = Tracks.from_tracksdata(g_td)
        base = (tr.t, tr.zyx, tr.edges)
        for rep in (False, True):
            lbl = "a" + str(ap) + "_d" + str(dis) + ("+repair" if rep else "")
            g = repair_chain(base, sc) if rep else base
            ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            EDGES[lbl] += int(len(g[2]))
            nf = int((np.bincount(g[2][:, 0], minlength=len(g[0])) >= 2).sum()) if len(g[2]) else 0
            FORKS[lbl] += nf
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[lbl].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(name + "/" + lbl + ": buckets do not sum")
            PER.setdefault(name, {{}})[lbl] = float(
                ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
        parts.append("d" + str(dis) + " " + format(
            PER[name]["a" + str(ap) + "_d" + str(dis) + "+repair"], ".4f"))
    CAND += int(len(edges))
    print("  " + "  ".join(parts) + "   pred " + str(int(t_pred)) + "s total "
          + str(int(time.time() - t0)) + "s", flush=True)

    out = {{"arms": LABELS, "grid": [list(g) for g in ILP_GRID], "blend_w": BLEND_W,
           "datasets": [n for n in names if n in PER],
           "summary": dict((l, summarise(ROWS[l])) for l in LABELS if ROWS[l]),
           "anatomy": dict((l, summarise_anatomy(ANAT[l])) for l in LABELS if ANAT[l]),
           "edges": EDGES, "forks": FORKS, "candidates": CAND, "per_dataset": PER}}
    (WORK / "bidir_ilp.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "bidir_ilp.json").read_text())
S, A, E, F = D["summary"], D["anatomy"], D["edges"], D["forks"]
ARMS, DS, GRID, BW = D["arms"], D["datasets"], [tuple(g) for g in D["grid"]], D["blend_w"]
REF = "a0.4_d2.0+repair"          # notes/38's w0.4+repair, re-solved from the cache
EXACT = S[REF]["score"] == S[REF]["score"] if REF in S else False
key = "score" if EXACT else "edge_jaccard"
# notes/38 §3: grade against numbers measured on the SAME datasets, never a different pool.
N38_W0_REPAIR, N38_W04_REPAIR, N38_W015_REPAIR = 0.9462, 0.9492, 0.9499
N38_W04_DIVJ, N38_W04_EDGEJ = 0.0263, 0.9444
print(f"{len(DS)} datasets, {len(GRID)} ILP settings, blend weight {BW}")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

def dis_of(a):
    return float(a.split("_d")[1].replace("+repair", ""))

def app_of(a):
    return float(a.split("_d")[0][1:])

print(f"{'arm':<20}{'score':>9}{'vs n38':>9}{'edge_J':>9}{'div_J':>8}{'forks':>8}"
      f"{'mislink':>9}{'gap':>7}{'detect':>8}{'edges':>10}")
print("-" * 92)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    dj = st.get("division_jaccard")
    ref = N38_W04_REPAIR if a.endswith("+repair") else 0.9433
    print(f"{a:<20}{st[key]:>9.4f}{st[key]-ref:>+9.4f}{st['edge_jaccard']:>9.4f}"
          f"{(dj if dj == dj else 0):>8.4f}{F[a]:>8,}{an['fn_mislink']:>9,}"
          f"{an['fn_gap']:>7,}{an['fn_detect']:>8,}{E[a]:>10,}")

print()
print("=" * 92)
print("PREDICTION GRADING")
print("=" * 92)

print("\n1. the cached blended candidates re-solve to notes/38's w0.4 numbers")
if REF not in S:
    print(f"   NOT GRADED — {REF} missing.")
else:
    dj, ej = S[REF].get("division_jaccard", float("nan")), S[REF]["edge_jaccard"]
    ok_dj = abs(dj - N38_W04_DIVJ) <= 0.01
    ok_ej = abs(ej - N38_W04_EDGEJ) <= 0.005
    print(f"   div_J  {dj:.4f} vs {N38_W04_DIVJ:.4f}   {'PASS' if ok_dj else 'FAIL'}")
    print(f"   edge_J {ej:.4f} vs {N38_W04_EDGEJ:.4f}   {'PASS' if ok_ej else 'FAIL'}")
    if not (ok_dj and ok_ej):
        print("   The cache does NOT round-trip to the graph notes/38 scored. Nothing below")
        print("   is readable — diagnose that first (notes/31 made the same check for the")
        print("   same reason).")

print("\n2. raising the disappearance penalty raises the fork count")
at_a04 = sorted((dis_of(a), a) for a in ARMS
                if not a.endswith("+repair") and abs(app_of(a) - 0.4) < 1e-9)
if len(at_a04) >= 3:
    print("   " + "  ".join(f"dis={d:g}:{F[a]:,}" for d, a in at_a04))
    counts = [F[a] for _, a in at_a04]
    ok2 = all(b >= a for a, b in zip(counts, counts[1:]))
    print(f"   ->  {'PASS — the knob still works on blended candidates' if ok2 else 'FAIL — forks do not track the penalty'}")
else:
    print(f"   NOT GRADED — {len(at_a04)} arms at appearance 0.4.")

print("\n3. some setting recovers div_J above 0.05 (the unblended control had 0.069)")
djs = [(S[a].get("division_jaccard", 0.0), a) for a in ARMS if a in S]
djs = [(v, a) for v, a in djs if v == v]
if djs:
    bv, ba = max(djs)
    ok3 = bv > 0.05
    for v, a in sorted(djs, reverse=True)[:5]:
        print(f"   {a:<20} div_J {v:.4f}   forks {F[a]:,}")
    print(f"   best {ba} at {bv:.4f}  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   The blend removed the EVIDENCE for divisions, not just the incentive: no")
        print("   termination penalty in this grid brings them back. That closes the")
        print("   'trade them back' idea and makes w=0.15 the operating point notes/38")
        print("   already found, not a compromise to be improved on.")
    else:
        print("   The forks were recoverable — the blend removed the incentive, not the")
        print("   evidence. Check prediction 4 for whether the recovery pays for itself.")

print("\n4. the best blended arm beats notes/38's best (0.9499) by more than 0.001")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    cands = [(S[a][key], a) for a in ARMS if a.endswith("+repair") and a in S]
    bv, ba = max(cands)
    d = bv - N38_W015_REPAIR
    ok4 = d > 0.001
    for v, a in sorted(cands, reverse=True)[:5]:
        print(f"   {a:<20} {v:.4f}  ({v-N38_W015_REPAIR:+.4f} vs notes/38 best)")
    print(f"   best {ba} at {d:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   Re-locating the ILP with the blend on does not beat simply using a smaller")
        print("   blend weight. The two levers are substitutes, not complements, and w=0.15")
        print("   at the notes/36 weights stands.")

print("\n5. the optimum is interior to the disappearance grid")
if not EXACT or len(at_a04) < 3:
    print("   NOT GRADED.")
else:
    vals = [(d, S[a + "+repair"][key]) for d, a in at_a04 if a + "+repair" in S]
    if len(vals) >= 3:
        bd = max(vals, key=lambda kv: kv[1])[0]
        lo, hi = vals[0][0], vals[-1][0]
        ok5 = bd not in (lo, hi)
        print("   " + "  ".join(f"dis={d:g}:{v:.4f}" for d, v in vals))
        print(f"   best dis={bd:g}  ->  "
              f"{'PASS — interior' if ok5 else ('FAIL — still climbing' if bd == hi else 'at the grid floor')}")
    else:
        print("   NOT GRADED — too few repair arms.")

print()
print("=" * 92)
if EXACT:
    best = max((a for a in ARMS if a.endswith("+repair") and a in S), key=lambda a: S[a][key])
    d38 = S[best][key] - N38_W015_REPAIR
    d0 = S[best][key] - N38_W0_REPAIR
    print(f"BEST ARM: {best} at {S[best][key]:.4f}")
    print(f"  vs notes/38 no-blend control (0.9462):  {d0:+.4f}")
    print(f"  vs notes/38 best w0.15      (0.9499):  {d38:+.4f}   <- the delta that matters")
    print(f"  div_J {S[best].get('division_jaccard', 0):.4f}  edge_J {S[best]['edge_jaccard']:.4f}"
          f"  forks {F[best]:,}")
    if d38 <= 0.001:
        print("  INSIDE NOISE vs notes/38. The blend at w=0.15 with the notes/36 weights")
        print("  stands; re-locating the ILP does not add. Submit that, or move on to the")
        print("  temporal linker (notes/33 §1), which is still untouched.")
    else:
        print(f"  Submittable: blend w={BW} with appearance {app_of(best)} / "
              f"disappearance {dis_of(best)}.")
else:
    print("NO BEST ARM — score column is NaN.")
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
