"""Build notebooks/claude_relink_sweep.ipynb — put the model's edge_prob back to work.

`notes/26` split the edge loss and `fn_mislink` came out largest: **473 edges, 3.42 %** of
all GT edges, where both endpoints are detected and the graph joins the wrong pair.
`notes/27` swept pure geometry to exhaustion against it and repaired **12.5 %**.
`notes/28` then confirmed on the leaderboard (**0.867 -> 0.880**, +0.0130 against +0.0115
measured) that graph repair transfers, which is what makes the rest worth building.

What geometry does not have is information the pack already computed and threw away.
`predict_video` returns candidate edges as `(src, tgt, prob, dist)` -- a learned
probability for **every** candidate -- and the ILP consumes them into a global solution and
discards the rest. `pipeline/relink.py` puts that probability back into a local assignment,
alongside velocity continuity.

This run also **caches the candidate table with probabilities**, which no previous cache
has, so follow-up sweeps cost minutes instead of the ~25 min prediction pass.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_relink_sweep.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Motion relink — the model's own edge probabilities, put back to work

```
0.752  classical champion
0.867  pack + ILP
0.880  + gap-close + linefit-smooth      <- notes/28, +0.0130 on the leaderboard
0.913–0.916  the cluster, same weights   <- 0.033–0.036 away
```

`fn_mislink` is the largest remaining bucket — **473 edges, 3.42 %** of all GT edges, both
endpoints detected and the wrong pair joined. Geometry has been swept to exhaustion against
it (`notes/27`) and repaired 12.5 %.

The pack computes a learned probability for **every candidate edge** and the ILP throws the
losers away. This run puts it back, in a per-frame assignment whose cost is

```
cost = (1 − vw)·distance + vw·velocity_error − bonus·edge_prob
```

## Two things the module refuses to do, and why

- **Forks are held fixed.** The pack emits ~54 divisions per 24 datasets and the division
  term is worth 0.1 of the 1.1 maximum. A solver re-deriving whole frames collapses every
  fork to one child — trading a term we score 0.000 on for one we would score worse on.
- **A change budget.** `notes/27` §1 measured that node-rewiring repairs buy mislinks and
  pay in *detection failures*, ~4:1 where they work and inverting when pushed.
  `max_change_frac` caps the share of edges rewritten per frame, so a bad cost function
  degrades instead of destroying.

## Pre-registered predictions

1. **The control reproduces 0.8806 ± 0.0005.**
2. **Relink reduces `fn_mislink`.** It is the bucket the cost function targets; if it does
   not move, the cost function is wrong and no score change is interpretable.
3. **The learned bonus beats geometry+velocity alone.** If `bonus=0` matches `bonus=0.78`,
   `edge_prob` adds nothing and this whole premise is dead — which is the single most
   valuable thing this run can tell me.
4. **Score is single-peaked in `max_change_frac`**, and rewriting more is not monotonically
   better. If it *is* monotone, the budget is not binding and the governor is untested.

*Training data, contaminated for these weights (`notes/24` §2). `notes/28` measured a
transfer ratio of 1.13× on the one submission made — a direction, not a coefficient.*
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
        print(r.stdout[-1500:]); print(r.stderr[-1500:])
    return r.returncode == 0

gpu = sh("nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader").stdout.strip()
print("accelerator:"); [print("  " + l) for l in gpu.splitlines() if l.strip()]
if "P100" in gpu:
    print("P100 (sm_60) -> replacing torch")
    print(f"  torch replacement "
          f"{'ok' if pip_install(['torch==2.5.1'], extra=('--index-url', 'https://download.pytorch.org/whl/cu121')) else 'FAILED'}")

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

PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(), ["/kaggle/input"])
REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
for lbl, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP)):
    print(f"  {lbl:<12} {v}")
if None in (PACK, REPO, COMP):
    raise SystemExit("missing mount")
TRAIN = COMP / "train"
if not (REPO / "pipeline" / "relink.py").exists():
    raise SystemExit("the mounted repo predates pipeline/relink.py — push a new dataset "
                     "version before running this")

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")

CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    r = sh("git", "clone", "--depth", "1",
           "https://github.com/royerlab/kaggle-cell-tracking-competition", str(CELLMOT))
    print(f"official scorer clone rc={r.returncode}")
os.environ["CELLMOT_REPO"] = str(CELLMOT)

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, tracksdata; print('numpy', numpy.__version__, "
           "'| torch', torch.__version__, '| cuda', torch.cuda.is_available())")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")

N_DATASETS, DET_THRESHOLD = 24, 0.985
print(f"config: n={N_DATASETS}, det_threshold={DET_THRESHOLD}, setup {time.time()-T_START:.0f}s")
""")

md("""## 1. Predict once (caching probabilities), then sweep

The cache written here carries the **candidate table with `edge_prob`**, which no previous
cache has. Attaching this kernel to a follow-up makes the next sweep cost minutes.
""")

code(r"""
WORKER = WORK / "run_relink.py"
WORKER.write_text(f'''
import json, os, sys, time, types
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
PACK = Path({str(PACK)!r}); REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
WORK = Path({str(WORK)!r}); N_DATASETS = {N_DATASETS}; DET_THRESHOLD = {DET_THRESHOLD}
T0 = time.time()

import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(PACK / "repo" / "src"))
sys.path.insert(0, str(PACK / "repo" / "scripts"))
_ds = types.ModuleType("dataspec")
_ds.USERNAME = "claude"; _ds.INTERACTIVE = False
_ds.WEIGHTS_PATH = PACK / "weights"; _ds.DATASET_PATH = TRAIN
_ds.PREDICTIONS_PATH = WORK / "predictions"
sys.modules["dataspec"] = _ds

import predict_unet_transformer as P
import tracksdata as td
from harness import Harness
from harness.tracks import Tracks, read_geff, read_estimated_nodes, read_scale
from harness.purescore import summarise
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy
from pipeline.relink import motion_relink
from pipeline.repair import close_gaps, linefit_smooth
print("worker numpy", np.__version__, "torch", torch.__version__, DEV, flush=True)

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)

names = sorted({{p.stem for p in TRAIN.glob("*.zarr")}} & {{p.stem for p in TRAIN.glob("*.geff")}})
budgets = {{}}
for n in names:
    b = read_estimated_nodes(TRAIN / f"{{n}}.geff")
    if b == b and b > 0:
        budgets[n] = float(b)
ordered = sorted(budgets, key=lambda n: budgets[n])
idx = np.linspace(0, len(ordered) - 1, N_DATASETS).astype(int)
SUBSET = [ordered[i] for i in sorted(set(idx.tolist()))]
print(f"{{len(SUBSET)}} datasets, budget-stratified (notes/25 §2)", flush=True)

# ---- the sweep -------------------------------------------------------------------
def R(**kw):
    def f(g, sc, cand):
        e = motion_relink(g[0], g[1], g[2], cand=cand, scale=sc, **kw)
        return (g[0], g[1], e)
    return f

def repair_chain(g, sc):
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038, max_added_abs=1650)
    return linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

ARMS = [("control", None, False)]
# Isolate velocity: geometry alone vs geometry+velocity, both WITHOUT probabilities.
ARMS += [("geom_only",  R(velocity_weight=0.0, learned_bonus=0.0, max_change_frac=0.10), False),
         ("geom_vel",   R(velocity_weight=0.5, learned_bonus=0.0, max_change_frac=0.10), False)]
# Prediction 3: does edge_prob add anything on top of that?
for b in (0.78, 2.0, 5.0):
    ARMS.append((f"prob{{b}}", R(velocity_weight=0.5, learned_bonus=b,
                                max_change_frac=0.10), True))
# Prediction 4: is the change budget binding?
for fr in (0.03, 0.05, 0.25, 0.50):
    ARMS.append((f"frac{{fr}}", R(velocity_weight=0.5, learned_bonus=0.78,
                                 max_change_frac=fr), True))
# Radii, since notes/27 found the copied constants were already near-optimal elsewhere.
ARMS.append(("wide", R(tight_um=7.5, relaxed_um=13.0, velocity_weight=0.5,
                       learned_bonus=0.78, max_change_frac=0.10), True))
ARMS.append(("tightr", R(tight_um=4.5, relaxed_um=7.0, velocity_weight=0.5,
                         learned_bonus=0.78, max_change_frac=0.10), True))
print(f"{{len(ARMS)}} arms + each composed with the notes/28 repair chain", flush=True)

ILP_W = (-1.0, 0.1, 0.1, 1.0)
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      ilp_edge_weight=ILP_W[0], ilp_appearance_weight=ILP_W[1],
                      ilp_disappearance_weight=ILP_W[2], ilp_division_weight=ILP_W[3])
h = Harness(data_dir=TRAIN, cache_dir=None)
LABELS = [a for a, _, _ in ARMS] + [f"{{a}}+repair" for a, _, _ in ARMS]
ROWS = {{l: [] for l in LABELS}}
ANAT = {{l: [] for l in LABELS}}
EDGES = {{l: 0 for l in LABELS}}
PER = {{}}

for name in SUBSET:
    t0 = time.time()
    coords, edges = P.predict_video(model, TRAIN / f"{{name}}.zarr", DEV, cfg=cfg,
                                    window_size=window_size, unet_batch_size=4,
                                    downsample=downsample)
    cand = np.asarray(edges, float)          # (K, 4): src, tgt, prob, dist
    g_td = P.build_graph(coords, edges)
    if g_td.num_edges() > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=ILP_W[0] * td.EdgeAttr("edge_prob"),
            appearance_weight=ILP_W[1], disappearance_weight=ILP_W[2],
            division_weight=ILP_W[3])
        with P.suppress_output():
            g_td = solver.solve(g_td)
    base_g = Tracks.from_tracksdata(g_td)
    base = (base_g.t, base_g.zyx, base_g.edges)
    sc = read_scale(TRAIN / f"{{name}}.zarr")
    gt = read_geff(TRAIN / f"{{name}}.geff")
    # THE CACHE no previous run has: candidate edges WITH probabilities.
    np.savez_compressed(WORK / f"cand_{{name}}.npz", t=base[0], zyx=base[1],
                        edges=base[2], cand=cand)
    print(f"\\n{{name}}  pred={{base_g.n_nodes:,}} edges={{base_g.n_edges:,}} "
          f"cand={{len(cand):,}} forks={{base_g.n_divisions}} {{time.time()-t0:.0f}}s",
          flush=True)

    for arm, fn, needs_prob in ARMS:
        for with_repair in (False, True):
            lbl = f"{{arm}}+repair" if with_repair else arm
            g = base if fn is None else fn(base, sc, cand if needs_prob else None)
            if with_repair:
                g = repair_chain(g, sc)
            ROWS[lbl].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            EDGES[lbl] += int(len(g[2]))
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[lbl].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(f"{{name}}/{{lbl}}: buckets do not sum")
            PER.setdefault(name, {{}})[lbl] = float(
                ROWS[lbl][-1].get("adj_edge_jaccard", float("nan")))
    best = max(PER[name], key=lambda k: PER[name][k] if PER[name][k] == PER[name][k] else -9)
    print(f"    control {{PER[name]['control']:.4f}}  best {{best}} {{PER[name][best]:.4f}} "
          f"({{PER[name][best]-PER[name]['control']:+.4f}})  {{time.time()-t0:.0f}}s total",
          flush=True)

    out = {{"arms": LABELS, "datasets": [n for n in SUBSET if n in PER],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "edges": EDGES, "per_dataset": PER}}
    (WORK / "relink_sweep.json").write_text(json.dumps(out, indent=2, default=float))

print(f"\\nworker done in {{time.time()-T0:.0f}}s", flush=True)
''')

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"\nworker exited {rc} after {time.time()-t0:.0f}s")
if rc != 0:
    raise SystemExit(f"worker failed ({rc})")
""")

md("""## 2. The four predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "relink_sweep.json").read_text())
S, A, E, ARMS, DS = D["summary"], D["anatomy"], D["edges"], D["arms"], D["datasets"]
base, abase = S["control"], A["control"]
EXACT = base["score"] == base["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(ARMS)} arms")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

print(f"{'arm':<20}{'score':>9}{'delta':>9}{'edge_J':>9}"
      f"{'d_mislink':>11}{'d_detect':>10}{'d_gap':>8}{'d_edges':>10}")
print("-" * 86)
for a in ARMS:
    if a not in S:
        continue
    s, an = S[a], A[a]
    print(f"{a:<20}{s[key]:>9.4f}{s[key]-base[key]:>+9.4f}{s['edge_jaccard']:>9.4f}"
          f"{an['fn_mislink']-abase['fn_mislink']:>+11}"
          f"{an['fn_detect']-abase['fn_detect']:>+10}"
          f"{an['fn_gap']-abase['fn_gap']:>+8}{E[a]-E['control']:>+10,}")

print()
print("=" * 86)
print("PREDICTION GRADING")
print("=" * 86)

print("\n1. the control reproduces 0.8806 +- 0.0005")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
elif len(DS) != 24:
    print(f"   NOT GRADED — {len(DS)} datasets completed, not 24.")
else:
    ok = abs(base["score"] - 0.8806) <= 0.0005
    print(f"   control = {base['score']:.4f}  ->  {'PASS' if ok else 'FAIL'}")

print("\n2. relink reduces fn_mislink  (the bucket the cost function targets)")
pure = [a for a in ARMS if a != "control" and not a.endswith("+repair")]
if pure:
    best_m = min(pure, key=lambda a: A[a]["fn_mislink"])
    d = abase["fn_mislink"] - A[best_m]["fn_mislink"]
    print(f"   best: {best_m}  mislink {abase['fn_mislink']:,} -> "
          f"{A[best_m]['fn_mislink']:,} ({d:+})  ->  {'PASS' if d > 0 else 'FAIL'}")
    if d <= 0:
        print("   The cost function does not target what it was built for. No score")
        print("   change below is interpretable as relink working — diagnose first.")

print("\n3. the learned bonus beats geometry+velocity alone  (is edge_prob worth anything?)")
if "geom_vel" in S and "prob0.78" in S:
    gv, pb = S["geom_vel"][key], S["prob0.78"][key]
    best_p = max([a for a in pure if a.startswith("prob")], key=lambda a: S[a][key])
    print(f"   geom_vel {gv-base[key]:+.4f}   prob0.78 {pb-base[key]:+.4f}   "
          f"best prob arm {best_p} {S[best_p][key]-base[key]:+.4f}")
    ok3 = S[best_p][key] > gv + 1e-6
    print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   edge_prob adds NOTHING over geometry and velocity. That is the single")
        print("   most valuable result this run can produce: the premise that the ILP")
        print("   discards useful information is wrong, and relink is a dead end.")
    if "geom_only" in S:
        print(f"   (velocity alone contributed "
              f"{S['geom_vel'][key]-S['geom_only'][key]:+.4f} on top of geometry)")

print("\n4. score is single-peaked in max_change_frac")
curve = [(0.0, base[key])]
for lbl, fr in (("frac0.03", 0.03), ("frac0.05", 0.05), ("prob0.78", 0.10),
                ("frac0.25", 0.25), ("frac0.5", 0.50)):
    if lbl in S:
        curve.append((fr, S[lbl][key]))
curve.sort()
if len(curve) >= 4:
    ys = [y for _, y in curve]
    pk = int(np.argmax(ys))
    print("   " + "  ".join(f"{f:g}={y:.4f}" for f, y in curve))
    print(f"   peak at frac={curve[pk][0]:g}")
    mono = all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))
    print(f"   ->  {'FAIL — monotone, the budget never binds and the governor is untested' if mono else 'PASS'}")

print()
print("=" * 86)
best = max((a for a in ARMS if a != "control"),
           key=lambda a: (S[a][key] if a in S and S[a][key] == S[a][key] else float("-inf")))
print(f"BEST ARM: {best} at {S[best][key]:.4f} ({S[best][key]-base[key]:+.4f})")
print(f"  notes/28 chain alone: {S['control+repair'][key]-base[key]:+.4f}  "
      f"(this run's reproduction of the +0.0115 that scored 0.880)")
print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
      f"undetected {A[best]['fn_detect']:,}")
print(f"  reachable band was +0.047..+0.079 on train; notes/28 measured 1.13x transfer")
print("=" * 86)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
