"""Build notebooks/claude_div_probe.ipynb — measure the division term's price curve.

`notes/25` §1: the pack's ILP output scores **0.0000 of the 0.1000** the division term is
worth (37 forks over 24 datasets, TP=0, FP=0, FN=27) while edge Jaccard sits at 0.9293 and
the node-budget multiplier at 1.0012. Divisions are the only term with real room left, and
the whole 0.046 gap to the cluster is less than half of what is unclaimed there.

The arithmetic (read off the official `division_metrics.py`, not guessed) says a
speculative fork is cheap: `division_jaccard = TP / (FP + D)` with `D` fixed by the ground
truth, and a fork is only a chargeable FP if it matched an annotated GT node, sat inside a
GT division window, or is structurally invalid. With annotation at 1-in-8 to 1-in-167, most
forks are unevaluable and cost nothing.

**What that argument cannot give is the number.** How fast chargeable FPs actually
accumulate when forks are placed deliberately depends on the matched-node fraction, which
runs 1.3 %-6.5 % across this subset. So this is a sweep, not a bet.

## Design

**One prediction pass, many scorings.** The 24 post-ILP graphs cost 1,442 s to produce in
`claude_pack_diag`; re-predicting per sweep cell would be absurd. Each dataset is predicted
once and every sweep variant is scored against it before moving on.

**Degrade by dataset count, not by sweep completeness.** The running table is printed after
every dataset, so a truncated run still yields a complete sweep over fewer datasets rather
than a partial sweep over all of them. Datasets are visited in bisection order, so any
prefix stays spread across the size range instead of being all-small or all-large.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_div_probe.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# What does a speculative division actually cost?

`notes/25` located the gap. This run prices the fix.

| term | ours (pack + ILP) | available |
|---|---|---|
| adjusted edge Jaccard | 0.9304 | ~1.0 |
| node-budget multiplier | 1.0012 | ~1.01 reachable |
| **division Jaccard** | **0.0000** | **0.1000** |

37 forks over 24 datasets: **TP = 0, FP = 0, FN = 27**. Every one unevaluable.

## Why this might be nearly free, and why that has to be measured

From the official `division_metrics.py`:

- `FN = D − TP` by construction and `summarise` micro-averages, so
  **`division_jaccard = TP / (FP + D)`** with `D` fixed by the ground truth. A true
  positive never raises the denominator.
- A fork is a **chargeable** FP only if it matched a GT node with out-degree ≥ 1, sat
  inside a GT division window and failed the topology test, or is structurally invalid.
- Ground truth is annotated at 1-in-8 (`6bba`) to 1-in-167 (`44b6`), so **1.3 %–6.5 %** of
  predicted nodes are matched at all. Most forks land where nothing can charge them.

The same link is also an **edge**, and `notes/04` §10's break-even applies there:
`p > J/(1+J)` ≈ 48 % at J = 0.93. But one edge is ~1/5,000 of a dataset's edge Jaccard and
one division is ~1/27 of the pooled division Jaccard — three orders of magnitude apart per
event. **That asymmetry is the whole thesis, and cells 2–3 test it rather than assert it.**

## Pre-registered predictions

1. **`cap = 0` reproduces 0.9304 ± 0.0005.** If not, the cache or the scoring path is
   broken and nothing else here is readable.
2. **Chargeable FPs grow sub-linearly in forks emitted.** If FP tracks forks 1:1, the
   unevaluable-surface argument is wrong and the division angle is far weaker.
3. **The score curve is single-peaked in the cap**, peaking well above the pack's current
   ~1.5 forks/dataset.
4. **Edge Jaccard falls monotonically with forks, by less than the division term gains at
   the peak.** If edge losses dominate, the 1000× leverage estimate is wrong.

Failing 2 or 4 kills the division angle and sends the effort to edge repair. That is the
point of running the sweep before building the inserter properly.

*Scores are on training data, which `notes/24` §2 established is contaminated for these
weights. **Absolute levels are inflated.** Every number this run is for is a delta across
the sweep on fixed datasets, which contamination shifts equally.*
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
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'}")

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

# The inserter must be the version under test, not a stale copy baked into a dataset.
if not (REPO / "pipeline" / "divisions.py").exists():
    raise SystemExit("the mounted repo predates pipeline/divisions.py — push a new "
                     "dataset version before running this")

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")

# The official scorer: this run predicts divisions on purpose, and purescore's division
# term is exact only for fork-free graphs (notes/24 §1).
CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    r = sh("git", "clone", "--depth", "1",
           "https://github.com/royerlab/kaggle-cell-tracking-competition", str(CELLMOT))
    print(f"official scorer clone rc={r.returncode}")
os.environ["CELLMOT_REPO"] = str(CELLMOT)

probe = sh(sys.executable, "-c",
           "import numpy, torch, zarr, tracksdata; "
           "print('numpy', numpy.__version__, '| torch', torch.__version__, "
           "'| cuda', torch.cuda.is_available())")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")

N_DATASETS   = 24       # stratified by the scorer's node budget (notes/25 §2)
DET_THRESHOLD = 0.985   # the PUBLIC NOTEBOOK's value, so this is comparable to pack_diag
WALL_BUDGET_S = 9 * 3600
print(f"config: n={N_DATASETS}, det_threshold={DET_THRESHOLD}, "
      f"wall budget {WALL_BUDGET_S/3600:.0f} h")
""")

md("""## 1. Predict once per dataset, score every sweep variant against it

In a subprocess: the pack's wheels upgrade numpy under a process that has already
imported the old one (`notes/24` §1).

The running table prints after each dataset, so a run that is cut short still leaves a
**complete sweep over fewer datasets** rather than a partial sweep over all of them.
""")

code(r"""
WORKER = WORK / "run_div_probe.py"
WORKER.write_text(f'''
import json, os, sys, time, types
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
PACK = Path({str(PACK)!r}); REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
WORK = Path({str(WORK)!r})
N_DATASETS = {N_DATASETS}
DET_THRESHOLD = {DET_THRESHOLD}
WALL_BUDGET_S = {WALL_BUDGET_S}
T0 = time.time()

import torch
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("worker numpy", np.__version__, "torch", torch.__version__, DEV, flush=True)

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
from pipeline.divisions import insert_divisions

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)
print(f"model window_size={{window_size}} downsample={{downsample}}", flush=True)

names = sorted({{p.stem for p in TRAIN.glob("*.zarr")}} & {{p.stem for p in TRAIN.glob("*.geff")}})

# Stratify on the scorer's node budget, never the annotated count -- notes/25 §2 records
# why (annotation rate varies 20x, so the annotated count sorts by labelling protocol).
budgets = {{}}
for n in names:
    b = read_estimated_nodes(TRAIN / f"{{n}}.geff")
    if b == b and b > 0:
        budgets[n] = float(b)
ordered = sorted(budgets, key=lambda n: budgets[n])
idx = np.linspace(0, len(ordered) - 1, N_DATASETS).astype(int)
SUBSET = [ordered[i] for i in sorted(set(idx.tolist()))]

def spread_order(n):
    # BFS bisection: any prefix of the result is spread across the whole range, so a
    # truncated run is not biased to one end of the size distribution.
    out, seen, queue = [], set(), [(0, n - 1)]
    while queue:
        lo, hi = queue.pop(0)
        if lo > hi:
            continue
        mid = (lo + hi) // 2
        if mid not in seen:
            seen.add(mid); out.append(mid)
        queue.append((lo, mid - 1)); queue.append((mid + 1, hi))
    return out

VISIT = [SUBSET[i] for i in spread_order(len(SUBSET))]
print(f"{{len(ordered)}} datasets, budget {{budgets[ordered[0]]:,.0f}}..{{budgets[ordered[-1]]:,.0f}}",
      flush=True)
print(f"subset {{len(SUBSET)}}, visited in bisection order so any prefix stays stratified",
      flush=True)

# ---- the sweep -------------------------------------------------------------------
# cap=0 is the control and is identical at every radius, so it is run once.
SISTER_MAX_UM = 6.8
GRID = [("control", 0.0, 4.5)]
for mx in (4.5, 6.0):
    for cap in (0.002, 0.008, 0.02, 0.05):
        GRID.append((f"cap{{cap}}_r{{mx}}", cap, mx))
print(f"sweep: {{len(GRID)}} cells x sister_max_um={{SISTER_MAX_UM}}", flush=True)

ILP_EDGE_W, ILP_APP_W, ILP_DIS_W, ILP_DIV_W = -1.0, 0.1, 0.1, 1.0
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      ilp_edge_weight=ILP_EDGE_W, ilp_appearance_weight=ILP_APP_W,
                      ilp_disappearance_weight=ILP_DIS_W, ilp_division_weight=ILP_DIV_W)

h = Harness(data_dir=TRAIN, cache_dir=None)
ROWS = {{label: [] for label, _, _ in GRID}}      # label -> list of per-dataset metric rows
FORKS = {{label: 0 for label, _, _ in GRID}}      # label -> total forking nodes emitted
ADDED = {{label: 0 for label, _, _ in GRID}}      # label -> total edges inserted
PER = {{}}
done = []

def n_forks(edges, n):
    if len(edges) == 0:
        return 0
    return int((np.bincount(np.asarray(edges)[:, 0], minlength=n) >= 2).sum())

def report():
    print()
    print("=" * 96, flush=True)
    print(f"RUNNING TABLE after {{len(done)}} dataset(s), {{(time.time()-T0)/60:.1f}} min",
          flush=True)
    print(f"{{'cell':<16}}{{'score':>9}}{{'adj_edge':>10}}{{'edge_J':>9}}{{'div_J':>8}}"
          f"{{'dTP':>6}}{{'dFP':>6}}{{'dFN':>6}}{{'forks':>8}}{{'added':>8}}", flush=True)
    print("-" * 96, flush=True)
    base = None
    for label, _, _ in GRID:
        rows = ROWS[label]
        if not rows:
            continue
        s = summarise(rows)
        if base is None:
            base = s["score"]
        print(f"{{label:<16}}{{s['score']:>9.4f}}{{s['adj_edge_jaccard']:>10.4f}}"
              f"{{s['edge_jaccard']:>9.4f}}"
              f"{{(s['division_jaccard'] if s['division_jaccard']==s['division_jaccard'] else 0):>8.4f}}"
              f"{{int(s['division_tp']):>6}}{{int(s['division_fp']):>6}}"
              f"{{int(s['division_fn']):>6}}{{FORKS[label]:>8,}}{{ADDED[label]:>8,}}",
              flush=True)
    print("-" * 96, flush=True)
    if base is not None:
        print(f"deltas vs control:", flush=True)
        for label, _, _ in GRID[1:]:
            if ROWS[label]:
                s = summarise(ROWS[label])
                print(f"  {{label:<16}} {{s['score'] - base:+.4f}}", flush=True)
    print("=" * 96, flush=True)

for name in VISIT:
    if time.time() - T0 > WALL_BUDGET_S:
        print(f"\\nwall budget reached after {{len(done)}} datasets — stopping cleanly",
              flush=True)
        break
    t0 = time.time()
    coords, edges = P.predict_video(model, TRAIN / f"{{name}}.zarr", DEV, cfg=cfg,
                                    window_size=window_size, unet_batch_size=4,
                                    downsample=downsample)
    g_td = P.build_graph(coords, edges)
    n_cand = g_td.num_edges()
    if n_cand > 0:
        solver = td.solvers.ILPSolver(
            edge_weight=ILP_EDGE_W * td.EdgeAttr("edge_prob"),
            appearance_weight=ILP_APP_W, disappearance_weight=ILP_DIS_W,
            division_weight=ILP_DIV_W)
        with P.suppress_output():
            g_td = solver.solve(g_td)
    base_g = Tracks.from_tracksdata(g_td)
    t_pred = time.time() - t0

    # Cache, so a follow-up sweep never re-pays the prediction cost.
    np.savez_compressed(WORK / f"cache_{{name}}.npz",
                        t=base_g.t, zyx=base_g.zyx, edges=base_g.edges)

    scale = read_scale(TRAIN / f"{{name}}.zarr")
    print(f"\\n{{name}}  budget={{budgets[name]:,.0f}}  pred={{base_g.n_nodes:,}}  "
          f"edges={{base_g.n_edges:,}}  base_forks={{base_g.n_divisions}}  "
          f"predict {{t_pred:.0f}}s  scale={{tuple(round(s,4) for s in scale)}}", flush=True)

    for label, cap, mx in GRID:
        ts = time.time()
        if cap <= 0:
            g = base_g
            added = np.zeros((0, 2), np.int64)
        else:
            added = insert_divisions(base_g.t, base_g.zyx, base_g.edges, scale=scale,
                                     max_um=mx, sister_max_um=SISTER_MAX_UM,
                                     frame_frac_cap=cap)
            e2 = (np.vstack([base_g.edges, added]) if len(added) else base_g.edges)
            g = Tracks(base_g.t, base_g.zyx, e2)
        row = h.score_graph(name, g)
        ROWS[label].append(row)
        FORKS[label] += n_forks(g.edges, g.n_nodes)
        ADDED[label] += int(len(added))
        PER.setdefault(name, {{}})[label] = {{
            "added": int(len(added)), "forks": n_forks(g.edges, g.n_nodes),
            "score": float(row.get("adj_edge_jaccard", float("nan"))),
            "edge_jaccard": float(row.get("edge_jaccard", float("nan"))),
            "division_tp": int(row.get("division_tp", 0)),
            "division_fp": int(row.get("division_fp", 0)),
            "division_fn": int(row.get("division_fn", 0)),
        }}
        print(f"    {{label:<16}} +{{len(added):>5,}} edges  forks={{n_forks(g.edges, g.n_nodes):>5,}}  "
              f"edge_J={{row.get('edge_jaccard', float('nan')):.4f}}  "
              f"div TP/FP/FN={{int(row.get('division_tp',0))}}/{{int(row.get('division_fp',0))}}"
              f"/{{int(row.get('division_fn',0))}}  ({{time.time()-ts:.0f}}s)", flush=True)

    done.append(name)
    PER[name]["budget"] = float(budgets[name])
    PER[name]["pred_nodes"] = int(base_g.n_nodes)
    report()

    out = {{"grid": [{{"label": l, "cap": c, "max_um": m}} for l, c, m in GRID],
           "sister_max_um": SISTER_MAX_UM, "det_threshold": DET_THRESHOLD,
           "datasets": done, "per_dataset": PER,
           "summary": {{l: summarise(ROWS[l]) for l, _, _ in GRID if ROWS[l]}},
           "forks": FORKS, "added": ADDED}}
    (WORK / "div_probe.json").write_text(json.dumps(out, indent=2, default=float))

print("\\nworker done", flush=True)
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

md("""## 2. Grade the four pre-registered predictions

Each is graded from the run's own numbers. A failure here is a result, not a bug — two of
the four are specifically designed so that failing them kills the division angle.
""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "div_probe.json").read_text())
GRID, S, FORKS, ADDED = D["grid"], D["summary"], D["forks"], D["added"]
n_ds = len(D["datasets"])
print(f"{n_ds} datasets, {len(GRID)} sweep cells, sister_max_um={D['sister_max_um']}\n")

def cell(label):
    s = S.get(label)
    if not s:
        return None
    dj = s.get("division_jaccard")
    return {"score": s["score"], "adj": s["adj_edge_jaccard"], "ej": s["edge_jaccard"],
            "dj": (dj if dj == dj else 0.0), "tp": int(s["division_tp"]),
            "fp": int(s["division_fp"]), "fn": int(s["division_fn"]),
            "forks": FORKS[label], "added": ADDED[label]}

base = cell("control")

# The node budget can be unreadable (a .geff without `estimated_number_of_nodes`), which
# makes adj_edge_jaccard and therefore `score` NaN. Silently grading NaN would print
# "FAIL: edge losses dominate" on missing data and send the whole effort the wrong way, so
# detect it, say so, and grade on the unadjusted combination instead.
EXACT = base["score"] == base["score"]
def sc(c):
    return c["score"] if EXACT else c["ej"] + 0.1 * c["dj"]
if not EXACT:
    print("!! the control's score is NaN: `estimated_number_of_nodes` was unreadable, so")
    print("   the node-budget multiplier and every adjusted score are undefined. The")
    print("   division columns below are still exact. Grading falls back to")
    print("   edge_jaccard + 0.1*division_jaccard, which omits the multiplier — that is")
    print("   a substitution, not the metric, and prediction 1 cannot be graded at all.\n")

print(f"{'cell':<16}{'score':>9}{'delta':>9}{'edge_J':>9}{'div_J':>8}"
      f"{'dTP':>6}{'dFP':>6}{'forks':>8}{'added':>8}")
print("-" * 79)
rows = []
for g in GRID:
    c = cell(g["label"])
    if not c:
        continue
    rows.append((g, c))
    print(f"{g['label']:<16}{sc(c):>9.4f}{sc(c)-sc(base):>+9.4f}"
          f"{c['ej']:>9.4f}{c['dj']:>8.4f}{c['tp']:>6}{c['fp']:>6}"
          f"{c['forks']:>8,}{c['added']:>8,}")

print()
print("=" * 79)
print("PREDICTION GRADING")
print("=" * 79)

# 1 --------------------------------------------------------------------------------
print(f"\n1. control reproduces pack_diag's 0.9304 +- 0.0005")
if not EXACT:
    print("   NOT GRADED — the score column is a substitution, see the note above.")
elif n_ds != 24:
    print(f"   NOT GRADED — pack_diag ran 24 datasets, this run has {n_ds}, so the control")
    print(f"   legitimately shifts. control = {base['score']:.4f}, for the record.")
else:
    ok1 = abs(base["score"] - 0.9304) <= 0.0005
    print(f"   control = {base['score']:.4f}   ->  {'PASS' if ok1 else 'FAIL'}")
    if not ok1:
        print("   The control is the whole run's anchor. A miss means the graphs or the")
        print("   scoring path differ from pack_diag and NOTHING below is readable —")
        print("   including any apparent gain. Diagnose this before reading further.")

# 2 --------------------------------------------------------------------------------
# MIN_ADDED: below this, FP-per-1k is dominated by whether a single fork happened to be
# chargeable. The synthetic dry run produced 25,000 FP/1k off 2 inserted edges and would
# have "passed" this check on noise.
MIN_ADDED = 50
swept = [(g, c) for g, c in rows if g["cap"] > 0 and c["added"] >= MIN_ADDED]
print(f"\n2. chargeable FPs grow SUB-linearly in forks emitted")
if len(swept) >= 2:
    print(f"   {'cell':<16}{'added':>9}{'dFP':>7}{'FP per 1k added':>18}")
    ratios = []
    for g, c in sorted(swept, key=lambda gc: gc[1]["added"]):
        r = 1000.0 * c["fp"] / c["added"]
        ratios.append((c["added"], r))
        print(f"   {g['label']:<16}{c['added']:>9,}{c['fp']:>7}{r:>18.2f}")
    lo = float(np.mean([r for a, r in ratios[:max(1, len(ratios)//3)]]))
    hi = float(np.mean([r for a, r in ratios[-max(1, len(ratios)//3):]]))
    ok2 = hi <= lo * 1.5
    print(f"   FP/1k at the small end {lo:.2f} -> large end {hi:.2f}  ->  "
          f"{'PASS (sub-linear or flat)' if ok2 else 'FAIL (FP scales with forks)'}")
    if not ok2:
        print("   The unevaluable-surface argument in notes/25 §1 is wrong: deliberately")
        print("   placed forks DO land where the scorer can charge them. The division")
        print("   term is then priced like any other and the angle is much weaker.")
else:
    ok2 = None
    print(f"   NOT GRADED — fewer than 2 cells inserted at least {MIN_ADDED} edges, and")
    print("   an FP ratio off a handful of insertions measures nothing but luck.")

# 3 --------------------------------------------------------------------------------
print(f"\n3. the score curve is single-peaked in the cap")
for mx in sorted({g["max_um"] for g in GRID if g["cap"] > 0}):
    curve = [(g["cap"], sc(c)) for g, c in rows if g["cap"] > 0 and g["max_um"] == mx]
    curve = [(0.0, sc(base))] + sorted(curve)
    if len(curve) < 3:
        continue
    ys = [y for _, y in curve]
    peak = int(np.argmax(ys))
    up = all(ys[i] <= ys[i + 1] + 1e-9 for i in range(peak))
    down = all(ys[i] >= ys[i + 1] - 1e-9 for i in range(peak, len(ys) - 1))
    print(f"   max_um={mx}:  " + "  ".join(f"cap{c:g}={y:.4f}" for c, y in curve))
    print(f"      peak at cap={curve[peak][0]:g} ({ys[peak]:+.4f} vs control), "
          f"{'single-peaked' if up and down else 'NOT single-peaked'}")
    if peak == 0:
        print("      peak IS the control: every insertion setting tried lost points.")

# 4 --------------------------------------------------------------------------------
print(f"\n4. edge Jaccard falls with forks, by less than divisions gain at the peak")
best = max(rows[1:], key=lambda gc: sc(gc[1])) if len(rows) > 1 else None
if best and best[1]["added"] == 0:
    print("   NOT GRADED — the highest-scoring swept cell inserted nothing, so there is")
    print("   no trade to price. Every cap tried was too small to fire on these frames.")
elif best:
    g, c = best
    # The edge side is the ADJUSTED edge Jaccard when it exists, because inserting nodes'
    # worth of edges cannot move node count but the multiplier still scales the term.
    d_edge = (c["adj"] - base["adj"]) if EXACT else (c["ej"] - base["ej"])
    d_div = 0.1 * (c["dj"] - base["dj"])
    print(f"   best cell {g['label']}:  {'adj_edge' if EXACT else 'edge_J'} {d_edge:+.4f}"
          f"   0.1*div {d_div:+.4f}   net {sc(c)-sc(base):+.4f}")
    ok4 = d_div > -d_edge
    print(f"   -> {'PASS (division gain outweighs the edge cost)' if ok4 else 'FAIL (edge losses dominate)'}")
    if not ok4:
        print("   The 1000x leverage estimate in notes/25 §1 is wrong at these densities.")
        print("   Divisions are not the cheap term; move the effort to edge repair.")

print()
print("=" * 79)
if best and sc(best[1]) > sc(base):
    g, c = best
    print(f"VERDICT: insertion helps. Best {g['label']} at {sc(c):.4f} "
          f"({sc(c)-sc(base):+.4f} vs control),")
    print(f"         {c['added']:,} edges added across {n_ds} datasets, "
          f"division TP {c['tp']} / FP {c['fp']} / FN {c['fn']}.")
    print("         Contamination caveat stands: this is a DELTA on training data, and")
    print("         only the leaderboard can confirm the absolute level.")
else:
    print("VERDICT: insertion does not help at any setting tried. The division term stays")
    print("         at 0.000 and the effort belongs in edge repair. Record it and move on.")
print("=" * 79)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
