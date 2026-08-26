"""Build notebooks/claude_pack_diag.ipynb — find where the 0.046 is, before writing any repair.

`claude_submit_pack` v3 scored **0.867** on the leaderboard with the ILP running. The
cluster reports 0.913-0.916 on the same weights. The pack's own manifest names the missing
piece: it produces the *"ILP candidate graph before notebook-level graph repair"*, and the
public notebook (`...-learned-graph-w-gap-recovery`) does that repair.

Reimplementing that repair is ~700 lines of tuned heuristics. Before spending days on it,
this run answers two questions that decide HOW to spend them:

1. **Which metric term is weak?** edge Jaccard, the node-budget multiplier, or divisions.
   The repair has separate machinery for each, and knowing which one is short says which
   part to build first -- and how much is even available.
2. **Is the per-dataset thesis real?** Their repair is parameterised by GLOBAL constants
   (`GAP_CLOSE_MAX_ADDED_ABS = 1650`, `MOTION_RELINK_MAX_FRAME_NODES = 3200`,
   `GAP_CLOSE_UM = 5.75`) applied identically to every dataset, on a corpus that
   `notes/04` §9 measured varying **20.8x** in node count (3,783 -> 78,644). If their
   node ratio drifts systematically with dataset size, our per-dataset budget regression
   (10.7 % median error, reproduced in four runs) is the differentiator. If it does not,
   that thesis is dead and should be abandoned rather than pursued on faith.

Datasets are stratified by ground-truth node count rather than hash-sampled, precisely so
question 2 is answerable.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_pack_diag.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Where is the 0.046?

| run | LB |
|---|---|
| our classical champion | 0.752 |
| pack, ILP bypassed | 0.843 |
| **pack, ILP running** | **0.867** |
| the cluster, same weights | **0.913–0.916** |

The pack's manifest is explicit that it emits the *"ILP candidate graph **before
notebook-level graph repair**"*. That repair is ~700 lines of tuned heuristics in the
public notebook. **This run decides how to spend the effort of rebuilding it**, and it
does so before any of it is written.

## Two questions

**1. Which term is short?** The metric is
`adjusted_edge_jaccard + 0.1 · division_jaccard`, and `adj_edge_jaccard` is edge Jaccard
times a node-budget multiplier. Those are three separate places to lose 0.046, and the
repair has different machinery for each:

| term | repair machinery |
|---|---|
| edge Jaccard | motion relink, single-parent repair, gap closing |
| budget multiplier | `GAP_CLOSE_MAX_ADDED_*` caps |
| division Jaccard | `add_safe_divisions_postlink` |

Every classical arm we ever ran scored **0.000** on divisions, and that term is worth
**0.1 of the 1.1 maximum** — so if the pack is also near zero there, that alone is a large
share of the gap and is the cheapest thing to fix.

**2. Is the per-dataset thesis real?** Their repair uses global constants — an *absolute*
cap of 1,650 added nodes, a fixed 5.75 µm gap radius — on a corpus varying **20.8×** in
node count. We have a per-dataset budget regression at 10.7 % median error.

The test: does the pack's `total_node_ratio` drift systematically with dataset size? A
correlation means their constants are miscalibrated at the extremes and our regression is
a genuine differentiator. **No correlation means the thesis is dead**, and I would rather
learn that from one run than from three weeks of building on it.

Datasets are **stratified by GT node count**, not hash-sampled, so question 2 is
answerable at all.

*(Scores here are on training data, which `notes/24` §2 established is contaminated for
these weights — their splits were never published. Absolute values are inflated. The
**decomposition across terms** and the **correlation with size** are what this run is for,
and neither depends on the absolute level.)*
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

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")

# The official scorer: the pack's model predicts divisions, and purescore's division term
# is exact only for fork-free graphs (notes/24 §1).
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

N_DATASETS = 24          # stratified by GT node count; ~100 s each with ILP
DET_THRESHOLD = 0.985    # the PUBLIC NOTEBOOK's value (their script default is 0.99)
print(f"config: n={N_DATASETS}, det_threshold={DET_THRESHOLD} (public notebook's value)")
""")

md("""## 1. Run the pack with ILP, decomposed

Everything in a subprocess: the pack's wheels upgrade numpy under a process that has
already imported the old one (`notes/24` §1).
""")

code(r"""
WORKER = WORK / "run_diag.py"
WORKER.write_text(f'''
import json, os, sys, time, types
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
PACK = Path({str(PACK)!r}); REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
WORK = Path({str(WORK)!r})
N_DATASETS = {N_DATASETS}
DET_THRESHOLD = {DET_THRESHOLD}

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
from harness.tracks import Tracks, read_geff
from harness.purescore import summarise

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)
print(f"model window_size={{window_size}} downsample={{downsample}}", flush=True)

names = sorted({{p.stem for p in TRAIN.glob("*.zarr")}} & {{p.stem for p in TRAIN.glob("*.geff")}})

# STRATIFY BY GT NODE COUNT. The whole point of question 2 is whether their global
# constants misbehave at the density extremes, so a hash-sample that happens to avoid the
# extremes would answer it by accident and wrongly.
sizes = {{}}
for n in names:
    try:
        g = read_geff(TRAIN / f"{{n}}.geff")
        sizes[n] = len(g.t)
    except Exception:
        pass
ordered = sorted(sizes, key=lambda n: sizes[n])
idx = np.linspace(0, len(ordered) - 1, N_DATASETS).astype(int)
SUBSET = [ordered[i] for i in sorted(set(idx.tolist()))]
print(f"{{len(ordered)}} datasets, GT nodes {{sizes[ordered[0]]:,}} .. {{sizes[ordered[-1]]:,}} "
      f"({{sizes[ordered[-1]]/max(1,sizes[ordered[0]]):.1f}}x spread)", flush=True)
print(f"stratified subset ({{len(SUBSET)}}): "
      f"{{[sizes[n] for n in SUBSET]}}", flush=True)

ILP_EDGE_W, ILP_APP_W, ILP_DIS_W, ILP_DIV_W = -1.0, 0.1, 0.1, 1.0
cfg = P.PredictConfig(det_threshold=DET_THRESHOLD, use_ilp=True,
                      ilp_edge_weight=ILP_EDGE_W, ilp_appearance_weight=ILP_APP_W,
                      ilp_disappearance_weight=ILP_DIS_W, ilp_division_weight=ILP_DIV_W)

h = Harness(data_dir=TRAIN, cache_dir=None)
PER = {{}}

def predict(name, data_dir):
    t0 = time.time()
    coords, edges = P.predict_video(model, Path(data_dir) / f"{{name}}.zarr", DEV, cfg=cfg,
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
    g = Tracks.from_tracksdata(g_td)
    PER[name] = {{"gt_nodes": sizes.get(name), "pred_nodes": int(g.n_nodes),
                 "pred_edges": int(g.n_edges), "cand_edges": int(n_cand),
                 "forks": int(g.n_divisions), "sec": round(time.time() - t0, 1)}}
    print(f"  {{name:<24}} gt={{sizes.get(name):>7,}} pred={{g.n_nodes:>7,}} "
          f"edges={{g.n_edges:>7,}} forks={{g.n_divisions:>5,}} "
          f"ILP {{g.n_edges}}/{{n_cand}} ({{time.time()-t0:.0f}}s)", flush=True)
    return g

res = h.evaluate(predict, arm="pack_ilp", names=SUBSET, verbose=False)
s = dict(res.summary)
for name, row in res.rows.items():
    PER[name].update({{k: (float(v) if isinstance(v, (int, float)) else v)
                      for k, v in row.items()}})

out = {{"summary": {{k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in s.items()}},
       "per_dataset": PER, "n": len(SUBSET), "det_threshold": DET_THRESHOLD,
       "hours": (time.time() - time.time()) / 3600}}
(WORK / "pack_diag.json").write_text(json.dumps(out, indent=2, default=float))
print("\\nSUMMARY:", json.dumps(out["summary"], indent=2, default=float), flush=True)
print("worker done", flush=True)
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

md("""## 2. The two answers""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "pack_diag.json").read_text())
S, PER = D["summary"], D["per_dataset"]

print("=" * 66)
print("QUESTION 1 — which term is short?")
print("=" * 66)
for k in ("score", "adj_edge_jaccard", "edge_jaccard", "division_jaccard", "node_recall"):
    v = S.get(k, float("nan"))
    print(f"  {k:<22} {v:.4f}" if v == v else f"  {k:<22} nan")
mult = (S["adj_edge_jaccard"] / S["edge_jaccard"]) if S.get("edge_jaccard") else float("nan")
print(f"  {'budget multiplier':<22} {mult:.4f}   (ceiling 1.1 when under budget)")

dtp = sum(r.get("division_tp", 0) for r in PER.values())
dfp = sum(r.get("division_fp", 0) for r in PER.values())
dfn = sum(r.get("division_fn", 0) for r in PER.values())
forks = sum(r.get("forks", 0) for r in PER.values())
print(f"\n  divisions: TP={dtp:,}  FP={dfp:,}  FN={dfn:,}   (predicted forks {forks:,})")
div_contrib = 0.1 * (S.get("division_jaccard") or 0.0)
print(f"  division term contributes {div_contrib:.4f} of the score "
      f"(max 0.1000, so {0.1 - div_contrib:.4f} is still on the table)")

print(f"\n  Reading it: the score is adj_edge_jaccard + 0.1*division_jaccard.")
print(f"    adj_edge_jaccard = {S.get('adj_edge_jaccard', float('nan')):.4f}")
print(f"    0.1 * division_j = {div_contrib:.4f}")
print(f"    total            = {S.get('score', float('nan')):.4f}")

print()
print("=" * 66)
print("QUESTION 2 — does their calibration drift with dataset size?")
print("=" * 66)
rows = [(n, r) for n, r in PER.items()
        if r.get("gt_nodes") and r.get("total_node_ratio") == r.get("total_node_ratio")]
rows.sort(key=lambda kv: kv[1]["gt_nodes"])
print(f"{'dataset':<26}{'GT nodes':>10}{'pred':>10}{'ratio':>9}{'edge_J':>9}{'forks':>8}")
print("-" * 72)
for n, r in rows:
    print(f"{n:<26}{r['gt_nodes']:>10,}{r['pred_nodes']:>10,}"
          f"{r['total_node_ratio']:>+9.3f}{r.get('edge_jaccard', float('nan')):>9.4f}"
          f"{r.get('forks', 0):>8,}")

if len(rows) >= 6:
    x = np.log10([r["gt_nodes"] for _, r in rows])
    y = np.array([r["total_node_ratio"] for _, r in rows], float)
    ok = np.isfinite(x) & np.isfinite(y)
    r_pearson = float(np.corrcoef(x[ok], y[ok])[0, 1])
    lo = y[ok][:len(y[ok]) // 3].mean()
    hi = y[ok][-(len(y[ok]) // 3):].mean()
    print(f"\n  corr(log10 GT nodes, node ratio) = {r_pearson:+.3f}")
    print(f"  mean ratio, smallest third: {lo:+.4f}")
    print(f"  mean ratio, largest third:  {hi:+.4f}")
    print(f"  drift across the range:     {hi - lo:+.4f}")
    if abs(r_pearson) > 0.4:
        print("\n  ** THESIS SUPPORTED ** — their node ratio moves systematically with")
        print("     dataset size, which is what a global constant on a 20x density range")
        print("     would do. Our per-dataset budget regression targets exactly this.")
    else:
        print("\n  ** THESIS NOT SUPPORTED ** — no systematic drift with size. Their global")
        print("     constants are not obviously miscalibrated, and the per-dataset budget")
        print("     angle should be ABANDONED rather than pursued on faith. Spend the")
        print("     remaining effort on whichever term question 1 says is short.")

(WORK / "pack_diag_report.json").write_text(json.dumps(
    {"summary": S, "n": D["n"],
     "divisions": {"tp": dtp, "fp": dfp, "fn": dfn, "pred_forks": forks},
     "multiplier": mult,
     "size_corr": (r_pearson if len(rows) >= 6 else None)}, indent=2, default=float))
print("\nwrote pack_diag_report.json")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try: ast.parse("\n".join("pass" if l.strip().startswith("!") else l for l in src.splitlines()))
        except SyntaxError as e: raise SystemExit(f"cell {i}: {e}\n{src}")
