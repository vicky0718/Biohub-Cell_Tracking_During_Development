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

Datasets are stratified by the scorer's node budget `estimated_number_of_nodes`, not
hash-sampled and **not** by the annotated GT node count. The first version of this notebook
used the annotated count, which made question 2 unanswerable: annotation rate varies 20x
between the two embryos (`notes/04` §9), so that count sorts by labelling protocol rather
than by size, and the correlation it produced ran with the *opposite sign* to the real one.
`notes/25` §2 records the correction.
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

Datasets are **stratified by the scorer's node budget** (`estimated_number_of_nodes`),
which is both the true size and the quantity the multiplier is computed against. Not by
the *annotated* GT node count: annotation rate varies 20× between the two embryos, so that
count sorts by labelling protocol as much as by size. The first run of this notebook made
exactly that substitution and reported a correlation of the wrong sign (`notes/25` §2).

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
from harness.tracks import Tracks, read_geff, read_estimated_nodes
from harness.purescore import summarise

WEIGHTS = PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth"
model, window_size, downsample = P.load_model(WEIGHTS, DEV)
print(f"model window_size={{window_size}} downsample={{downsample}}", flush=True)

names = sorted({{p.stem for p in TRAIN.glob("*.zarr")}} & {{p.stem for p in TRAIN.glob("*.geff")}})

# STRATIFY BY THE SCORER'S NODE BUDGET, `estimated_number_of_nodes`.
#
# NOT by the annotated GT node count, which is what an earlier version of this notebook
# used and which made its answer to question 2 uninterpretable. Annotation rate varies 20x
# between embryos (`notes/04` §9: 6bba is labelled 1-in-8, 44b6 1-in-167), so the annotated
# count sorts by *labelling protocol* far more strongly than by embryo size, and every
# small-count dataset in the resulting subset came from one embryo and every large-count
# one from the other. `n_total` is both the true size and the quantity the budget
# multiplier is actually computed against, so it is the only correct axis here.
sizes, budgets = {{}}, {{}}
for n in names:
    try:
        g = read_geff(TRAIN / f"{{n}}.geff")
        sizes[n] = len(g.t)
        b = read_estimated_nodes(TRAIN / f"{{n}}.geff")
        if b == b and b > 0:
            budgets[n] = float(b)
    except Exception:
        pass
if len(budgets) < len(sizes) * 0.9:
    raise SystemExit(f"only {{len(budgets)}}/{{len(sizes)}} datasets expose a node budget; "
                     "question 2 cannot be answered on the annotated count -- see above")
ordered = sorted(budgets, key=lambda n: budgets[n])
idx = np.linspace(0, len(ordered) - 1, N_DATASETS).astype(int)
SUBSET = [ordered[i] for i in sorted(set(idx.tolist()))]
print(f"{{len(ordered)}} datasets, budget {{budgets[ordered[0]]:,.0f}} .. "
      f"{{budgets[ordered[-1]]:,.0f}} "
      f"({{budgets[ordered[-1]]/max(1.0,budgets[ordered[0]]):.1f}}x spread)", flush=True)
print(f"stratified subset ({{len(SUBSET)}}) budgets: "
      f"{{[round(budgets[n]) for n in SUBSET]}}", flush=True)
print(f"  their annotated counts, for contrast:   {{[sizes[n] for n in SUBSET]}}", flush=True)

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
    PER[name] = {{"gt_nodes": sizes.get(name), "budget": budgets.get(name),
                 "pred_nodes": int(g.n_nodes),
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
print("=" * 74)
print("QUESTION 2 — does their calibration drift with dataset size?")
print("=" * 74)
print("Size means the scorer's node budget `estimated_number_of_nodes`, NOT the annotated")
print("GT node count. Annotation rate varies 20x between the two embryos, so the annotated")
print("count sorts by labelling protocol rather than by size and answers a different")
print("question. Both are printed; only the budget column is the test.")
print()

def _budget(r):
    b = r.get("budget")
    if b:
        return float(b)
    # Fall back to inverting ratio = (pred - n_total)/n_total. Exact, and it keeps this
    # cell readable against a report produced before `budget` was recorded.
    rat = r.get("total_node_ratio")
    return (r["pred_nodes"] / (1.0 + rat)) if (rat == rat and rat > -1) else float("nan")

rows = [(n, r) for n, r in PER.items()
        if r.get("pred_nodes") and r.get("total_node_ratio") == r.get("total_node_ratio")]
for _, r in rows:
    r["_budget"] = _budget(r)
rows = [(n, r) for n, r in rows if np.isfinite(r["_budget"])]
rows.sort(key=lambda kv: kv[1]["_budget"])

print(f"{'dataset':<26}{'budget':>10}{'annot':>9}{'pred':>10}{'ratio':>9}"
      f"{'edge_J':>9}{'forks':>7}")
print("-" * 80)
for n, r in rows:
    print(f"{n:<26}{r['_budget']:>10,.0f}{(r.get('gt_nodes') or 0):>9,}"
          f"{r['pred_nodes']:>10,}{r['total_node_ratio']:>+9.3f}"
          f"{r.get('edge_jaccard', float('nan')):>9.4f}{r.get('forks', 0):>7,}")

def _corr(xs, ys):
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
        return float("nan")
    return float(np.corrcoef(x[ok], y[ok])[0, 1])

r_pearson = r_annot = float("nan")
by_embryo = {}
if len(rows) >= 6:
    y = [r["total_node_ratio"] for _, r in rows]
    r_pearson = _corr(np.log10([r["_budget"] for _, r in rows]), y)
    r_annot = _corr(np.log10([max(1, r.get("gt_nodes") or 1) for _, r in rows]), y)
    third = max(1, len(rows) // 3)
    lo = float(np.mean(y[:third])); hi = float(np.mean(y[-third:]))

    print(f"\n  corr(log10 BUDGET,    node ratio) = {r_pearson:+.3f}   <- the size test")
    print(f"  corr(log10 annotated, node ratio) = {r_annot:+.3f}   "
          "<- confounded by annotation rate; not the test")
    print(f"  mean ratio, smallest third by budget: {lo:+.4f}")
    print(f"  mean ratio, largest third by budget:  {hi:+.4f}")
    print(f"  drift across the range:               {hi - lo:+.4f}")

    # Within-embryo, because a between-embryo correlation can be produced entirely by the
    # two embryos differing in both size and behaviour, with no size effect inside either.
    for n, r in rows:
        by_embryo.setdefault(n.split("_")[0], []).append(r)
    print("\n  within each embryo (this is what rules out the between-embryo confound):")
    for emb, rs in sorted(by_embryo.items()):
        yy = [r["total_node_ratio"] for r in rs]
        rc = _corr(np.log10([r["_budget"] for r in rs]), yy)
        bl = min(r["_budget"] for r in rs); bh = max(r["_budget"] for r in rs)
        print(f"    {emb:<10} n={len(rs):<3} mean ratio {np.mean(yy):+.4f} "
              f"sd {np.std(yy):.4f}  budget {bl:,.0f}..{bh:,.0f}  corr {rc:+.3f}")
    print("  A flat within-embryo correlation is only evidence of no effect if that")
    print("  embryo's budgets actually SPAN the range where the effect appears — compare")
    print("  the budget ranges above before reading a +0.0 as a refutation.")

    if abs(r_pearson) > 0.4:
        sign = "UNDER" if r_pearson < 0 else "OVER"
        print(f"\n  ** DRIFT PRESENT ** — node ratio moves with true dataset size, and the")
        print(f"     sign says they {sign}-predict on the LARGEST datasets. That is what a")
        print("     global constant does on a 20x range, and it is what a per-dataset")
        print("     budget calibration is for. Confirm the sign against the table before")
        print("     building on it — the correction direction depends on it.")
    else:
        print("\n  ** NO DRIFT ** — their global constants are not obviously miscalibrated")
        print("     against true size. The per-dataset budget angle should be ABANDONED")
        print("     rather than pursued on faith; spend the effort on whichever term")
        print("     question 1 says is short.")

(WORK / "pack_diag_report.json").write_text(json.dumps(
    {"summary": S, "n": D["n"],
     "divisions": {"tp": dtp, "fp": dfp, "fn": dfn, "pred_forks": forks},
     "multiplier": mult,
     "budget_corr": r_pearson, "annotated_corr_confounded": r_annot,
     "within_embryo": {e: {"n": len(rs),
                           "mean_ratio": float(np.mean([r["total_node_ratio"] for r in rs])),
                           "budget_min": float(min(r["_budget"] for r in rs)),
                           "budget_max": float(max(r["_budget"] for r in rs))}
                       for e, rs in by_embryo.items()}}, indent=2, default=float))
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
