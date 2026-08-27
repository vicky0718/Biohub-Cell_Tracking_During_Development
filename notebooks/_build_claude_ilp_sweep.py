"""Build notebooks/claude_ilp_sweep.ipynb — sweep the ILP's own weights, not its output.

Two independent relink designs have now failed from opposite ends (`notes/30`): mine was
loose enough to fire and made `fn_mislink` monotonically worse; the user's was strict
enough to be safe and made **zero swaps** across the whole test set. Both were *overrides*
of the global solve.

Nobody has yet asked whether the global solve is well parameterised. It runs on the pack's
inherited defaults -- `edge_weight=-1.0, appearance=0.1, disappearance=0.1, division=1.0`
-- which were never measured here. Two of them look questionable on the record we already
have:

* `notes/03` §3, from the same lab's own zebrafish Ultrack config **[PRIMARY]**:
  `appear_weight = -0.002`, `disappear_weight = -0.01` -- a deliberate **5x asymmetry**,
  "discouraging track termination more than initiation". The pack's `0.1 / 0.1` is
  symmetric and does not reflect that.
* `division_weight = 1.0` is a penalty paid to create forks, on a term we score **0.000 of
  0.100** on. `notes/25` priced *geometric* fork insertion at 1 TP per 2,223 guesses, but
  the ILP's forks are model-driven -- a different precision regime that has never been
  measured separately.

`claude_relink_sweep` wrote `cand_*.npz` for all 24 datasets: coords, the post-ILP graph,
and the model's candidate edges **with probabilities**. That is the ILP's own input, so
every arm here is a **re-solve of a cached instance** -- no model, no GPU, no prediction
pass. Attaching that kernel turns a ~25-minute inference pass into seconds of solver time.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_ilp_sweep.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Sweep the ILP's weights — the one parameter set nobody has examined

```
0.752  classical champion
0.867  pack + ILP
0.880  + gap-close + linefit-smooth        <- best scored
0.913–0.916  the cluster, same weights     <- 0.033–0.036 away
```

Two relink designs, built independently with opposite risk postures, have both failed
(`notes/30`): loose enough to fire ⇒ net harmful; strict enough to be safe ⇒ **zero swaps
in the entire test set**. Both were *overrides* of the global solve.

**This run asks whether the solve itself is well parameterised.** Its weights are the
pack's inherited defaults and have never been measured here:

| weight | default | why it is suspect |
|---|---|---|
| `division_weight` | 1.0 | a penalty to create forks, on a term we score **0.000 of 0.100** |
| `appearance` / `disappearance` | 0.1 / 0.1 | the same lab's own zebrafish config uses **−0.002 / −0.01**, a deliberate 5× asymmetry (`notes/03` §3, primary source) |
| `edge_weight` | −1.0 | sets how far the learned probability outweighs those penalties |

## Why this is nearly free

`claude_relink_sweep` cached `cand_*.npz` for all 24 datasets — coords, post-ILP graph, and
candidate edges **with probabilities**. That is exactly what the ILP consumes, so each arm
is a **re-solve of a cached instance**: no model, no GPU, no prediction pass.

## Pre-registered predictions

1. **The control reproduces 0.8806 ± 0.0005**, re-solving from cache with the default
   weights. This is the load-bearing check — it proves the cached candidates round-trip
   through the solver to the same graph the last four runs scored. If it misses, the cache
   is not a faithful ILP input and nothing else in the run is readable.
2. **Lowering `division_weight` raises the fork count.** A mechanical check on whether the
   knob does what its name says, before any score is read from it.
3. **More forks does NOT raise `division_jaccard` above 0.01.** `notes/25` measured the
   ILP's own forks as *unevaluable* — 37 emitted, 0 TP, 0 FP — and geometric insertion at
   1 TP per 2,223. If model-driven forks behave differently this fails, and that would be
   the most valuable result here; if it passes, the division term is confirmed dead from a
   third independent direction and should never be revisited.
4. **The asymmetric appear/disappear arm beats the symmetric default.** Straight from
   `notes/03` §3's primary-source config. If it fails, the pack's symmetric choice is
   vindicated and the lab's own setting does not transfer to this metric.

*Training data, contaminated for these weights (`notes/24` §2). `notes/28` measured a
1.13× transfer ratio on the one submission that tested it — a direction, not a coefficient.*
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
# claude_relink_sweep's output: coords + post-ILP graph + candidate edges WITH probabilities.
CACHE = find_dir(lambda p: any(p.glob("cand_*.npz")), ["/kaggle/input"])
for lbl, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP), ("cand cache", CACHE)):
    print(f"  {lbl:<13} {v}")
if None in (PACK, REPO, COMP, CACHE):
    raise SystemExit("missing mount — CACHE needs claude-relink-sweep attached as a kernel source")
TRAIN = COMP / "train"
n_cached = len(list(CACHE.glob("cand_*.npz")))
print(f"  cached instances  {n_cached}")
if n_cached == 0:
    raise SystemExit("no cand_*.npz found")

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
           "import numpy, zarr, tracksdata; import tracksdata.solvers; "
           "print('numpy', numpy.__version__)")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")
print(f"no GPU or model needed — setup took {time.time()-T_START:.0f}s")
""")

md("""## 1. Re-solve each cached instance under every weight setting

The cache holds `cand` as `(K, 3)` — source, target, probability — already remapped into
the graph's index space by `claude_relink_sweep`. Rebuilding a tracksdata graph from it and
re-solving is the whole run.
""")

code(r"""
WORKER = WORK / "run_ilp.py"
WORKER.write_text(f'''
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
CACHE = Path({str(CACHE)!r}); WORK = Path({str(WORK)!r})
T0 = time.time()

sys.path.insert(0, str(REPO))
import polars as pl
import tracksdata as td
from harness import Harness
from harness.tracks import Tracks, read_geff, read_scale
from harness.purescore import summarise
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy
from pipeline.repair import close_gaps, linefit_smooth
print("worker numpy", np.__version__, flush=True)

def repair_chain(g, sc):
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038, max_added_abs=1650)
    return linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

def build_td(t, zyx, cand):
    # Rebuild the ILP's INPUT graph: every cached node, every candidate edge with its
    # probability. A docstring here would terminate the outer f-string that writes this
    # file -- the same trap that has bitten four notebooks in this project already.
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    ids = g.bulk_add_nodes([{{"t": int(tt), "z": float(z), "y": float(y), "x": float(x)}}
                            for tt, (z, y, x) in zip(t, zyx)])
    if len(cand):
        g.bulk_add_edges([{{"source_id": ids[int(s)], "target_id": ids[int(d)],
                           "edge_prob": float(p)}} for s, d, p in cand[:, :3]])
    return g, np.asarray(ids, np.int64)

def solve(g_td, ew, ap, dis, dv):
    if g_td.num_edges() == 0:
        return g_td
    solver = td.solvers.ILPSolver(
        edge_weight=ew * td.EdgeAttr("edge_prob"),
        appearance_weight=ap, disappearance_weight=dis, division_weight=dv)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return solver.solve(g_td)

# ---- the sweep -------------------------------------------------------------------
# (label, edge_weight, appearance, disappearance, division)
DEF = (-1.0, 0.1, 0.1, 1.0)                       # the pack's inherited defaults
ARMS = [("control", *DEF)]
# Prediction 2/3: does cheapening forks make more of them, and does that ever score?
for dv in (0.5, 0.2, 0.0, -0.5):
    ARMS.append((f"div{{dv}}", DEF[0], DEF[1], DEF[2], dv))
# Prediction 4: notes/03 §3's primary-source asymmetry, scaled to this solver's magnitudes.
for ap, dis in ((0.02, 0.10), (0.05, 0.25), (0.10, 0.50)):
    ARMS.append((f"asym{{ap}}_{{dis}}", DEF[0], ap, dis, DEF[3]))
# Symmetric magnitude, as the control for the asymmetry arms -- otherwise a gain from
# "asymmetric" could really be a gain from "larger", which is a different finding.
for s in (0.02, 0.25):
    ARMS.append((f"sym{{s}}", DEF[0], s, s, DEF[3]))
# How far the learned probability outweighs the appear/disappear penalties.
for ew in (-0.5, -2.0, -4.0):
    ARMS.append((f"edge{{ew}}", ew, DEF[1], DEF[2], DEF[3]))
print(f"{{len(ARMS)}} arms, each also scored with the notes/28 repair chain", flush=True)

names = sorted(p.stem[len("cand_"):] for p in CACHE.glob("cand_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()]
print(f"{{len(names)}} cached instances with ground truth", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
LABELS = [a[0] for a in ARMS] + [f"{{a[0]}}+repair" for a in ARMS]
ROWS = {{l: [] for l in LABELS}}
ANAT = {{l: [] for l in LABELS}}
FORKS = {{l: 0 for l in LABELS}}
EDGES = {{l: 0 for l in LABELS}}
PER = {{}}

for name in names:
    t0 = time.time()
    z = np.load(CACHE / f"cand_{{name}}.npz")
    t, zyx, cand = z["t"], z["zyx"], z["cand"]
    sc = read_scale(TRAIN / f"{{name}}.zarr")
    gt = read_geff(TRAIN / f"{{name}}.geff")
    base_td, _ = build_td(t, zyx, cand)
    print(f"\\n{{name}}  nodes={{len(t):,}} cand={{len(cand):,}}", flush=True)

    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for with_repair in (False, True):
            key = f"{{lbl}}+repair" if with_repair else lbl
            g = (tr.t, tr.zyx, tr.edges)
            if with_repair:
                g = repair_chain(g, sc)
            ROWS[key].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            EDGES[key] += int(len(g[2]))
            nf = int((np.bincount(g[2][:, 0], minlength=len(g[0])) >= 2).sum()) if len(g[2]) else 0
            FORKS[key] += nf
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[key].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(f"{{name}}/{{key}}: buckets do not sum")
            PER.setdefault(name, {{}})[key] = float(
                ROWS[key][-1].get("adj_edge_jaccard", float("nan")))
    best = max(PER[name], key=lambda k: PER[name][k] if PER[name][k] == PER[name][k] else -9)
    print(f"    control {{PER[name]['control']:.4f}}  best {{best}} {{PER[name][best]:.4f}} "
          f"({{PER[name][best]-PER[name]['control']:+.4f}})  {{time.time()-t0:.0f}}s", flush=True)

    out = {{"arms": LABELS, "datasets": [n for n in names if n in PER],
           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "forks": FORKS, "edges": EDGES, "per_dataset": PER}}
    (WORK / "ilp_sweep.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "ilp_sweep.json").read_text())
S, A, F, E = D["summary"], D["anatomy"], D["forks"], D["edges"]
ARMS, DS, GRID = D["arms"], D["datasets"], {g["label"]: g for g in D["grid"]}
base, abase = S["control"], A["control"]
EXACT = base["score"] == base["score"]
key = "score" if EXACT else "edge_jaccard"
print(f"{len(DS)} datasets, {len(ARMS)} arms")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

print(f"{'arm':<20}{'score':>9}{'delta':>9}{'edge_J':>9}{'div_J':>8}"
      f"{'forks':>8}{'d_mislink':>11}{'d_edges':>10}")
print("-" * 84)
for a in ARMS:
    if a not in S:
        continue
    s, an = S[a], A[a]
    dj = s.get("division_jaccard")
    print(f"{a:<20}{s[key]:>9.4f}{s[key]-base[key]:>+9.4f}{s['edge_jaccard']:>9.4f}"
          f"{(dj if dj == dj else 0):>8.4f}{F[a]:>8,}"
          f"{an['fn_mislink']-abase['fn_mislink']:>+11}{E[a]-E['control']:>+10,}")

print()
print("=" * 84)
print("PREDICTION GRADING")
print("=" * 84)

# 1 ---------------------------------------------------------------------------------
print("\n1. the control reproduces 0.8806 +- 0.0005 by re-solving from cache")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
elif len(DS) != 24:
    print(f"   NOT GRADED — {len(DS)} datasets completed, not 24.")
else:
    ok = abs(base["score"] - 0.8806) <= 0.0005
    print(f"   control = {base['score']:.4f}  ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("   The cached candidates do NOT round-trip through the solver to the graph")
        print("   the last four runs scored. Nothing below is readable — the cache is not")
        print("   a faithful ILP input, and that is the thing to diagnose first.")

# `GRID` is keyed by the BASE arm labels only, while `ARMS` also carries every `+repair`
# variant — so anything reading weights out of GRID must filter to the base arms first.
BASE = [a for a in ARMS if a in GRID]

# 2 ---------------------------------------------------------------------------------
print("\n2. lowering division_weight raises the fork count")
dv = [(GRID[a]["division"], a) for a in BASE if a.startswith("div")]
dv = sorted(dv + [(GRID["control"]["division"], "control")], reverse=True)
if len(dv) >= 3:
    print("   " + "  ".join(f"dv={d:g}:{F[a]:,}" for d, a in dv))
    counts = [F[a] for _, a in dv]
    ok2 = counts == sorted(counts)
    print(f"   ->  {'PASS — the knob does what its name says' if ok2 else 'FAIL — forks do not track the weight'}")

# 3 ---------------------------------------------------------------------------------
print("\n3. more forks does NOT raise division_jaccard above 0.01")
best_dj, best_a = 0.0, None
for a in ARMS:
    v = S[a].get("division_jaccard", 0.0)
    if v == v and v > best_dj:
        best_dj, best_a = v, a
print(f"   best division_jaccard anywhere: {best_dj:.4f} ({best_a})")
ok3 = best_dj <= 0.01
print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
if ok3:
    print("   The division term is now dead from a THIRD independent direction: the ILP's")
    print("   own model-driven forks score no better than notes/25's geometric ones.")
    print("   Do not revisit it.")
else:
    print("   Model-driven forks DO score where geometric ones did not. That is the most")
    print("   valuable result in this run and reopens the 0.1 division term — size it")
    print("   against the edge cost before acting.")

# 4 ---------------------------------------------------------------------------------
print("\n4. the asymmetric appear/disappear arm beats the symmetric default")
asym = [a for a in BASE if a.startswith("asym")]
sym = [a for a in BASE if a.startswith("sym")]
if asym:
    best_as = max(asym, key=lambda a: S[a][key])
    print(f"   best asymmetric: {best_as} {S[best_as][key]-base[key]:+.4f} "
          f"(appear {GRID[best_as]['appear']}, disappear {GRID[best_as]['disappear']})")
    for a in sym:
        print(f"   symmetric control {a}: {S[a][key]-base[key]:+.4f}")
    ok4 = S[best_as][key] > base[key] + 1e-6
    print(f"   vs default  ->  {'PASS' if ok4 else 'FAIL'}")
    if ok4 and sym:
        best_sym = max(sym, key=lambda a: S[a][key])
        margin = S[best_as][key] - S[best_sym][key]
        print(f"   vs best symmetric ({best_sym}): {margin:+.4f}")
        if margin <= 0:
            print("   !! The gain is from MAGNITUDE, not asymmetry — a symmetric arm at the")
            print("      same scale does as well. Report it as that, not as the lab's config")
            print("      transferring.")
    if not ok4:
        print("   The lab's own zebrafish asymmetry does not transfer to this metric, and")
        print("   the pack's symmetric default is vindicated.")

print()
print("=" * 84)
best = max((a for a in ARMS if a != "control"),
           key=lambda a: (S[a][key] if a in S and S[a][key] == S[a][key] else float("-inf")))
print(f"BEST ARM: {best} at {S[best][key]:.4f} ({S[best][key]-base[key]:+.4f})")
print(f"  notes/28 chain alone (control+repair): {S['control+repair'][key]-base[key]:+.4f}"
      "   <- the +0.0115 that scored 0.880")
if best.endswith("+repair"):
    stem = best[:-len("+repair")]
    print(f"  the ILP change alone ({stem}): {S[stem][key]-base[key]:+.4f}")
    print(f"  so the ILP change adds {S[best][key]-S['control+repair'][key]:+.4f} on top of repair")
print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
      f"undetected {A[best]['fn_detect']:,}")
print("=" * 84)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
