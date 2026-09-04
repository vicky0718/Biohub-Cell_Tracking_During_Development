"""Build notebooks/claude_gapum.ipynb — sweep close_gaps' radius and read the score.

notes/57 found pipeline/divisions.py's gates rejecting 88% of real divisions --
constants adopted from a public notebook and never checked. notes/59 applied the same
check to linking: close_gaps' max_um=5.75 rejects up to 23.4% of real two-frame spans,
while cap_edge_length's 14.0 correctly drops only 0.10%.

That 23.4% is an UPPER BOUND, not a measurement: it doubles a single-frame step to
stand in for a two-frame span, which assumes straight-line motion. This run sweeps the
radius on the cached instances and reads the metric instead of inferring it.

close_gaps also carries max_added_frac=0.038 and max_added_abs=1650. If those bind
before the radius does, the radius was never the constraint -- prediction 2 tests it,
which is why inserted-node counts are reported per arm.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_gapum.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Is `close_gaps`'s 5.75 µm radius costing us score?

```
0.901 submitted (rank ~1388/3038)    0.938 = rank 100    0.947 gold
```

`notes/57` found `pipeline/divisions.py`'s geometry gates rejecting **88%** of real
divisions — constants adopted from a public notebook and never checked. `notes/59` applied
the same check to linking and found:

```
128,883 single-frame GT links, all dt=1
euclidean displacement   median 1.82   p90 4.14   p95 5.34   p99 8.38   max 60.76

close_gaps max_um=5.75  (vs 2-frame span)   rejects up to 23.4%
cap_edge_length max_um=14.0                 rejects 0.10%   <- correctly set
```

**But that 23.4% is an upper bound, not a measurement.** `close_gaps` bridges a two-frame
hole, and I approximated a two-frame span as **2× a single-frame step** — which assumes
straight-line motion. Real motion wanders, so the true `t → t+2` displacement is at most
twice a single step and usually less. The geometric argument says *"look here"*; it does not
say the radius costs score.

This run reads the score directly instead of inferring it. No geometry, no assumption: sweep
the radius on the cached instances and see what the metric says.

## What could make it a non-issue

`close_gaps` carries three limits, not one:

```
max_um          = 5.75     the radius under test
max_added_frac  = 0.038    ceiling on inserted nodes, as a fraction
max_added_abs   = 1650     ceiling on inserted nodes, absolute
```

If the two budget caps bind first, the radius was never the constraint and `notes/59`'s
argument is moot regardless of whether the 23.4% is right. **Prediction 2 tests exactly
that**, and it is the reason this run reports inserted-node counts per arm rather than only
scores.

There is also a reason to expect the trade to be tight rather than free. `notes/26` and
`notes/34` both measured gap-closing as the **minor** half of the repair chain — worth
**+0.0013** against `linefit_smooth`'s +0.0086 — so even a well-set radius is playing for
thousandths. And `notes/52` measured the node budget at `ratio = -0.129`: we sit 12.9% under,
so inserted nodes are affordable, but the multiplier still costs 0.1 per unit of ratio.

## The grid

The shipped chain (`gaps(2) -> smooth -> prune(6)`), one ILP solve, radius swept:

```
g5.75   the shipped value, and the anchor
g8.0    between the shipped value and p95
g10.7   notes/59's p95 of 2x single-frame spans
g14.0   cap_edge_length's value, an upper bracket
g20.0   past anything defensible, to find where it turns
```

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `g5.75` equals `claude_divsweep`'s `inc/g2sp6` — total 0.9188, `div_J`
   0.1154, 1,443 forks. Otherwise nothing below is comparable.
2. **A wider radius actually inserts more nodes** (>50 on average). If not, `max_added_frac`
   or `max_added_abs` binds first and the radius was never the constraint.
3. **Some radius beats 5.75 by more than 0.0015** (`notes/44`'s floor). The crux. Failing it
   says `notes/59`'s geometric argument does not survive contact with the metric — which is
   a result, and the reason to run this rather than just widening the constant.
4. **The best arm holds in sign on both embryos.** `notes/59` found linking geometry
   *identical* across embryos (medians 1.72 vs 1.82), unlike divisions, so this one should
   pass if anything real is happening.

*The honest outcome here is as likely to be "the constant was fine" as "widen it". Both are
worth one CPU run, and only one of them is available by reading the geometry.*
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
from pipeline.repair import close_gaps, linefit_smooth, prune_short_tracks
from harness.tracks import read_estimated_nodes
print("worker numpy", np.__version__, flush=True)

def shipped(g, sc, max_um):
    # claude_divsweep's inc/g2sp6 with the gap RADIUS exposed. Everything else is the
    # submitted chain: gaps(max_gap=2) -> smooth -> prune(6). The two insertion caps stay
    # at their shipped values so prediction 2 can tell whether they bind before the radius.
    r = close_gaps(*g, scale=sc, max_um=max_um, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# 5.75 is shipped and is the anchor; 10.7 is notes/59's p95 of doubled single-frame steps;
# 14.0 is cap_edge_length's value; 20.0 is past anything defensible, to find the turn.
POST = [("g" + str(_u), _u) for _u in (5.75, 8.0, 10.7, 14.0, 20.0)]

def apply_post(g, sc, max_um):
    g = shipped(g, sc, max_um)
    return (g[0], g[1], g[2])

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
# The WEIGHT axis is closed -- notes/36 ran 18 settings and nothing beat ratio0.4_2.0
# (closest -0.0009), div_J climbs to 0.1500 only where the score has fallen to 0.8615, and
# cheapening division_weight makes MORE forks and a WORSE div_J. Two solves, no more: a
# control to prove the cache and solver still reproduce, and the incumbent to sweep under.
# One solve. The weight axis is closed three times over (notes/36, 50) and the control
# arm has already served its purpose -- claude_divsweep confirmed div_J 0.0000 there.
ARMS = [("inc", -1.0, 0.4, 2.0, 1.0)]          # ratio0.4_2.0, what we ship
print(f"{{len(POST)}} gap radii on one solve", flush=True)

names = sorted(p.stem[len("cand_"):] for p in CACHE.glob("cand_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()]
print(f"{{len(names)}} cached instances with ground truth", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
LABELS = [p[0] for p in POST]
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

    n_est = read_estimated_nodes(TRAIN / f"{{name}}.geff")
    print(f"    estimated_number_of_nodes {{n_est:,.0f}}", flush=True)
    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for key, max_um in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, max_um)
            ROWS[key].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
            EDGES[key] += int(len(g[2]))
            nf = int((np.bincount(g[2][:, 0], minlength=len(g[0])) >= 2).sum()) if len(g[2]) else 0
            FORKS[key] += nf
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[key].append(a)
            if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
                raise SystemExit(f"{{name}}/{{key}}: buckets do not sum")
            r = ROWS[key][-1]
            # div_J is MICRO-averaged (purescore.summarise), so the per-dataset record has
            # to carry counts, not a ratio -- a per-dataset div_J cannot be averaged into
            # the reported one. notes/47 is the fourth time a ratio was aggregated wrongly.
            PER.setdefault(name, {{}})[key] = {{
                "adj": float(r.get("adj_edge_jaccard", float("nan"))),
                "score": float(r.get("score", float("nan"))),
                "dtp": float(r.get("division_tp", 0.0)),
                "dfp": float(r.get("division_fp", 0.0)),
                "dfn": float(r.get("division_fn", 0.0)),
                # the budget itself -- reported even on a clean failure, because no run
                # so far has measured where we actually sit against N_est.
                "nodes": float(r.get("num_pred_nodes", len(g[0]))),
                "ratio": float(r.get("total_node_ratio", float("nan"))),
                "n_est": float(n_est)}}
    base = PER[name]["g5.75"]
    best_k = max(PER[name], key=lambda k: PER[name][k]["adj"]
                 if PER[name][k]["adj"] == PER[name][k]["adj"] else -9)
    print(f"    none adj {{base['adj']:.4f}} nodes {{base['nodes']:,.0f}} "
          f"ratio {{base['ratio']:+.3f}}  |  best {{best_k}} "
          f"adj {{PER[name][best_k]['adj']:.4f}} "
          f"({{PER[name][best_k]['adj']-base['adj']:+.4f}})   {{time.time()-t0:.0f}}s",
          flush=True)

    out = {{"arms": LABELS, "datasets": [n for n in names if n in PER],
           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "post": [{{"label": p[0], "max_um": p[1]}} for p in POST],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "forks": FORKS, "edges": EDGES, "per_dataset": PER}}
    (WORK / "gapum.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. Grading — does a wider gap radius pay?""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "gapum.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
BASE = "g5.75"                       # the shipped radius
REF_TOTAL, REF_DIVJ, REF_FORKS = 0.9188, 0.1154, 1443   # claude_divsweep's inc/g2sp6
print(f"{len(DS)} datasets | {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# Full-set figures come from purescore.summarise -- div_J micro-averaged, adj_edge
# weight-averaged. Recomputing would substitute unweighted means (notes/47).
def divJ(arm, names=None):
    if names is None:
        return S.get(arm, {}).get("division_jaccard", float("nan"))
    r = rows(arm, names)
    tp, fp, fn = (sum(x[k] for x in r) for k in ("dtp", "dfp", "dfn"))
    return tp / (fp + tp + fn) if (fp + tp + fn) > 0 else float("nan")

def adj(arm, names=None):
    if names is None:
        return S.get(arm, {}).get("adj_edge_jaccard", float("nan"))
    v = [x["adj"] for x in rows(arm, names) if x["adj"] == x["adj"]]
    return sum(v) / len(v) if v else float("nan")

def total(arm, names=None):
    if names is None and arm in S and S[arm].get("score") == S[arm].get("score"):
        return S[arm]["score"]
    return adj(arm, names) + 0.1 * divJ(arm, names)

def mean(arm, key, names=None):
    v = [x[key] for x in rows(arm, names) if x.get(key) == x.get(key)]
    return sum(v) / len(v) if v else float("nan")

print(f"\n{'arm':<10}{'total':>9}{'adj_edge':>10}{'edge_J':>9}{'div_J':>8}"
      f"{'nodes':>10}{'ratio':>9}{'added':>9}{'forks':>8}")
print("-" * 82)
b_nodes = mean(BASE, "nodes")
for a in ARMS:
    print(f"{a:<10}{total(a):>9.4f}{adj(a):>10.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}{divJ(a):>8.4f}"
          f"{mean(a,'nodes'):>10,.0f}{mean(a,'ratio'):>9.3f}"
          f"{mean(a,'nodes')-b_nodes:>+9,.0f}{F.get(a,0):>8,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

print(f"\n1. the shipped radius reproduces claude_divsweep's inc/g2sp6")
ok1 = (abs(total(BASE) - REF_TOTAL) < 0.002 and abs(divJ(BASE) - REF_DIVJ) < 0.010
       and abs(F.get(BASE, 0) - REF_FORKS) <= 20)
print(f"   {BASE}: total {total(BASE):.4f} (want {REF_TOTAL})  div_J {divJ(BASE):.4f} "
      f"(want {REF_DIVJ})  forks {F.get(BASE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The chain has moved. Nothing below is comparable.")

print("\n2. a wider radius actually inserts more nodes (the cap does not bind first)")
adds = [(a, mean(a, "nodes") - b_nodes) for a in ARMS if a != BASE]
for a, d in adds:
    print(f"   {a:<10}{d:>+10,.0f} nodes vs {BASE}")
ok2 = any(d > 50 for _, d in adds)
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   close_gaps also carries max_added_frac=0.038 and max_added_abs=1650.")
    print("   Those bind before the radius does, so max_um was never the constraint and")
    print("   notes/59's geometric argument, upper bound or not, is moot.")

print(f"\n3. some radius beats {BASE} by more than 0.0015 (notes/44's floor)")
cand = [a for a in ARMS if a != BASE]
best = max(cand, key=lambda a: total(a) if total(a) == total(a) else -9) if cand else BASE
gain = total(best) - total(BASE)
ok3 = gain > 0.0015
print(f"   best {best} {total(best):.4f} vs {BASE} {total(BASE):.4f}   {gain:+.4f}"
      f"  ->  {'PASS' if ok3 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best)-adj(BASE):+.4f}"
      f"   0.1*div_J {0.1*(divJ(best)-divJ(BASE)):+.4f}"
      f"   nodes {mean(best,'nodes')-b_nodes:+,.0f}")
if not ok3:
    print("   notes/59 measured 5.75um rejecting up to 23.4% of real two-frame spans, but")
    print("   that was 2x a single-frame step -- an UPPER bound assuming straight-line")
    print("   motion. Read directly, the radius is not costing score, and the geometric")
    print("   argument does not survive contact with the metric.")

print(f"\n4. the best arm holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
if best == BASE or abs(total(best) - total(BASE)) < 1e-9:
    ok4 = False
    print("   NOT GRADED — no arm differs from the shipped radius")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][BASE]["adj"] for n in ns
             if best in PER.get(n, {}) and BASE in PER[n]]
        per[e] = (sum(d) / len(d) if d else float("nan"),
                  divJ(best, ns) - divJ(BASE, ns), len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    t = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok4 = len(t) > 1 and (all(x > 0 for x in t) or all(x < 0 for x in t))
    print(f"   signs agree  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   notes/49: the test set is a THIRD pair of embryos.")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4])}/4 predictions passed")
if ok1 and ok3 and ok4:
    print(f"SUBMITTABLE: close_gaps max_um {BASE[1:]} -> {best[1:]} gains {gain:+.4f}.")
elif ok1 and ok2 and not ok3:
    print("CLOSED: the radius is not the constraint. notes/59's 23.4% was an upper bound")
    print("from a straight-line assumption; measured directly it buys nothing.")
elif ok1 and not ok2:
    print("CLOSED: max_added_frac/max_added_abs bind before the radius does.")
else:
    print("NOT COMPARABLE: reproduction failed; fix that first.")
print("=" * 78)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
