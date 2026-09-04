"""Build notebooks/claude_static.ipynb — do not smooth what is not moving.

notes/59: 8.4% of GT links (10,772 of 128,883) have EXACTLY zero displacement --
frozen frames, because the volumes are crops of one master acquisition, plus
annotations interpolated between labelled frames. Where the truth is static our
detections jitter, a line fit through jitter has a spurious slope, and
linefit_smooth drags the node along it -- and that stage is the MAJOR half of the
repair chain (+0.0086 of +0.0113, notes/26/27).

linefit_smooth's new static_um zeroes the slope below a speed threshold, pulling
toward the window mean, which is the right estimator for a static point. Default 0.0
is an exact no-op, so the anchor arm is a construction check.

Derived from the gapum builder: same cached-graph sweep, different parameter.
Unlike close_gaps this constant has never been tuned on the metric.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_static.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Don't smooth what isn't moving

```
0.937 fork (rank ~320)    0.940 = rank 100    0.947 gold
```

`notes/59` measured every ground-truth link in the training set and found:

```
128,883 single-frame GT links, all dt=1
10,772 of them (8.36%) have EXACTLY 0.0 um displacement
```

**One ground-truth link in twelve connects two nodes at identical coordinates.** `notes/58`
recorded hengck23's two explanations and both are confirmed at that scale: the volumes are
crops of one master acquisition so *"tracks freeze after the same frame indices"*, and some
annotations are *"the result of interpolation — label frame t=1 and t=3 and interpolate for
t=2."*

## Why that breaks the biggest repair we have

`linefit_smooth` is the **major** half of the repair chain — `notes/26`/`27` attribute
**+0.0086 of +0.0113** to it, against gap-closing's +0.0013. It pulls each node toward a
local straight-line fit of its own track.

Where the truth is **static**, our detections still jitter. A line fitted through that
jitter has a **spurious slope**, and smoothing then drags the node *along* it — away from
the fixed position it should sit at. The fit is confidently wrong precisely where the
answer is simplest.

`pipeline/repair.py::linefit_smooth` now takes `static_um`: when the fitted speed falls
below it, the slope is zeroed and the node is pulled toward the **window mean**, which is
the right estimator for a static point. Default `0.0` is an exact no-op.

On a synthetic static track with jitter a line reads as a trend (`tests/test_repair.py`):

```
mean |error| vs the true fixed position
  raw detections        0.3000
  linefit, static off   0.2981     <- the fit buys almost nothing
  linefit, static on    0.2238     <- 25% closer
```

and a genuinely moving track (v = 2.0/frame) is bit-identical either way.

## Why this one is ours

Every other lever tried lately came from reading someone else's notebook, and `notes/60`
recorded the rule that explains why they keep failing: **a parameter already tuned on the
metric is near its optimum on the metric.** `close_gaps`' radius, the fork's division gates
— both refused to move.

`static_um` is not a re-tune of an existing constant. It is a **new term**, derived from a
property of the annotation that we measured ourselves and that appears in no public
notebook. That does not make it right; it makes it untested rather than retested.

## The grid

The shipped chain (`gaps(2) -> smooth -> prune(6)`), one ILP solve, `static_um` swept:

```
s0.0    off -- the shipped chain, and the anchor
s0.3    below the median single-frame step (1.82 um): only near-frozen chains
s0.6
s1.0
s1.8    ~the median step: aggressive, most slow chains treated as static
```

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `s0.0` equals `claude_divsweep`'s `inc/g2sp6` — 0.9188, `div_J` 0.1154,
   1,443 forks. It is an exact no-op by construction, so anything else is a bug.
2. **The fallback fires on a real fraction of nodes.** Node positions differ from `s0.0` on
   more than 1% of nodes at `s0.6`. If almost nothing changes, the fitted speeds are all
   above the threshold and the frozen links are not reaching this stage.
3. **Some `static_um` beats `s0.0` by more than 0.0015** (`notes/44`'s floor). The crux.
4. **The effect is non-monotonic** — it rises then falls. Monotone up means the grid stopped
   too early; monotone down means treating chains as static is simply worse and the
   frozen-GT reasoning does not survive contact with the metric.
5. **The best arm holds in sign on BOTH embryos.**

*`notes/59`'s zero-displacement measurement is solid and unaffected either way. What is on
trial is whether acting on it pays.*
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

def shipped(g, sc, static_um):
    # claude_divsweep's inc/g2sp6 with linefit_smooth's static_um exposed. Everything else
    # is the submitted chain: gaps(max_gap=2) -> smooth -> prune(6).
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2,
                       static_um=static_um)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# 0.0 is the shipped chain and an EXACT no-op, so it anchors the run by construction.
# The median single-frame GT step is 1.82um (notes/59), so 0.3-1.0 catches near-frozen
# chains and 1.8 treats most slow chains as static -- past anything defensible, to find
# where it turns.
# v1 peaked at its TOP value (s1.8, +0.0010), which its own prediction 4 said means the
# grid stopped too early. 1.8um is ~the median single-frame step (notes/59), so beyond it
# most chains are treated as static and the arm approaches "use the window mean, never the
# line". mean_only (1e9) tests that endpoint directly -- notes/26/27 credited smoothing
# with +0.0086 and never compared the line fit against a plain moving average.
POST = [("s" + str(_u), _u) for _u in (0.0, 1.8, 2.5, 3.5, 5.0)] + [("mean_only", 1e9)]

def apply_post(g, sc, static_um):
    g = shipped(g, sc, static_um)
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
print(f"{{len(POST)}} static_um values on one solve", flush=True)

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
        anchor_zyx = None
        for key, static_um in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, static_um)
            if anchor_zyx is None:
                anchor_zyx = np.asarray(g[1], float).copy()
                moved_frac, moved_um = 0.0, 0.0
            else:
                _d = (np.asarray(g[1], float) - anchor_zyx) * np.asarray(sc, float)
                _n = np.linalg.norm(_d, axis=1)
                moved_frac = float((_n > 1e-9).mean()); moved_um = float(_n.mean())
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
                # v1's prediction 2 compared node COUNTS, which static_um cannot
                # change -- it only MOVES nodes -- so the check was blind by construction
                # and wrongly reported "the fallback never fires". Record displacement.
                "moved_frac": moved_frac, "moved_um": moved_um,
                "nodes": float(r.get("num_pred_nodes", len(g[0]))),
                "ratio": float(r.get("total_node_ratio", float("nan"))),
                "n_est": float(n_est)}}
    base = PER[name]["s0.0"]
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
           "post": [{{"label": p[0], "static_um": p[1]}} for p in POST],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "forks": FORKS, "edges": EDGES, "per_dataset": PER}}
    (WORK / "static.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. Grading — does skipping the fit on static chains pay?""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "static.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
BASE = "s0.0"                        # smoothing unchanged: an exact no-op
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

print(f"\n1. s0.0 reproduces inc/g2sp6 — static_um=0 is an exact no-op by construction")
ok1 = (abs(total(BASE) - REF_TOTAL) < 0.002 and abs(divJ(BASE) - REF_DIVJ) < 0.010
       and abs(F.get(BASE, 0) - REF_FORKS) <= 20)
print(f"   {BASE}: total {total(BASE):.4f} (want {REF_TOTAL})  div_J {divJ(BASE):.4f} "
      f"(want {REF_DIVJ})  forks {F.get(BASE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The chain has moved. Nothing below is comparable.")

print("\n2. the fallback actually fires (positions move on >1% of nodes)")
# v1 compared node COUNTS here, which static_um cannot change -- it only MOVES nodes --
# so the check was blind by construction and wrongly reported "never fires" while the
# score column was visibly moving. Measure displacement against the s0.0 anchor instead.
adds = [(a, mean(a, "moved_frac"), mean(a, "moved_um")) for a in ARMS if a != BASE]
for a, f, u in adds:
    print(f"   {a:<11}{f:>8.2%} of nodes moved, mean {u:.3f} um")
ok2 = any(f > 0.01 for _, f, _ in adds)
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   Fitted speeds are all above the threshold, so the frozen chains notes/59")
    print("   measured are not reaching this stage. The 8.4% is real; it is not visible here.")
    print("   ")

print(f"\n3. some static_um beats {BASE} by more than 0.0015 (notes/44's floor)")
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
    print("   notes/59 measured 8.4% of GT links at exactly zero displacement, and the")
    print("   synthetic test shows the fallback helps a static track. Read on the metric,")
    print("   it does not pay: either those links are not where our errors are, or the")
    print("   window mean is no better than the line where it matters.")

print(f"\n4. the best arm holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
if best == BASE or abs(total(best) - total(BASE)) < 1e-9:
    ok4 = False
    print("   NOT GRADED — no arm differs from s0.0")
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
    print(f"SUBMITTABLE: static_um {BASE[1:]} -> {best[1:]} gains {gain:+.4f}.")
elif ok1 and ok2 and not ok3:
    print("CLOSED: acting on the frozen-GT finding does not pay. The 8.4% measurement")
    print("stands; this particular way of exploiting it does not.")
elif ok1 and not ok2:
    print("CLOSED: the fallback never fires -- positions do not move at any threshold.")
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
