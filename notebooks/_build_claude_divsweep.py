"""Build notebooks/claude_divsweep.ipynb — where half the division term went.

notes/36 measured div_J 0.1154 at ratio0.4_2.0 with close_gaps(max_gap=1) +
linefit_smooth, on 24 datasets. notes/42 measured 0.0645 at the SAME ILP weights on
12 datasets, after the config audit moved max_gap to 2 and added
prune_short_tracks(6). Worth +0.0051 if it is real and recoverable.

But the two numbers are on different dataset samples, and notes/44 measured that
sample as easy by +0.0116 — so the drop may not exist at all. Every arm here runs on
the same cached instances so that question gets a clean answer either way.

Derived from the ilp_sweep3 builder, which already re-solves cached candidate graphs
with no GPU. notes/36 closed the weight axis, so the weights are held at two points
and the POST-PROCESSING chain becomes the grid, with div_J promoted from a reported
column to the quantity being optimised.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_divsweep.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Where did half the division term go? Post-processing, with `div_J` as the read-out

```
0.901 submitted    0.926 bronze    0.944 gold
score = adj_edge_jaccard + 0.1 * div_J   (max 1.1)
```

Two numbers, from this project's own record, that cannot both describe the chain we ship:

```
notes/36   div_J 0.1154   ratio0.4_2.0 + close_gaps(max_gap=1) + linefit_smooth   n=24
notes/42   div_J 0.0645   the "best config" chain that became the submission       n=12
```

**Same ILP weights.** The difference is everything downstream of the solve: `max_gap` went
1 -> 2, and `prune_short_tracks(min_frames=6)` was added. `notes/42` §3 caught the drop and
named the cause correctly — *"the config audit optimised the total and gave half of it
back, which nobody noticed because the total went up"* — and then the direction was
dropped for divisions-by-classifier, which `notes/43` closed on 151 training events.

At stake: `div_J` 0.0645 -> 0.1154 is **+0.0051** on the score. `notes/43`'s ceiling for
driving chargeable false forks to zero is **+0.016**. The gap to bronze is 0.025.

## The confound this run exists to kill

**The two numbers are on different dataset samples**, and `notes/44` measured that exact
sample as biased: *"the 12 were an easy subset by +0.0116."* Comparing 0.1154 (n=24)
against 0.0645 (n=12) is `notes/47`'s error shape for the fifth time — a ratio read across
two populations. **The drop may not exist.**

So every arm here runs on **the same cached instances**, and prediction 2 is written so that
"the drop was a sampling artifact" is a clean, cheap close rather than a disappointment.

## Why post-processing and not the weights

The weight axis is **closed, three times**. `claude_ilp_sweep3` ran 18 settings and nothing
beat `ratio0.4_2.0` (closest −0.0009). `notes/36` §: `div_J` keeps climbing to 0.1500 at
`r5_1.6` while the score falls to 0.8615, and cheapening `division_weight` makes *more*
forks and a *worse* `div_J`. Forks are already precision-selected by the termination
penalty; the solver is not the problem.

`close_gaps` and `prune_short_tracks` are. Both were tuned by reading the **total**, and both
have a documented mechanism for destroying a correctly-found fork:

* `close_gaps` **inserts** edges. `pipeline/divisions.py`'s FP rules: a sister with
  in-degree > 0 makes the fork `malformed`, *"an automatic false positive — the worst
  possible trade."* At `max_gap=2` it bridges further and has more chances to attach a
  second parent.
* `prune_short_tracks` **deletes** components. It carries
  `keep_division_components=True`, which should protect forks — *should*. That flag has
  never been measured with `div_J` as the read-out, only asserted.
* `linefit_smooth` **moves nodes**, and the division topology test is geometric.

The cache is `claude_relink_sweep`'s, built at `det_threshold=0.985`. We ship 0.975;
`notes/44` measured the whole 0.965–0.99 interval moving the score by 0.0001, so the two are
inside one plateau. (`notes/49`: 0.999 is *not*, which is why it is not in this grid.)

## Design: one solve, many free post-chains

`notes/40`'s split, applied one layer down. The ILP solve is the expensive step, so it runs
**twice** — pack defaults and the incumbent — and eight post-processing chains are graded on
each solved graph for free.

```
raw        nothing                            g2s        gaps(2) + smooth
g1         gaps(max_gap=1)                    g2p6       gaps(2) + prune(6)      <- no smooth
g1s        gaps(1) + smooth       <- notes/36 g2sp6      gaps(2) + smooth + prune(6)  <- SHIPPED
g1sp6      gaps(1) + smooth + prune(6)        g2sp6_nk   same, keep_division_components=OFF
```

Each stage is isolated by a pair that differs in exactly one thing, so a `div_J` drop is
attributable rather than merely visible.

## Pre-registered predictions

Graded **per embryo**, both means printed, after `notes/49`.

1. **Reproduction.** `ctl` gives `div_J` ≈ 0.000 (the pack's dead division term) and
   `inc/g1s` lands within 0.010 of `notes/36`'s **0.1154**. If either fails, the cache or
   the solver has moved and nothing below is comparable to the record.
2. **The drop is real and post-processing causes it.** `inc/g2sp6` (what we ship) scores
   `div_J` at least **0.020** below `inc/g1s` *on identical datasets*. **If it does not, the
   0.1154 → 0.0645 gap was a sampling artifact, this direction closes in one cheap run, and
   that is a result** — it removes the last unexamined item on `notes/44`'s shortlist.
3. **One stage dominates.** The largest single-stage drop is more than half the total drop,
   naming a culprit instead of diffusing blame across three.
4. **The recovery is not free.** Arms that restore `div_J` lose `adj_edge_jaccard`, per
   `notes/36`'s trade. The interesting outcome is an arm that gains `0.1·div_J` **more than**
   it loses on edges — that is the only thing here worth a submission slot.
5. **It holds on both embryos.** The best arm beats `g2sp6` in sign on `44b6` *and* `6bba`.
   `notes/49`: a pooled win across crops of two embryos is not evidence about a third.

*If 2 fails, the division term is finished and the remaining gap is entirely the edge term,
exactly as `notes/36` §concluded. If 2 passes and 4 fails, we know the cost and can price
it. Only 2-and-4 together are submittable.*
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
print("worker numpy", np.__version__, flush=True)

def _gaps(g, sc, mg):
    return close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                      max_added_abs=1650, max_gap=mg)

def _smooth(g, sc):
    return linefit_smooth(*g, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

def _prune(g, sc, keep):
    return prune_short_tracks(*g, min_frames=6, keep_division_components=keep)

# (label, stages). Each pair differs from another by exactly ONE stage, so a div_J drop
# is attributable. g1s is notes/36's chain (div_J 0.1154); g2sp6 is what we ship.
POST = [
    ("raw",      []),
    ("g1",       [lambda g, s: _gaps(g, s, 1)]),
    ("g1s",      [lambda g, s: _gaps(g, s, 1), _smooth]),
    ("g2s",      [lambda g, s: _gaps(g, s, 2), _smooth]),
    ("g1sp6",    [lambda g, s: _gaps(g, s, 1), _smooth,
                  lambda g, s: _prune(g, s, True)]),
    ("g2p6",     [lambda g, s: _gaps(g, s, 2),
                  lambda g, s: _prune(g, s, True)]),
    ("g2sp6",    [lambda g, s: _gaps(g, s, 2), _smooth,
                  lambda g, s: _prune(g, s, True)]),
    ("g2sp6_nk", [lambda g, s: _gaps(g, s, 2), _smooth,
                  lambda g, s: _prune(g, s, False)]),
]

def apply_post(g, sc, stages):
    for fn in stages:
        g = fn(g, sc)
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
ARMS = [("ctl", -1.0, 0.1, 0.1, 1.0),          # pack defaults: div_J 0.0000 in notes/36
        ("inc", -1.0, 0.4, 2.0, 1.0)]          # ratio0.4_2.0, what we ship
print(f"{{len(ARMS)}} solves x {{len(POST)}} post-chains = {{len(ARMS)*len(POST)}} arms",
      flush=True)

names = sorted(p.stem[len("cand_"):] for p in CACHE.glob("cand_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()]
print(f"{{len(names)}} cached instances with ground truth", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
LABELS = [f"{{a[0]}}/{{p[0]}}" for a in ARMS for p in POST]
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
        for post_lbl, stages in POST:
            key = f"{{lbl}}/{{post_lbl}}"
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, stages)
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
                "dfn": float(r.get("division_fn", 0.0))}}
    ref, shipped = PER[name]["inc/g1s"], PER[name]["inc/g2sp6"]
    print(f"    inc/g1s adj {{ref['adj']:.4f}} dTP {{ref['dtp']:.0f}} dFP {{ref['dfp']:.0f}}"
          f"   inc/g2sp6 adj {{shipped['adj']:.4f}} dTP {{shipped['dtp']:.0f}} "
          f"dFP {{shipped['dfp']:.0f}}   {{time.time()-t0:.0f}}s", flush=True)

    out = {{"arms": LABELS, "datasets": [n for n in names if n in PER],
           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "post": [p[0] for p in POST],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "forks": FORKS, "edges": EDGES, "per_dataset": PER}}
    (WORK / "divsweep.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. Grading — `div_J` against the chain we ship, per embryo""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "divsweep.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
POST = D["post"]; SOLVES = [g["label"] for g in D["grid"]]
REF, SHIP = "inc/g1s", "inc/g2sp6"     # notes/36's chain, and the one in the submission
NOTES36_DIVJ = 0.1154                  # what REF must reproduce
print(f"{len(DS)} datasets | {len(SOLVES)} solves x {len(POST)} post-chains = {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# On the FULL set every figure comes from purescore.summarise, which is the metric:
# div_J micro-averaged (counts pooled, then divided) and adj_edge_jaccard weight-averaged
# by TP+FP+FN. Recomputing here would silently substitute unweighted means for both --
# printing one next to the other is the aggregation mismatch notes/47 §2 was about.
# For an embryo SUBSET there is no per-dataset weight in the record, so those columns are
# pooled counts for div_J and an unweighted mean for adj, and are labelled as deltas only.
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

print(f"\n{'arm':<14}{'total':>9}{'adj_edge':>10}{'div_J':>9}{'0.1divJ':>9}"
      f"{'forks':>8}{'edges':>10}")
print("-" * 69)
for a in ARMS:
    print(f"{a:<14}{total(a):>9.4f}{adj(a):>10.4f}{divJ(a):>9.4f}"
          f"{0.1 * divJ(a):>9.4f}{F.get(a, 0):>8,}{E.get(a, 0):>10,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

# ---- 1. reproduction -------------------------------------------------------------
print("\n1. ctl has a dead division term, and inc/g1s reproduces notes/36's div_J 0.1154")
c, r = divJ("ctl/g1s"), divJ(REF)
ok1 = c < 0.02 and abs(r - NOTES36_DIVJ) < 0.010
print(f"   ctl/g1s div_J {c:.4f} (want <0.02)   {REF} div_J {r:.4f} "
      f"(want {NOTES36_DIVJ:.4f} +-0.010)  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The cache or the solver has moved since notes/36. Nothing below is")
    print("   comparable to the record, and that is the finding.")

# ---- 2. THE CRUX -----------------------------------------------------------------
print(f"\n2. the shipped chain loses >0.020 of div_J vs {REF}, on IDENTICAL datasets")
drop = divJ(REF) - divJ(SHIP)
ok2 = drop > 0.020
print(f"   {REF} {divJ(REF):.4f}  ->  {SHIP} {divJ(SHIP):.4f}   drop {drop:+.4f}"
      f"  ->  {'PASS' if ok2 else 'FAIL'}")
if not ok2:
    print("   notes/42 read div_J 0.0645 at n=12 against notes/36's 0.1154 at n=24 and")
    print("   attributed the gap to config. On one dataset sample the gap is not there:")
    print("   it was the sampling artifact notes/44 predicted (the 12 were an easy")
    print("   subset by +0.0116). The division term is CLOSED, notes/36 §concluded")
    print("   correctly, and the whole remaining gap is the edge term.")

# ---- 3. attribution --------------------------------------------------------------
print("\n3. one stage dominates the drop (>half), rather than three sharing it")
# each pair differs by exactly one stage
STAGES = [("max_gap 1->2", "inc/g1s", "inc/g2s"),
          ("+prune(6)", "inc/g2s", "inc/g2sp6"),
          ("+linefit_smooth", "inc/g2p6", "inc/g2sp6"),
          ("keep_div_comp OFF", "inc/g2sp6", "inc/g2sp6_nk")]
print(f"   {'stage':<20}{'div_J before':>13}{'after':>9}{'delta':>9}{'forks lost':>12}")
deltas = []
for lbl, a, b in STAGES:
    if a not in ARMS or b not in ARMS:
        continue
    d = divJ(b) - divJ(a)
    deltas.append((abs(d), lbl))
    print(f"   {lbl:<20}{divJ(a):>13.4f}{divJ(b):>9.4f}{d:>+9.4f}"
          f"{F.get(a, 0) - F.get(b, 0):>12,}")
if deltas and drop > 1e-9:
    worst, who = max(deltas)
    ok3 = worst > drop / 2
    print(f"   largest single stage: {who} at {worst:.4f}, total drop {drop:.4f}"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   No single stage owns it; the chain degrades div_J cumulatively and")
        print("   there is no one knob to turn.")
else:
    ok3 = False
    print("   NOT GRADED — no drop to attribute")

# ---- 4. is the recovery free? ----------------------------------------------------
print(f"\n4. some arm beats {SHIP} on TOTAL, not just on div_J")
best = max(ARMS, key=lambda a: total(a) if total(a) == total(a) else -9)
gain = total(best) - total(SHIP)
ok4 = best != SHIP and gain > 0.0015      # notes/44's measurable floor
print(f"   best arm {best} {total(best):.4f} vs {SHIP} {total(SHIP):.4f}"
      f"   {gain:+.4f}  ->  {'PASS' if ok4 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best) - adj(SHIP):+.4f}"
      f"   0.1*div_J {0.1 * (divJ(best) - divJ(SHIP)):+.4f}")
if not ok4:
    print("   notes/36's trade holds: every fork recovered costs more edge score than")
    print("   the division term pays back. div_J is not free, and it is not the lever.")

# ---- 5. notes/49's rule ----------------------------------------------------------
print(f"\n5. the best arm holds on BOTH embryos (notes/49 -- n is 2, not {len(DS)})")
if best == SHIP:
    ok5 = False
    print("   NOT GRADED — the shipped chain is already best, nothing to transfer")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][SHIP]["adj"]
             for n in ns if best in PER.get(n, {}) and SHIP in PER[n]]
        dj = divJ(best, ns) - divJ(SHIP, ns)
        per[e] = (sum(d) / len(d) if d else float("nan"), dj, len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    tots = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok5 = len(tots) > 1 and (all(t > 0 for t in tots) or all(t < 0 for t in tots))
    print(f"   signs agree across embryos  ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   The arm wins on one embryo and loses on the other. The test set is a")
        print("   THIRD pair of embryos (notes/07 §3), so this does not transfer --")
        print("   this is exactly the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 78)
n_ok = sum([ok1, ok2, ok3, ok4, ok5])
print(f"{n_ok}/5 predictions passed")
if ok1 and ok2 and ok4 and ok5:
    print(f"SUBMITTABLE: {best} gains {gain:+.4f} and holds on both embryos.")
elif ok1 and not ok2:
    print("CLOSED: the div_J drop was a sampling artifact. The division term is done,")
    print("and notes/44's shortlist is now empty of cheap items.")
elif ok1 and ok2 and not ok4:
    print("PRICED, NOT SUBMITTABLE: the drop is real and attributable, but recovering it")
    print("costs more edge score than it returns. Records the trade; does not change the run.")
else:
    print("NOT COMPARABLE: reproduction failed; fix the cache before reading anything else.")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
