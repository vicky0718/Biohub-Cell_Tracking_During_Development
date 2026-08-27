"""Build notebooks/claude_ilp_sweep2.ipynb — the two regions `claude_ilp_sweep` left unsampled.

v1 (`notes/31`) swept the ILP's weights for the first time and produced the change that took
the leaderboard 0.880 -> 0.883 (`notes/32`). It also left two specific gaps, both of which
re-solve the SAME cached instances and so cost solver time and nothing else:

* **`division_weight` between 0.5 and 1.0.** v1 swept 0.5 / 0.2 / 0.0 / -0.5 and every one
  returned *byte-identical* results -- 7,149 forks, `div_J` 0.0559. The knob saturates
  immediately, so v1 measured two points, not five: the default (54 forks, `div_J` 0.0000)
  and saturation (7,149 forks, `div_J` 0.0562). Those two endpoints price out to a **wash
  within 0.0002** -- 132x more forks buys +0.0056 on the division term and gives back
  -0.0058 on the edge term. Everything between them is unsampled, and `division_jaccard`
  is 0.0562 of an available 1.000.
* **Asymmetry past 0.1/0.5.** `asym0.1_0.5` was the largest asymmetry in v1's grid and it
  won, so the optimum sits **on the boundary**. `notes/32` confirmed the direction transfers
  to the hidden test set, which is what makes extending the grid a search rather than a
  fishing trip. The mechanism has a known cost (`notes/31` §3: -29 mislink, -18 gap, **+36
  undetected**), so this is a search for where that tax inverts, not an assumption that more
  is better.

The baseline to beat here is **not** `control`. It is `asym0.1_0.5+repair` = **0.8958**, the
arm currently scoring 0.883. Every delta below is reported against both.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_ilp_sweep2.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# The two regions the first ILP sweep left unsampled

```
0.752  classical champion
0.867  pack + ILP
0.880  + gap-close + linefit-smooth
0.883  + disappearance 0.1 -> 0.5           <- best scored (notes/32)
0.913–0.916  the cluster, same weights      <- 0.030–0.033 away
```

`claude_ilp_sweep` (`notes/31`) was the first run to touch the solve's own weights, and it
produced the +0.003 that is currently on the board. It also left two gaps, both of which
re-solve the **same cached instances** — no model, no GPU, no prediction pass.

## Gap 1 — `division_weight` saturates instantly, so v1 measured two points, not five

| `division_weight` | forks | `division_jaccard` |
|---|---|---|
| 1.0 (default) | 54 | 0.0000 |
| 0.5 / 0.2 / 0.0 / −0.5 | 7,149 — **byte-identical** | 0.0562 |

The two endpoints price out to a **wash within 0.0002**: 132× more forks buys +0.0056 on
the division term and gives back −0.0058 on the edge term. That is not "dead" — it is
**live and priced at zero**, and a price can move. Nothing between 0.5 and 1.0 was ever
sampled, and `division_jaccard` is 0.0562 of an available 1.000.

## Gap 2 — the winning asymmetry sits on the grid boundary

`asym0.1_0.5` was the **largest** asymmetry v1 tried, and it won. `notes/32` confirmed the
direction transfers to the hidden test set. But the mechanism has a measured cost
(`notes/31` §3):

```
fn_mislink  411 -> 382 (-29)    fn_gap  177 -> 159 (-18)    fn_detect  218 -> 254 (+36)
```

Raising the disappearance penalty makes the solver reluctant to end a track, so it links
through ambiguity it previously abandoned — some links right, some stranding nodes that no
longer match. **This grid searches for where that tax inverts**, not for "more is better."

## The baseline is 0.8958, not 0.8806

`control` is kept only as the cache-integrity check. The arm to beat is
`asym0.1_0.5+repair` = **0.8958**, which is what 0.883 was scored from.

## Pre-registered predictions

1. **`control` reproduces 0.8806 ± 0.0005 and `asym0.1_0.5+repair` reproduces 0.8958 ±
   0.0005.** The load-bearing check: same cache, same solver, two known answers. If either
   misses, nothing below is readable.
2. **`division_weight` in (0.5, 1.0) gives fork counts strictly between 54 and 7,149.** If
   every value in the gap returns one of the two endpoint counts, the knob is a **step
   function** and there is nothing in between to find — an informative failure that closes
   gap 1 permanently.
3. **Some intermediate `division_weight` beats both endpoints on total score.** The real
   question. The endpoints wash to 0.0002; this asks whether the curve between them is flat
   or humped.
4. **The asymmetry optimum is interior to the extended grid** — i.e. pushing disappearance
   past 0.5 eventually *hurts*, as the `fn_detect` tax overtakes the mislink/gap gain. If
   the best arm is again on the boundary, the knob has more room than two grids can find and
   that is a different finding, reported as such.
5. **The asymmetry gain survives a matched-magnitude symmetric control at the new scales.**
   v1 established this at 0.1/0.5 (+0.0017 attributable to asymmetry itself). It has to be
   re-established at every new scale, or a gain from "larger" gets reported as a gain from
   "asymmetric" — the exact confusion v1's design was built to prevent.

*Training data, contaminated for these weights (`notes/24` §2). `notes/32` §2 has two
transfer measurements that do not agree (1.13× and 0.81×, overlapping only on a sliver once
leaderboard rounding is propagated) — so read a train-side gain as roughly its face value,
not through a multiplier.*
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
BEST = (-1.0, 0.1, 0.5, 1.0)                      # notes/32: the arm scoring 0.883
ARMS = [("control", *DEF), ("asym0.1_0.5", *BEST)]
# Prediction 2/3: the gap v1 never sampled. Below 0.5 every value was byte-identical, so
# anything interesting lives strictly between the default and where it saturates.
# 0.5 is included even though v1 already measured it: it is the SATURATION endpoint, and
# prediction 2 asks whether the gap lies strictly between the two endpoints. Re-measuring it
# here makes that check self-contained instead of leaning on a cross-run constant.
for dv in (0.9, 0.8, 0.7, 0.6, 0.5):
    ARMS.append((f"div{{dv}}", DEF[0], DEF[1], DEF[2], dv))
# Prediction 4: push disappearance past the 0.5 boundary the winner sits on, holding
# appearance at the pack's 0.1 exactly as the winning arm does.
for dis in (0.75, 1.0, 1.5, 2.0):
    ARMS.append((f"asym0.1_{{dis}}", DEF[0], 0.1, dis, DEF[3]))
# The lab's 5x RATIO carried to larger scales -- a different axis from raising disappearance
# alone, and the one notes/03 §3 actually documents.
for ap, dis in ((0.2, 1.0), (0.4, 2.0)):
    ARMS.append((f"ratio{{ap}}_{{dis}}", DEF[0], ap, dis, DEF[3]))
# Prediction 5: matched-magnitude symmetric controls at the new scales. Without these a
# gain from "larger" reads as a gain from "asymmetric".
for sy in (0.5, 1.0):
    ARMS.append((f"sym{{sy}}", DEF[0], sy, sy, DEF[3]))
# Do the two knobs interact? v1 swept them on separate axes and never crossed them, so
# "division is priced at zero" was measured only at the DEFAULT asymmetry.
for dv in (0.8, 0.6):
    ARMS.append((f"div{{dv}}+asym", BEST[0], BEST[1], BEST[2], dv))
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
    (WORK / "ilp_sweep2.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. The five predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "ilp_sweep2.json").read_text())
S, A, F, E = D["summary"], D["anatomy"], D["forks"], D["edges"]
ARMS, DS, GRID = D["arms"], D["datasets"], {g["label"]: g for g in D["grid"]}
base, abase = S["control"], A["control"]
EXACT = base["score"] == base["score"]
key = "score" if EXACT else "edge_jaccard"
# `GRID` is keyed by BASE arm labels only, while `ARMS` also carries every `+repair`
# variant -- so anything reading weights out of GRID must filter to the base arms first.
BASE = [a for a in ARMS if a in GRID]
REF = "asym0.1_0.5+repair"          # the arm currently scoring 0.883 -- the real baseline
print(f"{len(DS)} datasets, {len(ARMS)} arms ({len(BASE)} weight settings x with/without repair)")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

# weights of a base arm, or of the base arm a `+repair` label was built from.
# A docstring here would terminate the outer r-string that emits this cell -- the 6th time
# this exact trap has come up in this project, so it stays a comment.
def w(a):
    return GRID.get(a) or GRID.get(a[:-len("+repair")]) or {}

print(f"{'arm':<20}{'score':>9}{'vs ctl':>9}{'vs 0.883':>10}{'edge_J':>9}"
      f"{'div_J':>8}{'forks':>8}{'d_mis':>7}{'d_det':>7}")
print("-" * 87)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    dj = st.get("division_jaccard")
    print(f"{a:<20}{st[key]:>9.4f}{st[key]-base[key]:>+9.4f}"
          f"{(st[key]-S[REF][key] if REF in S else float('nan')):>+10.4f}"
          f"{st['edge_jaccard']:>9.4f}{(dj if dj == dj else 0):>8.4f}{F[a]:>8,}"
          f"{an['fn_mislink']-abase['fn_mislink']:>+7}{an['fn_detect']-abase['fn_detect']:>+7}")

print()
print("=" * 87)
print("PREDICTION GRADING")
print("=" * 87)

# 1 ---------------------------------------------------------------------------------
print("\n1. control reproduces 0.8806 and asym0.1_0.5+repair reproduces 0.8958 (+-0.0005)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
elif len(DS) != 24:
    print(f"   NOT GRADED — {len(DS)} datasets completed, not 24.")
else:
    hits = [("control", base["score"], 0.8806), (REF, S[REF]["score"], 0.8958)]
    ok1 = True
    for lbl, got, want in hits:
        good = abs(got - want) <= 0.0005
        ok1 &= good
        print(f"   {lbl:<20} {got:.4f} vs {want:.4f}  {'PASS' if good else 'FAIL'}")
    if not ok1:
        print("   The cache does NOT round-trip to the graphs notes/31 scored. Nothing below")
        print("   is readable — diagnose that before reading any arm.")

# 2 ---------------------------------------------------------------------------------
print("\n2. division_weight in (0.5, 1.0) gives fork counts strictly between the endpoints")
divs = sorted(((w(a)["division"], a) for a in BASE
               if w(a)["disappear"] == 0.1 and w(a)["division"] != 1.0), reverse=True)
lo, hi = F["control"], (F[dict(((d, a) for d, a in divs)).get(0.5, "control")]
                        if any(d == 0.5 for d, _ in divs) else None)
print(f"   default (dv=1.0): {lo:,} forks     saturation (dv=0.5): {hi:,} forks"
      if hi is not None else f"   default: {lo:,} forks; no dv=0.5 arm to bound with")
print("   " + "  ".join(f"dv={d:g}:{F[a]:,}" for d, a in divs))
if hi is not None:
    interior = [(d, F[a]) for d, a in divs if 0.5 < d < 1.0]
    strictly = [c for _, c in interior if lo < c < hi]
    ok2 = len(strictly) > 0
    print(f"   {len(strictly)}/{len(interior)} interior settings land strictly between "
          f"->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   The knob is a STEP FUNCTION, not a dial: every value in the gap returns an")
        print("   endpoint count. There is nothing between them to find and gap 1 is closed")
        print("   permanently — that is a real answer, not a failed run.")

# 3 ---------------------------------------------------------------------------------
print("\n3. some intermediate division_weight beats BOTH endpoints on total score")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ends = [S["control"][key]] + ([S[a][key] for d, a in divs if d == 0.5] or [])
    mids = [(S[a][key], a) for d, a in divs if 0.5 < d < 1.0 and a in S]
    if mids and len(ends) == 2:
        bv, ba = max(mids)
        ok3 = bv > max(ends) + 1e-6
        print(f"   endpoints {ends[0]:.4f} (dv=1.0) / {ends[1]:.4f} (dv=0.5)")
        print(f"   best interior: {ba} {bv:.4f}  ->  {'PASS' if ok3 else 'FAIL'}")
        print(f"   {'The curve between the endpoints is HUMPED — an intermediate fork count'if ok3 else 'The curve between the endpoints is flat or monotone: the wash notes/31'}")
        print(f"   {'buys the division term without paying the full edge cost.' if ok3 else 'measured at the endpoints holds across the whole gap. Division is priced'}")
        if not ok3:
            print("   at zero everywhere it can be reached, and should not be revisited.")
    else:
        print("   NOT GRADED — need both endpoints and at least one interior arm.")

# 4 ---------------------------------------------------------------------------------
print("\n4. the asymmetry optimum is INTERIOR to the extended grid")
asym = sorted((w(a)["disappear"], a) for a in BASE
              if w(a)["appear"] == 0.1 and w(a)["division"] == 1.0 and w(a)["disappear"] != 0.1)
if not EXACT or len(asym) < 3:
    print(f"   NOT GRADED — {'score column is NaN' if not EXACT else f'only {len(asym)} asymmetry arms'}.")
else:
    print("   " + "  ".join(f"dis={d:g}:{S[a][key]:.4f}" for d, a in asym))
    print("   " + "  ".join(f"dis={d:g}:det{A[a]['fn_detect']-abase['fn_detect']:+d}"
                            for d, a in asym))
    bd, ba = max(((S[a][key], (d, a)) for d, a in asym))[1]
    edge_of_grid = bd == max(d for d, _ in asym)
    ok4 = not edge_of_grid
    print(f"   best: {ba} (disappear {bd:g})  ->  {'PASS — interior' if ok4 else 'FAIL — on the boundary again'}")
    if ok4:
        print("   The fn_detect tax overtakes the mislink/gap gain, as notes/31 §3 predicted,")
        print("   and this grid contains the turn. That optimum is submittable.")
    else:
        print("   The knob still has room past the widest setting tried. Do NOT submit the")
        print("   boundary arm on the assumption it is near-optimal — extend once more first,")
        print("   and expect the turn to exist somewhere, since fn_detect rises monotonically.")

# 5 ---------------------------------------------------------------------------------
print("\n5. the asymmetry gain survives a matched-magnitude symmetric control")
sym = {w(a)["appear"]: a for a in BASE
       if w(a)["appear"] == w(a)["disappear"] and w(a)["appear"] != 0.1}
if not EXACT or not sym:
    print(f"   NOT GRADED — {'score column is NaN' if not EXACT else 'no symmetric controls'}.")
else:
    ok5 = True
    for d, a in asym:
        m = sym.get(d)
        if m is None:
            continue
        margin = S[a][key] - S[m][key]
        ok5 &= margin > 0
        print(f"   {a:<16} {S[a][key]:.4f}  vs  {m:<10} {S[m][key]:.4f}   {margin:+.4f}"
              f"  {'asymmetry' if margin > 0 else 'MAGNITUDE'}")
    print(f"   ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   At least one scale's gain is really a MAGNITUDE effect. Report it as that —")
        print("   the lab's ratio is not what is doing the work there.")

# interaction ------------------------------------------------------------------------
print("\n(extra) do the two knobs interact? v1 measured division only at the DEFAULT asymmetry")
combo = [a for a in BASE if a.endswith("+asym")]
if EXACT and combo:
    for c in combo:
        stem = c[:-len("+asym")]
        if stem in S and REF in S:
            solo = S[stem][key] - base[key]           # division alone, vs control
            both = S[c][key] - S["asym0.1_0.5"][key]  # division on top of the asymmetry
            print(f"   {stem}: alone {solo:+.4f}   on top of asym0.1_0.5 {both:+.4f}"
                  f"   {'ADDITIVE' if abs(both-solo) < 0.001 else 'INTERACTS'}")
else:
    print("   NOT GRADED.")

print()
print("=" * 87)
if not (EXACT and REF in S):
    print("NO BEST ARM — score column is NaN or the reference arm is missing.")
else:
    cands = [a for a in ARMS if a.endswith("+repair") and a in S]
    best = max(cands, key=lambda a: S[a][key])
    print(f"BEST ARM: {best} at {S[best][key]:.4f}")
    print(f"  vs control          {S[best][key]-base[key]:+.4f}")
    print(f"  vs the 0.883 arm    {S[best][key]-S[REF][key]:+.4f}   <- the only delta that matters")
    if S[best][key] - S[REF][key] <= 0.0005:
        print("  Nothing here beats what is already submitted by a readable margin.")
        print("  Do not spend a submission slot. Both gaps are then measured and closed.")
    else:
        print(f"  weights: {w(best)}")
    print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
          f"undetected {A[best]['fn_detect']:,}")
print("=" * 87)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
