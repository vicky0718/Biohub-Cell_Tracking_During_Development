"""Build notebooks/claude_ilp_sweep3.ipynb — past the boundary, on all three axes.

`notes/35`: sweep2 returned +0.0221 on `ratio0.4_2.0+repair` and failed prediction 4 for the
second run running. Every axis was still climbing at its largest setting:

    asym0.1_2.0   0.8975   the largest asym arm  -- won its axis
    sym1.0        0.9036   the largest sym arm   -- won its axis
    ratio0.4_2.0  0.9093   the largest ratio arm -- won overall

The `fn_detect` tax rises monotonically with magnitude (+46 -> +330 across the asym axis),
so a turn must exist. It is not inside sweep2's grid either.

Two things sweep2 established that shape this grid:

* **Divisions come from the termination penalty, not `division_weight`.** A high
  disappearance cost makes the solver unwilling to end a track, so it forks instead:
  div_J 0.0000 -> 0.1154. `ratio0.4_2.0` reaches the best div_J with a THIRD of `sym0.5`'s
  forks, so magnitude buys precision here, not volume.
* **The appear:disappear ratio flips sign with scale.** At magnitude ~0.5 the symmetric arm
  beat the 1:5 ratio by 0.0132; at magnitude ~2 the 1:5 ratio beat symmetric by 0.0057.
  Nothing in `notes/35` explains that, so the ratio gets its own axis here at fixed
  magnitude rather than being confounded with it again -- which is exactly the error
  `notes/35` §1 had to correct in the first place.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_ilp_sweep3.ipynb")
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
BEST = (-1.0, 0.4, 2.0, 1.0)                      # notes/35: sweep2's best, ON THE BOUNDARY
ARMS = [("control", *DEF), ("ratio0.4_2.0", *BEST)]
# AXIS 1 -- magnitude at the 1:5 ratio that won, pushed well past 2.0. If the fn_detect tax
# ever overtakes, it turns here.
for ap in (0.8, 1.2, 1.6, 2.4, 3.2):
    ARMS.append((f"r5_{{ap}}", DEF[0], ap, ap * 5.0, DEF[3]))
# AXIS 2 -- symmetric magnitude past 1.0, the other axis still climbing.
for sy in (2.0, 3.0, 4.0, 6.0):
    ARMS.append((f"sym{{sy}}", DEF[0], sy, sy, DEF[3]))
# AXIS 3 -- the RATIO itself, at FIXED total magnitude. sweep2 could not separate ratio from
# magnitude because it never held one constant; notes/35 §1 is the correction that mistake
# forced. Disappearance is pinned at 2.0 and only appearance moves.
# 2.0 is deliberately ABSENT: (2.0, 2.0) is sym2.0 on axis 2, and running it twice would
# spend a second 466 s solve to report one measurement as two. The grading splices sym2.0
# in as this axis's 1:1 endpoint instead.
for ap in (0.13, 0.25, 0.67, 1.0, 1.4):
    ARMS.append((f"ap{{ap}}_dis2.0", DEF[0], ap, 2.0, DEF[3]))
# AXIS 4 -- division_weight at the NEW operating point. notes/35 §2 found div0.7 humped at
# the DEFAULT magnitude; whether that still holds where the termination penalty is already
# making forks is a different question, and the two could easily be substitutes.
for dv in (0.7, 0.85):
    ARMS.append((f"div{{dv}}+r5", BEST[0], BEST[1], BEST[2], dv))
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
    (WORK / "ilp_sweep3.json").write_text(json.dumps(out, indent=2, default=float))

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
D = json.loads((WORK / "ilp_sweep3.json").read_text())
S, A, F, E = D["summary"], D["anatomy"], D["forks"], D["edges"]
ARMS, DS, GRID = D["arms"], D["datasets"], {g["label"]: g for g in D["grid"]}
base = S["control"]
EXACT = base["score"] == base["score"]
key = "score" if EXACT else "edge_jaccard"
BASE = [a for a in ARMS if a in GRID]
REF = "ratio0.4_2.0+repair"          # notes/35: sweep2's best, 0.9179 -- the arm to beat
print(f"{len(DS)} datasets, {len(ARMS)} arms ({len(BASE)} weight settings x with/without repair)")
if not EXACT:
    print("!! score column is NaN (unreadable node budget) — grading on `edge_jaccard`.")
print()

# weights of a base arm, or of the base arm a `+repair` label was built from.
# A docstring here would terminate the outer r-string that emits this cell.
def w(a):
    return GRID.get(a) or GRID.get(a[:-len("+repair")]) or {}

def mag(a):
    g = w(a)
    return max(g.get("appear", 0.0), g.get("disappear", 0.0))

print(f"{'arm':<18}{'score':>9}{'vs ctl':>9}{'vs 0.9179':>11}{'edge_J':>9}"
      f"{'div_J':>8}{'forks':>8}{'d_mis':>7}{'d_det':>7}")
print("-" * 86)
for a in ARMS:
    if a not in S:
        continue
    st, an = S[a], A[a]
    dj = st.get("division_jaccard")
    print(f"{a:<18}{st[key]:>9.4f}{st[key]-base[key]:>+9.4f}"
          f"{(st[key]-S[REF][key] if REF in S else float('nan')):>+11.4f}"
          f"{st['edge_jaccard']:>9.4f}{(dj if dj == dj else 0):>8.4f}{F[a]:>8,}"
          f"{an['fn_mislink']-A['control']['fn_mislink']:>+7}"
          f"{an['fn_detect']-A['control']['fn_detect']:>+7}")

print()
print("=" * 86)
print("PREDICTION GRADING")
print("=" * 86)

# 1 ---------------------------------------------------------------------------------
print("\n1. control reproduces 0.8806 and ratio0.4_2.0+repair reproduces 0.9179 (+-0.0005)")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
elif len(DS) != 24:
    print(f"   NOT GRADED — {len(DS)} datasets completed, not 24.")
else:
    ok1 = True
    for lbl, want in (("control", 0.8806), (REF, 0.9179)):
        got = S[lbl]["score"]
        good = abs(got - want) <= 0.0005
        ok1 &= good
        print(f"   {lbl:<22} {got:.4f} vs {want:.4f}  {'PASS' if good else 'FAIL'}")
    if not ok1:
        print("   The cache does NOT round-trip to the graphs notes/35 scored. Nothing below")
        print("   is readable — diagnose that before reading any arm.")

# 2 ---------------------------------------------------------------------------------
print("\n2. each axis TURNS inside this grid (the third time of asking)")
AXES = {
    "1: magnitude at 1:5": [a for a in BASE if a.startswith("r5_")] + ["ratio0.4_2.0"],
    "2: symmetric magnitude": [a for a in BASE if a.startswith("sym")],
    # sym2.0 IS this axis's 1:1 endpoint — same weights, it just lives on axis 2 so the
    # solve is not paid for twice.
    "3: ratio at fixed disappear=2.0": ([a for a in BASE if a.startswith("ap")]
                                        + ["ratio0.4_2.0", "sym2.0"]),
}
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ok2 = True
    for nm, arms in AXES.items():
        arms = [a for a in arms if a in S]
        if len(arms) < 3:
            print(f"   {nm}: NOT GRADED — {len(arms)} arms")
            continue
        ordered = sorted(arms, key=lambda a: (w(a)["appear"], w(a)["disappear"]))
        vals = [(a, S[a][key]) for a in ordered]
        best = max(vals, key=lambda kv: kv[1])[0]
        interior = best not in (ordered[0], ordered[-1])
        ok2 &= interior
        print(f"   {nm}")
        print("      " + "  ".join(f"{a.replace('ap','').replace('r5_','')}:{v:.4f}"
                                   for a, v in vals))
        print(f"      best {best}  ->  {'INTERIOR' if interior else 'ON THE BOUNDARY'}")
    print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   An axis that is STILL climbing after three grids is not a knob with a")
        print("   nearby optimum — it is telling us the objective wants far fewer surviving")
        print("   nodes than the pack's defaults produce, and the node-budget term is paying")
        print("   for it (notes/35 §3). Read the d_det column before extending a fourth time.")

# 3 ---------------------------------------------------------------------------------
print("\n3. the fn_detect tax rises monotonically with magnitude (the mechanism)")
pts = sorted((mag(a), a) for a in BASE if a in A)
taxes = [(m, A[a]["fn_detect"] - A["control"]["fn_detect"]) for m, a in pts]
print("   " + "  ".join(f"{m:g}:{t:+d}" for m, t in taxes[:12]))
vals = [t for _, t in taxes]
ok3 = sum(b >= a for a, b in zip(vals, vals[1:])) >= 0.8 * max(len(vals) - 1, 1)
print(f"   ->  {'PASS — monotone in 80%+ of steps' if ok3 else 'FAIL — not monotone'}")

# 4 ---------------------------------------------------------------------------------
print("\n4. at FIXED magnitude the ratio has an optimum (sweep2 could not separate these)")
ax3 = sorted(((w(a)["appear"], a) for a in AXES["3: ratio at fixed disappear=2.0"] if a in S))
if not EXACT or len(ax3) < 4:
    print(f"   NOT GRADED — {len(ax3)} arms at fixed disappear.")
else:
    for ap, a in ax3:
        print(f"   appear {ap:<5g} (ratio 1:{2.0/ap:>5.1f})  {S[a][key]:.4f}  div_J "
              f"{S[a].get('division_jaccard', 0):.4f}  forks {F[a]:,}")
    bap, ba = max(ax3, key=lambda t: S[t[1]][key])
    ok4 = ba not in (ax3[0][1], ax3[-1][1])
    print(f"   best appear {bap:g} (1:{2.0/bap:.1f})  ->  "
          f"{'PASS — interior' if ok4 else 'FAIL — on the boundary'}")
    print("   This is the number sweep2 could not produce: the ratio's own optimum, with")
    print("   magnitude held constant so it cannot be confounded with it (notes/35 §1).")

# 5 ---------------------------------------------------------------------------------
print("\n5. division_weight and the termination penalty are SUBSTITUTES, not complements")
combo = [a for a in BASE if a.endswith("+r5")]
if not EXACT or not combo or "ratio0.4_2.0" not in S:
    print("   NOT GRADED.")
else:
    r5 = S["ratio0.4_2.0"][key]
    ok5 = True
    for c in combo:
        d = S[c][key] - r5
        ok5 &= d <= 0.001
        print(f"   {c:<16} {S[c][key]:.4f}  ({d:+.4f} on top of ratio0.4_2.0)  "
              f"forks {F[c]:,} vs {F['ratio0.4_2.0']:,}  div_J "
              f"{S[c].get('division_jaccard', 0):.4f} vs {S['ratio0.4_2.0'].get('division_jaccard', 0):.4f}")
    print(f"   ->  {'PASS — substitutes, as expected' if ok5 else 'FAIL — they ADD'}")
    print("   " + ("Both knobs create forks, so once the termination penalty is making them"
                   if ok5 else
                   "They are COMPLEMENTARY: cheapening divisions buys something the"))
    print("   " + ("cheapening divisions has nothing left to buy."
                   if ok5 else
                   "termination penalty does not. That reopens division_weight at the new"
                   " operating point."))

print()
print("=" * 86)
if EXACT and REF in S:
    cands = [a for a in ARMS if a.endswith("+repair") and a in S]
    best = max(cands, key=lambda a: S[a][key])
    d = S[best][key] - S[REF][key]
    print(f"BEST ARM: {best} at {S[best][key]:.4f}")
    print(f"  vs control            {S[best][key]-base[key]:+.4f}")
    print(f"  vs sweep2's best      {d:+.4f}   <- the only delta that matters")
    # notes/34: a PASS threshold below the noise is how a +0.0000 arm got called
    # "submittable". The leaderboard reports 3 decimals; anything under 0.001 is not a
    # result, it is a coin flip.
    if d <= 0.001:
        print("  INSIDE NOISE (<0.001). sweep2's ratio0.4_2.0 stands; do not re-submit for this.")
    else:
        print(f"  weights: {w(best)}")
        print(f"  raw edge_J {S[best]['edge_jaccard']:.4f}  div_J "
              f"{S[best].get('division_jaccard', 0):.4f}  forks {F[best]:,}")
    print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
          f"undetected {A[best]['fn_detect']:,}")
else:
    print("NO BEST ARM — score column is NaN or the reference arm is missing.")
print("=" * 86)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
