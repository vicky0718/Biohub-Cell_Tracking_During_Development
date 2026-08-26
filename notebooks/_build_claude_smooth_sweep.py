"""Build notebooks/claude_smooth_sweep.ipynb — sweep position repair, and measure why it works.

`notes/26`: `linefit_smooth` was the biggest lever in the ablation (+0.0086 solo, **78 % of
the whole repair gain**), and I had predicted it would be worth less than +0.005. The
prediction failed because node recall (0.995) is blind to the failure smoothing fixes: a GT
node is almost always matched to *something*, but with 20-150x more predictions than
annotations it is frequently matched to a **nearby wrong prediction** that is not on the
true track. Smoothing flips the assignment to the right one. Node recall does not move.
Edge Jaccard does.

That mechanism is **inferred, not measured** — `claude_edge_anatomy` computed the anatomy
only for `control` and `all`, so it could not attribute smoothing's gain to a bucket. This
run computes the anatomy on **every arm**, which makes the mechanism falsifiable.

Two other things this run settles, both nearly free on the existing cache:

* **Every smoothing parameter is currently a number copied from the public notebook's
  config** -- `weight=0.76`, `window=2`, `max_shift_um=3.2` -- not one measured on our
  graphs. Building motion relink on the assumption that 0.76 is optimal would be building
  on an unexamined constant.
* **Gap closing is under-firing**: 30 of 212 gaps recovered (14 %) at a 5.75 um one-frame
  radius, for 8,576 added nodes. Widen it and see where the trade turns.

`cap_edge_length` and the two ILP-redundant repairs (`prune_isolated`, `single_parent_repair`)
are out of every arm here: `notes/26` §3 measured the first as harmful (-0.0002) and the
other two as exact no-ops.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_smooth_sweep.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Sweep position repair — and measure why it works

`notes/26` found the lever, and found me wrong about it:

```
only_smooth         +0.0086     <- biggest solo arm in the ablation
all                 +0.0113
all_minus_smooth    +0.0025     <- smoothing is +0.0088 of the +0.0113
```

**78 % of the whole repair gain comes from moving node positions**, changing neither the
node set nor the edge set. I had predicted under +0.005, reasoning from node recall 0.995.

## Why that reasoning was wrong

Node recall asks *"is this GT node matched by something?"* Matching is bipartite **within a
frame**, and the pack predicts 5,000–57,000 nodes against 50–1,950 annotated ones. So a GT
node is nearly always matched — often to a **nearby wrong prediction** off the true track,
whose edges then cannot score. Smoothing makes the true track's node closest, the
assignment flips, and its edges start counting. Node recall is unchanged throughout.

Same class of error as `notes/21`, where a 0.074 edge-Jaccard gap sat under **identical**
0.866 node recall. That is why `paired_recall` exists — and I used node recall to bound
position repair anyway.

**So this run computes the anatomy on every arm.** The mechanism above predicts a specific
signature, and a bucket table can refute it.

## What is swept

| axis | values | why |
|---|---|---|
| `weight` | 0.4, 0.76, 1.0 | 0.76 is the public notebook's constant, never measured here |
| `window` | 1, 2, 3, 5 | ditto for 2 |
| `max_shift_um` | 3.2, 7.0 | 3.2 may be clamping real corrections |
| second pass | on/off | is one pass converged? |
| gap-close radius | 5.75, 8.0, 11.0 | only 14 % of gaps recovered at 5.75 µm |

`cap_edge_length` is excluded — measured **harmful** (−0.0002). `prune_isolated` and
`single_parent_repair` too: exact no-ops, because the ILP already emits graphs with no
isolated nodes and no merges.

## Pre-registered predictions

1. **The control reproduces 0.8806 ± 0.0005.** Same cache, same scorer.
2. **Smoothing moves `fn_mislink` more than `fn_detect`.** This is the mechanism test. If
   the gain comes from GT nodes matching a *different, better* prediction, mislinks should
   fall. If instead `fn_detect` falls, smoothing is pulling previously-unmatched nodes into
   range — a different mechanism, and my §2 story is wrong.
3. **The score curve in `weight` is single-peaked**, and the peak is **not** at 0.76. A peak
   exactly at the copied constant would be a suspicious coincidence worth distrusting.
4. **A second pass adds less than the first.** If it adds more, one pass was not converged
   and the whole grid is measured at the wrong operating point.

*Training data, contaminated for these weights (`notes/24` §2). Absolute levels are
inflated; every number here is a **delta on fixed datasets**, which contamination shifts
equally.*
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
CACHE = find_dir(lambda p: any(p.glob("cache_*.npz")), ["/kaggle/input"])
for lbl, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
               ("graph cache", CACHE)):
    print(f"  {lbl:<14} {v}")
if None in (PACK, REPO, COMP, CACHE):
    raise SystemExit("missing mount — CACHE needs claude-div-probe attached as a kernel source")
TRAIN = COMP / "train"
print(f"  cached graphs  {len(list(CACHE.glob('cache_*.npz')))}")

for mod in ("repair.py", "anatomy.py"):
    if not (REPO / "pipeline" / mod).exists():
        raise SystemExit(f"the mounted repo predates pipeline/{mod} — push a new "
                         "dataset version before running this")

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
           "import numpy, zarr, tracksdata; print('numpy', numpy.__version__)")
print(probe.stdout.strip() or probe.stderr.strip()[-800:])
if probe.returncode != 0:
    raise SystemExit("dependency stack does not import in a fresh interpreter")
print(f"no GPU or model needed — setup took {time.time()-T_START:.0f}s")
""")

md("""## 1. The sweep, with the anatomy on every arm

Anatomy on every arm is the point: `notes/26` §2's mechanism is currently a story, and a
per-arm bucket table is what turns it into a measurement.
""")

code(r"""
WORKER = WORK / "run_smooth.py"
WORKER.write_text(f'''
import json, os, sys, time
from pathlib import Path
import numpy as np

os.environ["CELLMOT_REPO"] = {str(CELLMOT)!r}
REPO = Path({str(REPO)!r}); TRAIN = Path({str(TRAIN)!r})
CACHE = Path({str(CACHE)!r}); WORK = Path({str(WORK)!r})
T0 = time.time()

sys.path.insert(0, str(REPO))
from harness import Harness
from harness.tracks import Tracks, read_geff, read_scale
from harness.purescore import summarise
from pipeline.anatomy import BUCKETS, edge_anatomy, summarise_anatomy
from pipeline.repair import close_gaps, linefit_smooth

print("worker numpy", np.__version__, flush=True)

def smooth(g, sc, weight, window, shift, passes=1):
    for _ in range(passes):
        g = linefit_smooth(*g, window=window, weight=weight, scale=sc,
                           max_shift_um=shift)
    return g

def gapclose(g, sc, um):
    return close_gaps(*g, scale=sc, max_um=um,
                      max_added_frac=0.038, max_added_abs=1650)

# Each arm is (label, fn(graph, scale) -> graph). Built as closures over the grid so the
# label and the parameters cannot drift apart.
ARMS = [("control", lambda g, sc: g)]
for w in (0.4, 0.76, 1.0):
    for win in (1, 2, 3, 5, 8):
        ARMS.append((f"w{{w}}_win{{win}}",
                     (lambda w_, win_: lambda g, sc: smooth(g, sc, w_, win_, 3.2))(w, win)))
# Does the 3.2 um bound clamp real corrections?
for w, win in ((0.76, 2), (1.0, 3), (1.0, 8)):
    ARMS.append((f"w{{w}}_win{{win}}_shift7",
                 (lambda w_, win_: lambda g, sc: smooth(g, sc, w_, win_, 7.0))(w, win)))
# Is one pass converged? Two arms, because on the dry run the second pass added almost as
# much as the first -- if that repeats here, one pass was nowhere near converged and the
# whole (weight, window) grid is measured at the wrong operating point.
ARMS.append(("w0.76_win2_x2", lambda g, sc: smooth(g, sc, 0.76, 2, 3.2, passes=2)))
ARMS.append(("w0.76_win2_x3", lambda g, sc: smooth(g, sc, 0.76, 2, 3.2, passes=3)))
# Gap-close radius, alone, so its curve is not confounded with smoothing.
for um in (5.75, 8.0, 11.0):
    ARMS.append((f"gap{{um}}", (lambda um_: lambda g, sc: gapclose(g, sc, um_))(um)))
# The current best chain from notes/26 §4, as an anchor.
ARMS.append(("anchor_smooth_gap",
             lambda g, sc: gapclose(smooth(g, sc, 0.76, 2, 3.2), sc, 5.75)))
print(f"{{len(ARMS)}} arms", flush=True)

names = sorted(p.stem[len("cache_"):] for p in CACHE.glob("cache_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()]
print(f"{{len(names)}} cached graphs with ground truth", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
ROWS = {{a: [] for a, _ in ARMS}}
ANAT = {{a: [] for a, _ in ARMS}}
NODES = {{a: 0 for a, _ in ARMS}}
PER = {{}}

for name in names:
    t0 = time.time()
    z = np.load(CACHE / f"cache_{{name}}.npz")
    base = (z["t"], z["zyx"], z["edges"])
    sc = read_scale(TRAIN / f"{{name}}.zarr")
    gt = read_geff(TRAIN / f"{{name}}.geff")

    for arm, fn in ARMS:
        g = fn(base, sc)
        ROWS[arm].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))
        NODES[arm] += int(len(g[0]))
        a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
        ANAT[arm].append(a)
        if sum(a[k] for k in BUCKETS) != a["n_gt_edges"]:
            raise SystemExit(f"{{name}}/{{arm}}: buckets do not sum")
        PER.setdefault(name, {{}})[arm] = float(
            ROWS[arm][-1].get("adj_edge_jaccard", float("nan")))
    best = max(PER[name], key=lambda k: PER[name][k] if PER[name][k] == PER[name][k] else -9)
    print(f"  {{name:<24}} control {{PER[name]['control']:.4f}}  best {{best}} "
          f"{{PER[name][best]:.4f}} ({{PER[name][best]-PER[name]['control']:+.4f}})  "
          f"{{time.time()-t0:.0f}}s", flush=True)

out = {{"arms": [a for a, _ in ARMS], "datasets": names,
       "summary": {{a: summarise(ROWS[a]) for a, _ in ARMS}},
       "anatomy": {{a: summarise_anatomy(ANAT[a]) for a, _ in ARMS}},
       "nodes": NODES, "per_dataset": PER}}
(WORK / "smooth_sweep.json").write_text(json.dumps(out, indent=2, default=float))
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

md("""## 2. The grid, the mechanism, and the four predictions""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "smooth_sweep.json").read_text())
S, A, NODES, ARMS, DS = D["summary"], D["anatomy"], D["nodes"], D["arms"], D["datasets"]
base, abase = S["control"], A["control"]
print(f"{len(DS)} datasets, {len(ARMS)} arms\n")

EXACT = base["score"] == base["score"]
key = "score" if EXACT else "edge_jaccard"
if not EXACT:
    print("!! the score column is NaN (unreadable node budget) — grading on "
          "`edge_jaccard` instead, which is exact either way.\n")

BUCK = ("tp", "fn_gap", "fn_mislink", "fn_detect", "fn_nonconsec")
print(f"{'arm':<22}{'score':>9}{'delta':>9}{'edge_J':>9}"
      f"{'d_mislink':>11}{'d_detect':>10}{'d_gap':>8}{'d_nodes':>10}")
print("-" * 88)
for a in ARMS:
    s, an = S[a], A[a]
    print(f"{a:<22}{s[key]:>9.4f}{s[key]-base[key]:>+9.4f}{s['edge_jaccard']:>9.4f}"
          f"{an['fn_mislink']-abase['fn_mislink']:>+11}"
          f"{an['fn_detect']-abase['fn_detect']:>+10}"
          f"{an['fn_gap']-abase['fn_gap']:>+8}"
          f"{NODES[a]-NODES['control']:>+10,}")

print()
print("=" * 88)
print("PREDICTION GRADING")
print("=" * 88)

# 1 ---------------------------------------------------------------------------------
print("\n1. the control reproduces 0.8806 +- 0.0005")
if not EXACT:
    print("   NOT GRADED — score column is NaN.")
else:
    ok = abs(base["score"] - 0.8806) <= 0.0005
    print(f"   control = {base['score']:.4f}   ->  {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("   Same cache, same scorer. A miss means nothing below is readable.")

# 2 — the mechanism test ------------------------------------------------------------
print("\n2. smoothing moves fn_mislink more than fn_detect  (the mechanism test)")
sm = [a for a in ARMS if a.startswith("w") and "gap" not in a]
best_sm = max(sm, key=lambda a: (S[a][key] if S[a][key] == S[a][key] else float("-inf")))
d_mis = abase["fn_mislink"] - A[best_sm]["fn_mislink"]
d_det = abase["fn_detect"] - A[best_sm]["fn_detect"]
d_gap = abase["fn_gap"] - A[best_sm]["fn_gap"]
print(f"   best smoothing arm: {best_sm}")
print(f"   mislinks repaired {d_mis:+}   undetected repaired {d_det:+}   "
      f"gaps repaired {d_gap:+}")
ok2 = d_mis > d_det
print(f"   ->  {'PASS' if ok2 else 'FAIL'}")
if ok2:
    print("   Consistent with notes/26 §2: the GT node's bipartite match flips from a")
    print("   nearby wrong prediction to the one actually on the track. Node recall")
    print("   cannot see this, which is why it bounded the lever wrongly.")
else:
    print("   The gain is NOT coming from re-assignment. notes/26 §2's story is wrong —")
    print("   smoothing is pulling previously unmatched nodes into the 7 µm radius, which")
    print("   is a detection-localisation effect. Correct notes/26 before building on it.")

# 3 ---------------------------------------------------------------------------------
print("\n3. the weight curve is single-peaked, and the peak is NOT at the copied 0.76")
for win in sorted({int(a.split("win")[1]) for a in ARMS
                   if a.startswith("w") and "win" in a and "_" not in a.split("win")[1]}):
    curve = []
    for w in (0.4, 0.76, 1.0):
        lbl = f"w{w}_win{win}"
        if lbl in S:
            curve.append((w, S[lbl][key]))
    if len(curve) < 3:
        continue
    ys = [y for _, y in curve]
    pk = int(np.argmax(ys))
    print(f"   window={win}:  " + "  ".join(f"w{w:g}={y:.4f}" for w, y in curve)
          + f"   peak w={curve[pk][0]:g}")
peaks = {a: S[a][key] for a in sm}
top = max(peaks, key=peaks.get)
print(f"   best over the whole grid: {top} at {peaks[top]:.4f} "
      f"({peaks[top]-base[key]:+.4f})")
print(f"   ->  {'PASS — the copied constant was not optimal' if top != 'w0.76_win2' else 'FAIL — 0.76/2 wins, which is suspicious and worth distrusting'}")

# 4 ---------------------------------------------------------------------------------
print("\n4. a second pass adds less than the first")
if "w0.76_win2_x2" in S and "w0.76_win2" in S:
    one = S["w0.76_win2"][key] - base[key]
    two = S["w0.76_win2_x2"][key] - base[key]
    print(f"   one pass {one:+.4f}   two passes {two:+.4f}   second pass adds "
          f"{two-one:+.4f}")
    ok4 = (two - one) < one
    print(f"   ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   One pass was not converged, so the whole grid is measured at the wrong")
        print("   operating point. Re-run with more passes before reading the grid.")

# gap-close curve --------------------------------------------------------------------
print("\ngap-close radius (notes/26 §4: only 14% of gaps recovered at 5.75 µm)")
for a in [x for x in ARMS if x.startswith("gap")]:
    print(f"   {a:<12}{S[a][key]:>9.4f}{S[a][key]-base[key]:>+9.4f}"
          f"   gaps repaired {abase['fn_gap']-A[a]['fn_gap']:>+5}"
          f"   nodes {NODES[a]-NODES['control']:>+8,}")

print()
print("=" * 88)
best = max((a for a in ARMS if a != "control"),
           key=lambda a: (S[a][key] if S[a][key] == S[a][key] else float("-inf")))
print(f"BEST ARM: {best} at {S[best][key]:.4f} ({S[best][key]-base[key]:+.4f})")

# A sweep that peaks at its own edge has not found an optimum, it has run out of grid.
# Saying so is the difference between a measurement and a number.
import re
W_VALS, WIN_VALS = (0.4, 0.76, 1.0), (1, 2, 3, 5, 8)
# Match any arm carrying a (weight, window) — `w1.0_win8_shift7` and `w0.76_win2_x2` are
# grid points too, and an earlier version's `count("_") == 1` test silently skipped them.
_m = re.match(r"^w([\d.]+)_win(\d+)", best)
if _m:
    bw, bwin = float(_m.group(1)), int(_m.group(2))
    edge = []
    if bw in (min(W_VALS), max(W_VALS)):
        edge.append(f"weight={bw:g} is the {'lowest' if bw == min(W_VALS) else 'highest'} swept")
    if bwin in (min(WIN_VALS), max(WIN_VALS)):
        edge.append(f"window={bwin} is the {'smallest' if bwin == min(WIN_VALS) else 'largest'} swept")
    if edge:
        print(f"  !! BOUNDARY: {'; '.join(edge)}. The true optimum may lie outside the")
        print("     grid, so this is a lower bound on what position repair is worth.")
        print("     Widen the grid before treating this setting as tuned.")
print(f"  vs notes/26's best chain (+0.0115). Reachable ceiling was +0.047 to +0.079.")
print(f"  remaining: mislink {A[best]['fn_mislink']:,}  gap {A[best]['fn_gap']:,}  "
      f"undetected {A[best]['fn_detect']:,}")
print("  Next is motion relink, which needs a prediction pass caching edge_prob —")
print("  the current cache has no probabilities in it.")
print("=" * 88)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
