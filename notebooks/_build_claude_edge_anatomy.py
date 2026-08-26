"""Build notebooks/claude_edge_anatomy.ipynb — split the edge loss, then ablate the repairs.

`notes/25` decomposed the score *across* its three terms. `claude_div_probe` then priced
the term that decomposition pointed at — divisions — at **+0.0015**, and closed it. Two
hypotheses have now been costed and both came up near zero, and they share a cause: the
loss was never decomposed *within* the edge term, which is where nearly all of it is.

On the budget-stratified 24 the pack's post-ILP graph scores:

    edge_jaccard 0.8902   deficit 0.1098
    multiplier   0.9892   (BELOW 1 -- net over budget, so pruning is not a mirage)
    division_J   0.0000

with node recall **0.995**. Nearly every annotated cell is found, so the missing edges are
not missing cells: both endpoints are detected and the graph links them wrong, or not at
all. Those two failures have different fixes, and nothing so far says which dominates.

## Why this run is cheap

`claude_div_probe` cached all 24 post-ILP graphs as kernel output. Attaching that kernel
skips the 27-minute prediction entirely, so this needs **no GPU and no model** -- only the
pack's wheels, for a numpy/tracksdata stack coherent enough to run the official scorer
(the graphs fork, and `purescore`'s division term is exact only fork-free).

That makes the run minutes rather than half an hour, which is the point: whatever it says,
the follow-up is another few minutes rather than another half hour.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_edge_anatomy.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Where is the edge loss — gap, mislink, or never detected?

Two hypotheses priced, both near zero:

| lever | predicted | **measured** |
|---|---|---|
| per-dataset budget calibration | ~0.02 | **~0.002** |
| geometric division insertion | 0.02–0.04 | **+0.0015** |

Both failures share a cause: **I picked what to build before measuring where the loss
was.** This run makes the measurement that should have come first.

## The state of the edge term

```
edge_jaccard 0.8902     deficit 0.1098      <- effectively all of it
multiplier   0.9892     costing  0.0096     <- BELOW 1: net over budget
division_J   0.0000     costing  0.1000     <- priced at +0.0015, closed
node recall  0.9950
```

Node recall 0.995 means the cells **are** found. So the deficit is linking, not detection.
`notes/04` §7's "linking is worth ≤0.015" does not transfer: that ceiling was measured
with GT nodes fed in, a handful of competitors each. Here 5,000–57,000 predicted nodes
contest 50–1,950 annotated ones — 20–150× more candidates per real link.

## Part A — the anatomy

Every GT edge lands in exactly one bucket, and they sum to the GT edge count:

| bucket | meaning | fix |
|---|---|---|
| `tp` | both matched, prediction links them | — |
| `fn_gap` | both matched, **neither end committed**, still unlinked | gap closing |
| `fn_mislink` | both matched, but an end is already taken | motion relink |
| `fn_detect` | an endpoint never matched | nothing at graph level |
| `fn_nonconsec` | GT edge doesn't span `t→t+1` | nothing; scorer drops it |

Classified against `purescore.survivors`, not raw edges — a link the out-degree cap
truncated is not a link the metric ever saw.

## Part B — the ablation

Five repairs that need only `(t, zyx, edges)`: `prune_isolated`, `cap_edge_length`,
`single_parent_repair`, `linefit_smooth`, `close_gaps`. Each alone, all together, and each
left out of the full chain — leave-one-out because these interact and a solo delta can
flatter a repair that another one would have made redundant.

## Pre-registered predictions

1. **The control reproduces 0.8806 ± 0.0005.** Same graphs, same scorer. If it misses,
   the cache or the scoring path is wrong and nothing below is readable.
2. **The four buckets sum to the metric's own edge FN + TP**, per dataset. A leak means
   an edge was classified twice or not at all.
3. **`prune_isolated` is non-negative on every dataset.** A node with no edges cannot
   carry edge Jaccard, so this one is monotone by construction — a negative would mean
   the remap is broken.
4. **`linefit_smooth` gains less than +0.005.** Node recall is already 0.995, so there is
   almost nothing for position repair to recover. If it gains more, my reading of node
   recall is wrong and the plan should say so.

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
# claude_div_probe's kernel output: the 24 post-ILP graphs, so nothing is re-predicted.
CACHE = find_dir(lambda p: any(p.glob("cache_*.npz")), ["/kaggle/input"])
for lbl, v in (("pack", PACK), ("our repo", REPO), ("competition", COMP),
               ("graph cache", CACHE)):
    print(f"  {lbl:<14} {v}")
if None in (PACK, REPO, COMP, CACHE):
    raise SystemExit("missing mount — CACHE needs claude-div-probe attached as a kernel source")
TRAIN = COMP / "train"
n_cached = len(list(CACHE.glob("cache_*.npz")))
print(f"  cached graphs  {n_cached}")
if n_cached == 0:
    raise SystemExit("no cache_*.npz found")

# These modules are what the run is testing, so a stale dataset version is a silent
# no-op rather than an error. Fail loudly instead.
for mod in ("repair.py", "anatomy.py"):
    if not (REPO / "pipeline" / mod).exists():
        raise SystemExit(f"the mounted repo predates pipeline/{mod} — push a new "
                         "dataset version before running this")

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")

# The official scorer: the cached graphs fork (0-10 per dataset), and purescore's
# division term is exact only fork-free (notes/24 §1).
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

md("""## 1. Anatomy and ablation, over the cached graphs

In a subprocess: the pack's wheels upgrade numpy under a process that has already
imported the old one (`notes/24` §1).
""")

code(r"""
WORKER = WORK / "run_anatomy.py"
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
from pipeline.repair import (cap_edge_length, close_gaps, linefit_smooth,
                             prune_isolated, single_parent_repair)

print("worker numpy", np.__version__, flush=True)

SCALE_DEFAULT = (1.625, 0.40625, 0.40625)

def caplen(g, sc):   return cap_edge_length(*g, scale=sc, max_um=14.0)
def parent(g, sc):   return single_parent_repair(*g, scale=sc)
def gapclose(g, sc): return close_gaps(*g, scale=sc, max_um=5.75,
                                       max_added_frac=0.038, max_added_abs=1650)
def smooth(g, sc):   return linefit_smooth(*g, window=2, weight=0.76, scale=sc,
                                           max_shift_um=3.2)
def prune(g, sc):    return prune_isolated(*g)

# Chain order: drop junk, fix merges, bridge holes, then smooth positions, then sweep up
# nodes the earlier stages orphaned. `prune` must be last or it sweeps too early.
CHAIN = [("caplen", caplen), ("parent", parent), ("gapclose", gapclose),
         ("smooth", smooth), ("prune", prune)]

ARMS = [("control", [])]
ARMS += [(f"only_{{name}}", [(name, fn)]) for name, fn in CHAIN]
ARMS += [("all", list(CHAIN))]
# Leave-one-out, because these interact: a solo delta flatters a repair that another
# stage would have made redundant anyway.
ARMS += [(f"all_minus_{{name}}", [s for s in CHAIN if s[0] != name])
         for name, _ in CHAIN]
print(f"{{len(ARMS)}} arms: {{[a for a, _ in ARMS]}}", flush=True)

names = sorted(p.stem[len("cache_"):] for p in CACHE.glob("cache_*.npz"))
names = [n for n in names if (TRAIN / f"{{n}}.geff").exists()]
print(f"{{len(names)}} cached graphs with ground truth available", flush=True)

h = Harness(data_dir=TRAIN, cache_dir=None)
ROWS = {{a: [] for a, _ in ARMS}}
NODES = {{a: 0 for a, _ in ARMS}}
ANAT = {{"control": [], "all": []}}
PER = {{}}

def apply_chain(g, steps, sc):
    for _, fn in steps:
        g = fn(g, sc)
    return g

for name in names:
    t0 = time.time()
    z = np.load(CACHE / f"cache_{{name}}.npz")
    base = (z["t"], z["zyx"], z["edges"])
    sc = read_scale(TRAIN / f"{{name}}.zarr")
    gt = read_geff(TRAIN / f"{{name}}.geff")

    line = []
    for arm, steps in ARMS:
        g = apply_chain(base, steps, sc) if steps else base
        row = h.score_graph(name, Tracks(g[0], g[1], g[2]))
        ROWS[arm].append(row)
        NODES[arm] += int(len(g[0]))
        if arm in ANAT:
            a = edge_anatomy(g[0], g[1], g[2], gt.t, gt.zyx, gt.edges, scale=sc)
            ANAT[arm].append(a)
            # PREDICTION 2: the buckets must account for every GT edge, per dataset.
            tot = sum(a[k] for k in BUCKETS)
            if tot != a["n_gt_edges"]:
                raise SystemExit(f"{{name}}/{{arm}}: buckets sum {{tot}} != "
                                 f"{{a['n_gt_edges']}} GT edges")
        line.append((arm, row.get("adj_edge_jaccard", float("nan")), len(g[0])))
    PER[name] = {{arm: {{"adj": float(v), "nodes": int(n)}} for arm, v, n in line}}
    ctrl = line[0][1]
    best = max(line[1:], key=lambda r: (r[1] if r[1] == r[1] else -9))
    print(f"  {{name:<24}} control {{ctrl:.4f}}  best {{best[0]}} {{best[1]:.4f}} "
          f"({{best[1]-ctrl:+.4f}})  {{time.time()-t0:.0f}}s", flush=True)

print()
print("=" * 78, flush=True)
print("PART A — anatomy of the edge loss (control graphs)", flush=True)
print("=" * 78, flush=True)
A = summarise_anatomy(ANAT["control"])
n = A["n_gt_edges"]
print(f"  {{'bucket':<16}}{{'count':>10}}{{'share of GT edges':>20}}", flush=True)
for k in BUCKETS:
    print(f"  {{k:<16}}{{A[k]:>10,}}{{100.0*A[k]/n:>19.2f}}%", flush=True)
print(f"  {{'TOTAL':<16}}{{n:>10,}}", flush=True)
print(flush=True)
print(f"  of the mislinks: source already linked {{A['source_busy']:,}}, "
      f"target already claimed {{A['target_busy']:,}}", flush=True)
print(f"  REACHABLE by graph repair (gap + mislink): {{100.0*A['reachable']:.2f}}% "
      f"of all GT edges", flush=True)
print(f"  edge_jaccard from these counts: {{A['edge_jaccard']:.4f}}", flush=True)

A2 = summarise_anatomy(ANAT["all"])
print(flush=True)
print("  after the full repair chain:", flush=True)
for k in BUCKETS:
    print(f"    {{k:<16}}{{A2[k]:>10,}}  ({{A2[k]-A[k]:+,}})", flush=True)

out = {{"arms": [a for a, _ in ARMS], "datasets": names,
       "summary": {{a: summarise(ROWS[a]) for a, _ in ARMS if ROWS[a]}},
       "nodes": NODES, "per_dataset": PER,
       "anatomy": {{"control": A, "all": A2}},
       "anatomy_per_dataset": {{"control": ANAT["control"]}}}}
(WORK / "edge_anatomy.json").write_text(json.dumps(out, indent=2, default=float))
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

md("""## 2. Part B, and the four predictions graded

A failure here is a result. Predictions 3 and 4 exist specifically so that a repair that
looks good for the wrong reason gets caught.
""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "edge_anatomy.json").read_text())
S, NODES, PER = D["summary"], D["nodes"], D["per_dataset"]
ARMS, DS = D["arms"], D["datasets"]
base = S["control"]
print(f"{len(DS)} datasets, {len(ARMS)} arms\n")

# A .geff without `estimated_number_of_nodes` makes the multiplier, and therefore every
# adjusted score, NaN. NaN comparisons are False, so an ungraded prediction would print
# PASS on missing data — the exact failure this project keeps hitting. Detect it once.
EXACT = base["score"] == base["score"]
if not EXACT:
    print("!! adj_edge_jaccard is NaN: the node budget was unreadable, so the score and")
    print("   multiplier columns are undefined. The anatomy and edge_jaccard columns are")
    print("   still exact. Predictions that depend on the budget are NOT GRADED below,")
    print("   rather than being scored against NaN.\n")

print(f"{'arm':<20}{'score':>9}{'delta':>9}{'adj_edge':>10}{'edge_J':>9}"
      f"{'nodes':>11}{'d_nodes':>10}")
print("-" * 78)
for a in ARMS:
    s = S.get(a)
    if not s:
        continue
    print(f"{a:<20}{s['score']:>9.4f}{s['score']-base['score']:>+9.4f}"
          f"{s['adj_edge_jaccard']:>10.4f}{s['edge_jaccard']:>9.4f}"
          f"{NODES[a]:>11,}{NODES[a]-NODES['control']:>+10,}")

print()
print("=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

# 1 ---------------------------------------------------------------------------------
print("\n1. the control reproduces claude_div_probe's 0.8806 +- 0.0005")
ok1 = EXACT and abs(base["score"] - 0.8806) <= 0.0005
if not EXACT:
    print("   NOT GRADED — the score column is NaN, see the note above.")
else:
    print(f"   control = {base['score']:.4f}   ->  {'PASS' if ok1 else 'FAIL'}")
if EXACT and not ok1:
    print("   Same cached graphs, same scorer, so this should be exact. A miss means the")
    print("   cache or the scoring path differs and NOTHING below is readable — including")
    print(f"   any apparent gain. ({len(DS)} datasets here vs 24 in div_probe; if those")
    print("    differ, that alone explains it and the rest still stands.)")

# 2 ---------------------------------------------------------------------------------
# Enforced inside the worker per dataset — reaching here at all means it held.
print("\n2. the buckets account for every GT edge, per dataset")
A = D["anatomy"]["control"]
tot = sum(A[k] for k in ("tp", "fn_gap", "fn_mislink", "fn_detect", "fn_nonconsec"))
print(f"   pooled {tot:,} vs {A['n_gt_edges']:,} GT edges   ->  "
      f"{'PASS' if tot == A['n_gt_edges'] else 'FAIL'}")
print("   (also asserted per dataset in the worker — a leak would have aborted the run)")

# 3 ---------------------------------------------------------------------------------
print("\n3. prune_isolated is non-negative on EVERY dataset")
# prune_isolated touches only the node count, so `adj` is the ONLY column that can move.
# With a NaN budget there is nothing to compare — and `nan < x` is False, so grading it
# anyway would report a clean PASS on absent data.
deltas = [(d, PER[d]["only_prune"]["adj"] - PER[d]["control"]["adj"]) for d in DS]
gradable = [(d, v) for d, v in deltas if v == v]
neg = [(d, v) for d, v in gradable if v < -1e-9]
if len(gradable) < len(DS):
    print(f"   NOT GRADED — {len(DS)-len(gradable)}/{len(DS)} datasets have a NaN "
          "budget, and prune moves only the adjusted score.")
else:
    print(f"   {len(DS)-len(neg)}/{len(DS)} non-negative  ->  "
          f"{'PASS' if not neg else 'FAIL'}")
if neg:
    for d, v in sorted(neg, key=lambda x: x[1])[:5]:
        print(f"     {d:<26}{v:+.4f}")
    print("   A node with no incident edge cannot carry edge Jaccard, so a negative here")
    print("   means the index remap in prune_isolated is rewiring edges. Fix before")
    print("   reading anything else — every other arm runs prune as its last step.")

# 4 ---------------------------------------------------------------------------------
print("\n4. linefit_smooth gains less than +0.005")
# Falls back to edge_jaccard when the budget is missing: smoothing moves positions, so it
# acts on the edge term directly and that column is exact either way.
key = "score" if EXACT else "edge_jaccard"
d_smooth = S["only_smooth"][key] - base[key]
print(f"   only_smooth {d_smooth:+.4f} on `{key}`   ->  "
      f"{'PASS' if d_smooth < 0.005 else 'FAIL'}")
if d_smooth >= 0.005:
    print("   Node recall is 0.995, so position repair should have almost nothing to")
    print("   recover. A larger gain means that reading is wrong and the detector's")
    print("   localisation — not its recall — is a live lever. Say so and re-plan.")

# what the anatomy implies -----------------------------------------------------------
print()
print("=" * 78)
gap, mis, det = A["fn_gap"], A["fn_mislink"], A["fn_detect"]
worst = max((("gap closing", gap), ("motion relink", mis), ("detection", det)),
            key=lambda kv: kv[1])
n = A["n_gt_edges"]
print(f"DOMINANT FAILURE: {worst[0]} — {worst[1]:,} of {n:,} GT edges "
      f"({100.0*worst[1]/n:.1f}%)")
print(f"  gap {100.0*gap/n:.1f}%   mislink {100.0*mis/n:.1f}%   "
      f"undetected {100.0*det/n:.1f}%")
print(f"  reachable by ANY graph repair: {100.0*A['reachable']:.1f}%")
# `max` over NaN keys returns whichever arm happens to come first, which would name an
# arbitrary winner with total confidence. Rank on the exact column instead.
best = max((a for a in ARMS if a != "control"),
           key=lambda a: (S[a][key] if S[a][key] == S[a][key] else float("-inf")))
print(f"  best arm this run: {best} at {S[best][key]:.4f} "
      f"({S[best][key]-base[key]:+.4f} on `{key}`)")
if worst[0] == "detection":
    print("  -> The pack's detector is the ceiling. Graph repair cannot reach this;")
    print("     bank the wins above and stop building repairs.")
else:
    print(f"  -> Run 2 is {worst[0]}. If that is motion relink, it needs a re-prediction")
    print("     caching candidate edges WITH edge_prob — their relink uses a learned")
    print("     bonus of 0.78, and the current cache has no probabilities in it.")
print("=" * 78)
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
