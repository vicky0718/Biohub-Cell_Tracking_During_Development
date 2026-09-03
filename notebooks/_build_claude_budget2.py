"""Build notebooks/claude_budget2.ipynb — rank tracks under a per-dataset budget.

notes/51: fn_detect is now 583 against fn_mislink 226, so the graph side is bounded
at +0.020..+0.035 against a 0.034 gap. And the per-dataset node budget notes/04 §9
identified has never been used -- we still ship one global DET_THRESHOLD, while
r35's linker.py carries max_pred_nodes beside rank_tracks_by_geometry.

This is the THIRD selection rule. claude_budget cut with an NMS radius and
claude_topk with a confidence threshold; both cut at the DETECTION stage, before
anything knew which detections would end up in a good track, and both destroyed
recall. Cutting after linking removes a junk track's nodes AND its false-positive
edges together.

Derived from the divsweep builder, which already re-solves cached graphs with no GPU
and grades per embryo.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_budget2.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# The third selection rule: rank tracks under a per-dataset budget

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
score = adj_edge_jaccard + 0.1*div_J,   adj = max(0, edge_J * (1 - 0.1*ratio))
ratio = (N_pred - N_est) / N_est        floor at 0, NO ceiling
```

`notes/51` established two things that make this run the obvious next one.

**Detection is now the ceiling.** At the shipped chain `fn_detect` is 583 (4.21% of GT
edges) against `fn_mislink` 226 (1.63%) — the exact inverse of `notes/26`. Perfect linking
is worth +0.020 to +0.035 against a 0.034 gap, so the graph side cannot get there alone.

**We have never used the per-dataset budget.** `notes/04` §9 said it plainly — *"the two
datasets that are the leaderboard have node budgets 11× apart, 64 vs 698 cells per frame. A
detector with one global threshold cannot serve both"* — and we still ship one global
`DET_THRESHOLD`. r35's `linker.py`, read for the first time in `notes/51`, carries
`max_pred_nodes` beside `rank_tracks_by_geometry`: *"Pivot H — drop short false tracks to cut
|V̂|/φ penalty"*, *"R11 — rank tracks by link geometry (tight long tracks) under budget"*.

## Why this is not the two thinning runs that already failed

Both previous attempts cut at the **detection** stage, before anything knew which detections
would end up in a good track:

```
claude_budget    pool_kernel_um, an NMS radius     node_recall 0.983 -> 0.537   notes/46
claude_topk      det_threshold, a confidence cut   0.901 -> 0.863 on the LB     notes/48,49
```

**Cutting after linking is different in kind.** Dropping a junk track removes its nodes — a
budget gain — *and* its false-positive edges — a Jaccard gain. A detector-stage cut removes
nodes that a surviving track needed, which is precisely why both runs above destroyed recall.

The mechanism is confirmed from two directions: `notes/45` derived it, and forum thread
739018 (Michael Hernandez, then TWEAK) independently read it off the released evaluator.
TWEAK's synthetic case is the sharp statement — *"with edge TP/FP/FN fixed at 90/5/5,
reducing predicted nodes from 100 to 50 changed adjusted-edge Jaccard from 0.900 to 0.945."*
A node that is not an endpoint of a kept edge is pure cost.

## The grid

`pipeline/repair.rank_budget_prune`, applied as a final stage after the shipped chain
(`inc/g2sp6`). Every arm re-solves nothing — one ILP solve per dataset, then post-processing.

```
isolated          drop nodes in no edge at all. Ignores the budget; should be free.
geometry @ f      rank by span, tie-break on median step length. Keep until N_est * f.
length   @ f      span alone -- the ablation that says whether tightness carries anything.
f in {1.0, 0.9, 0.8, 0.7}
```

`N_est` is each dataset's own `estimated_number_of_nodes`, read from its `.geff`.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** The `none` arm equals `claude_divsweep`'s `inc/g2sp6` — total 0.9188,
   `div_J` 0.1154, 1,443 forks. Otherwise the chain has moved and nothing else is readable.
2. **`isolated` is non-negative on both embryos.** An edgeless node cannot contribute a TP
   edge, so dropping it should be free. If this *loses*, my reading of the metric is wrong
   and predictions 3-5 mean nothing.
3. **Some budget arm beats `none` by more than 0.0015** (`notes/44`'s floor). This is the
   crux. Failing it closes the last cheap direction and leaves only the detector.
4. **`geometry` beats `length` at the same factor.** If tightness carries nothing, r35's
   ranking is just "keep long tracks" and the mechanism is simpler than advertised.
5. **The best arm holds in sign on BOTH embryos.** `notes/49`: the test set is a third pair
   of embryos, and a pooled win across crops of two is not evidence about a third. This is
   the prediction that 0.901 -> 0.863 was missing.

*Node counts and `total_node_ratio` are reported per arm regardless, so even a clean failure
tells us where the budget actually sits — which no run so far has measured directly.*
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
from pipeline.repair import (close_gaps, linefit_smooth, prune_short_tracks,
                             rank_budget_prune)
from harness.tracks import read_estimated_nodes
print("worker numpy", np.__version__, flush=True)

def shipped(g, sc):
    # claude_divsweep's inc/g2sp6, the arm we submit: gaps(2) -> smooth -> prune(6).
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# (label, mode, factor). factor multiplies the dataset's estimated_number_of_nodes;
# None means the stage ignores the budget entirely.
POST = [("none", None, None), ("isolated", "isolated", None)]
for _f in (1.0, 0.9, 0.8, 0.7):
    POST.append((f"geometry{{_f}}", "geometry", _f))
    POST.append((f"length{{_f}}", "length", _f))

def apply_post(g, sc, mode, factor, n_est):
    g = shipped(g, sc)
    if mode is not None:
        tgt = float("nan") if factor is None else n_est * factor
        g = rank_budget_prune(g[0], g[1], g[2], n_target=tgt, scale=sc, mode=mode,
                              keep_division_components=True)
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
print(f"{{len(POST)}} budget arms on one solve", flush=True)

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
        for key, mode, factor in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, mode, factor, n_est)
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
    base = PER[name]["none"]
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
           "post": [{{"label": p[0], "mode": p[1], "factor": p[2]}} for p in POST],
           "summary": {{l: summarise(ROWS[l]) for l in LABELS if ROWS[l]}},
           "anatomy": {{l: summarise_anatomy(ANAT[l]) for l in LABELS if ANAT[l]}},
           "forks": FORKS, "edges": EDGES, "per_dataset": PER}}
    (WORK / "budget2.json").write_text(json.dumps(out, indent=2, default=float))

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

md("""## 2. Grading — the budget, per embryo""")

code(r"""
import numpy as np, json
D = json.loads((WORK / "budget2.json").read_text())
S, F, E = D["summary"], D["forks"], D["edges"]
ARMS, DS, PER = D["arms"], D["datasets"], D["per_dataset"]
NONE = "none"
REF_TOTAL, REF_DIVJ, REF_FORKS = 0.9188, 0.1154, 1443   # claude_divsweep's inc/g2sp6
print(f"{len(DS)} datasets | {len(ARMS)} arms")

def emb(n): return n.split("_")[0]
EMB = sorted({emb(n) for n in DS})
print("embryos: " + ",  ".join(f"{e} n={sum(emb(n) == e for n in DS)}" for e in EMB))

def rows(arm, names=None):
    return [PER[n][arm] for n in (names or DS) if n in PER and arm in PER[n]]

# Full-set figures come from purescore.summarise -- div_J micro-averaged, adj_edge
# weight-averaged by TP+FP+FN. Recomputing would substitute unweighted means for both
# (notes/47). Per-embryo subsets have no stored weight, so those are means, and are only
# ever used as DELTAS between two arms on the same datasets.
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

print(f"\n{'arm':<16}{'total':>9}{'adj_edge':>10}{'edge_J':>9}{'div_J':>8}"
      f"{'nodes':>10}{'ratio':>9}{'mult':>7}{'forks':>8}")
print("-" * 86)
for a in ARMS:
    r = mean(a, "ratio")
    print(f"{a:<16}{total(a):>9.4f}{adj(a):>10.4f}"
          f"{S.get(a,{}).get('edge_jaccard',float('nan')):>9.4f}{divJ(a):>8.4f}"
          f"{mean(a,'nodes'):>10,.0f}{r:>9.3f}{1 - 0.1 * r:>7.3f}{F.get(a,0):>8,}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

# ---- 1. reproduction -------------------------------------------------------------
print("\n1. the `none` arm reproduces claude_divsweep's inc/g2sp6")
ok1 = (abs(total(NONE) - REF_TOTAL) < 0.002 and abs(divJ(NONE) - REF_DIVJ) < 0.010
       and abs(F.get(NONE, 0) - REF_FORKS) <= 20)
print(f"   total {total(NONE):.4f} (want {REF_TOTAL})   div_J {divJ(NONE):.4f} "
      f"(want {REF_DIVJ})   forks {F.get(NONE,0):,} (want {REF_FORKS:,})"
      f"  ->  {'PASS' if ok1 else 'FAIL'}")
if not ok1:
    print("   The chain has moved since claude_divsweep. Nothing below is comparable.")

# ---- 2. isolated is free ---------------------------------------------------------
print("\n2. `isolated` is non-negative on BOTH embryos (an edgeless node cannot be a TP)")
iso = "isolated"
if iso not in ARMS:
    ok2 = False; print("   NOT GRADED — arm missing")
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][iso]["adj"] - PER[n][NONE]["adj"] for n in ns
             if iso in PER.get(n, {}) and NONE in PER[n]]
        per[e] = sum(d) / len(d) if d else float("nan")
    ok2 = all(v > -0.0005 for v in per.values())
    print("   " + "   ".join(f"{e} {v:+.4f}" for e, v in per.items())
          + f"   (pooled {adj(iso) - adj(NONE):+.4f})  ->  {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        print("   Dropping edgeless nodes LOSES score. That contradicts the metric as")
        print("   read in notes/45 and forum 739018, so the reading is wrong and")
        print("   predictions 3-5 cannot be interpreted.")

# ---- 3. THE CRUX -----------------------------------------------------------------
print(f"\n3. some budget arm beats `{NONE}` by more than 0.0015 (notes/44's floor)")
cand = [a for a in ARMS if a != NONE]
best = max(cand, key=lambda a: total(a) if total(a) == total(a) else -9) if cand else NONE
gain = total(best) - total(NONE)
ok3 = gain > 0.0015
print(f"   best {best} {total(best):.4f} vs {NONE} {total(NONE):.4f}"
      f"   {gain:+.4f}  ->  {'PASS' if ok3 else 'FAIL'}")
print(f"   decomposed:  adj_edge {adj(best) - adj(NONE):+.4f}"
      f"   0.1*div_J {0.1 * (divJ(best) - divJ(NONE)):+.4f}"
      f"   nodes {mean(best,'nodes') - mean(NONE,'nodes'):+,.0f}")
if not ok3:
    print("   Track-level ranking under a per-dataset cap is the THIRD selection rule,")
    print("   after uniform thinning (notes/46) and confidence thinning (notes/48,49).")
    print("   All three now fail. The node budget is not collectable by any rule we")
    print("   have, and the remaining gap is detection (notes/51: fn_detect 583).")

# ---- 4. does tightness carry anything? -------------------------------------------
print("\n4. `geometry` beats `length` at the same budget factor")
pairs = [(g, l) for g in ARMS if g.startswith("geometry")
         for l in ARMS if l == g.replace("geometry", "length")]
if not pairs:
    ok4 = False; print("   NOT GRADED — no matched pair")
else:
    wins = sum(total(g) > total(l) for g, l in pairs)
    for g, l in pairs:
        print(f"   {g:<16}{total(g):>9.4f}   vs {l:<16}{total(l):>9.4f}"
              f"   {total(g)-total(l):+.4f}")
    ok4 = wins > len(pairs) / 2
    print(f"   geometry wins {wins}/{len(pairs)}  ->  {'PASS' if ok4 else 'FAIL'}")
    if not ok4:
        print("   Tightness carries nothing; the rule is just 'keep long tracks'.")

# ---- 5. notes/49's rule ----------------------------------------------------------
print(f"\n5. the best arm holds in sign on BOTH embryos (n is {len(EMB)}, not {len(DS)})")
# An arm tied with the control has nothing to transfer, and reporting that as "wins on one
# embryo, loses on the other" borrows notes/49's language for a run that had no effect at
# all. claude_budget2 hit exactly this: `isolated` was byte-identical to `none`, every
# per-embryo delta was +0.0000, and `all(x > 0)` rejected the zeros as a sign disagreement.
if best == NONE or abs(total(best) - total(NONE)) < 1e-9:
    ok5 = False
    print("   NOT GRADED — no arm differs from the control"
          + ("" if best == NONE else f" ({best} is tied with it)"))
else:
    per = {}
    for e in EMB:
        ns = [n for n in DS if emb(n) == e]
        d = [PER[n][best]["adj"] - PER[n][NONE]["adj"] for n in ns
             if best in PER.get(n, {}) and NONE in PER[n]]
        per[e] = (sum(d) / len(d) if d else float("nan"),
                  divJ(best, ns) - divJ(NONE, ns), len(d))
    print(f"   {'embryo':<8}{'n':>4}{'adj delta':>12}{'div_J delta':>14}{'total':>10}")
    for e, (da, dj, n) in per.items():
        print(f"   {e:<8}{n:>4}{da:>+12.4f}{dj:>+14.4f}{da + 0.1 * dj:>+10.4f}")
    tots = [da + 0.1 * dj for da, dj, _ in per.values()]
    ok5 = len(tots) > 1 and (all(x > 0 for x in tots) or all(x < 0 for x in tots))
    print(f"   signs agree  ->  {'PASS' if ok5 else 'FAIL'}")
    if not ok5:
        print("   Wins on one embryo, loses on the other. The test set is a THIRD pair")
        print("   (notes/07 §3). This is the shape that cost 0.901 -> 0.863 in notes/49.")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4, ok5])}/5 predictions passed")
if ok1 and ok3 and ok5:
    print(f"SUBMITTABLE: {best} gains {gain:+.4f} and holds on both embryos.")
elif ok1 and ok2 and not ok3:
    print("CLOSED: all three selection rules fail. The budget is not collectable and")
    print("the remaining gap is detection (notes/51).")
elif ok1 and ok3 and not ok5:
    print("MEASURED, NOT SUBMITTABLE: real on train, does not agree across embryos.")
else:
    print("NOT COMPARABLE: reproduction or the metric reading failed; fix that first.")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
