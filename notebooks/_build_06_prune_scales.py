"""Build notebooks/06_prune_scales.ipynb."""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "06_prune_scales.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Experiment #5 — prune instead of cap, then tune the scales

Two things `notes/11` left on the table.

**The cap is the wrong instrument.** At `sep 4.5` it buys +0.058 of budget multiplier by
destroying −0.026 of edge Jaccard, because it truncates detections *before* linking and
cannot tell a spurious peak from a real cell. The gate rejected it anyway: it helps `6bba`
(+0.039) and hurts `44b6` (−0.029), and the hidden test is a **different pair of embryos**.

`sep4.5` **without** the cap still holds the best detection+linking quality we have ever
measured — **edge Jaccard 0.7195** — and is only **145,338 nodes (10.1 %) over budget**.
Remove those without touching an edge and the score is 0.7195, a further **+0.024**.

`prune_isolated_nodes` does exactly that: it drops detections that ended up with **no edge
at all**. Budget cost, zero edge contribution. Implemented since `notes/06`, never measured.
It is *informed* — it acts after linking, on evidence — where the cap is blind.

**And the scales are untuned.** The public rule-based ladder goes plain DoG **0.682** →
tuned scales **0.791** → multi-scale **0.824**. We are at 0.6957 using their published
*defaults*, `(1.5, 4.0)` and `(2.5, 6.0)`. Roughly **0.10** sits in tuning the same
detector.

## Pre-registered

1. **Pruning beats capping**, from the same `sep4.5` detector. *Falsified if* pruning
   scores no better than `sep4.5_cap` (0.6957) — which would mean the isolated nodes were
   carrying bipartite matches that linked nodes then lose.
2. **Pruning removes ≥5 % of nodes while costing <0.01 of edge Jaccard.** The mechanism
   claim, separate from the score. *Falsified if* edge Jaccard falls as much as the cap's
   −0.026, meaning "isolated" nodes were not actually inert.
3. **Some scale pair beats `(1.5,4.0)+(2.5,6.0)` by >0.01.** *Falsified if* the default
   pair is already at the optimum — which would mean the 0.10 gap to their tuned rung comes
   from something other than scales.

> `notes/11` §1: `05` picked its cap setting from the pooled score while the gate had just
> printed REJECT. The selector below reads `Verdict.promote`, not the number.
""")

code(r"""
# --- deps -------------------------------------------------------------------
import subprocess, sys

def pip_install(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode

print("installing geff + zarr ...")
pip_install(["geff", "zarr"])
""")

code(r"""
import sys, os, time, json, hashlib
from pathlib import Path

import numpy as np

WORK = Path("/kaggle/working")

def find_dir(is_match, roots, max_depth=5):
    # Breadth-first over the Kaggle mounts; never descends into .zarr/.geff.
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


REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                [WORK, "/kaggle/input"])
if REPO is None:
    raise SystemExit("Could not find harness/ and pipeline/. Add the project dataset as an input.")
sys.path.insert(0, str(REPO))

from harness import Harness, gate
from pipeline.classical import Config, estimated_total_nodes, make_predictor

import inspect
for need, where in (("fold_by", Harness.__init__), ("prune_isolated_nodes", Config)):
    have = (need in inspect.signature(where).parameters if where is not Config
            else need in Config.__dataclass_fields__)
    print(f"code has {need}: {have}")
    if not have:
        raise SystemExit(
            f"The uploaded snapshot lacks `{need}`. Re-upload the current repo as the\\n"
            "Kaggle dataset — otherwise this notebook cannot run the arms it claims to."
        )

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data.")
TRAIN, TEST = COMP / "train", COMP / "test"

CACHE = WORK / "cache"; CACHE.mkdir(exist_ok=True, parents=True)
train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})
print("project:", REPO, "| train:", len(train_names))
""")

code(r"""
SUBSET_SIZE = 60

def stable_key(n):
    return int(hashlib.sha1(n.encode()).hexdigest(), 16)

by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)

SUBSET = []
for pfx, names in sorted(by_prefix.items()):
    k = round(SUBSET_SIZE * len(names) / len(train_names))
    SUBSET += sorted(names, key=stable_key)[:k]
SUBSET = sorted(SUBSET)
assert len(SUBSET) == 60, f"subset drifted ({len(SUBSET)}) — comparison invalid"

h = Harness(data_dir=TRAIN, cache_dir=CACHE)
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
prefixes = {f: {n.split("_")[0] for n in v} for f, v in folds.items()}
print("folds:", {f: len(v) for f, v in sorted(folds.items())}, prefixes)
assert all(len(p) == 1 for p in prefixes.values()), "folds are NOT leave-one-embryo-out"

BUDGETS = {n: estimated_total_nodes(TRAIN / f"{n}.zarr") for n in SUBSET}
BUDGETS = {k: v for k, v in BUDGETS.items() if v}
print(f"subset budget {sum(BUDGETS.values()):,.0f} nodes over {len(BUDGETS)} datasets")

SCALES2 = [(1.5, 4.0), (2.5, 6.0)]
BEST_05_CAP   = 0.6957    # sep4.5 + cap  (gate REJECTED: 44b6 -0.0286)
BEST_05_NOCAP = 0.6671    # sep4.5 no cap, edge_J 0.7195 -- the quality to preserve

def run(name, cfg):
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg, budgets=BUDGETS), arm=name, names=SUBSET,
                     verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    print(f"{name:<26} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  nodes={n:>10,}  ratio={np.mean(ratios):+.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return res

results = {}
""")

md("""## 1. Prune vs cap

Same detector, three ways of getting the node count down: not at all, blindly (cap), and
on evidence (prune). `base` reproduces `05`'s `sep4.5_nocap` — if it does not land on
0.6671 the run is not comparable and everything below is suspect.
""")

code(r"""
BASE = dict(detector="dog", min_separation_um=4.5, dog_rel_threshold=0.005,
            dog_scales=SCALES2)

results["base_nocap"]   = run("base_nocap",   Config(**BASE, budget_fill=None))
results["cap"]          = run("cap",          Config(**BASE, budget_fill=1.0))
results["prune"]        = run("prune",        Config(**BASE, budget_fill=None,
                                                     prune_isolated_nodes=True))
results["prune_plus_cap"] = run("prune_plus_cap", Config(**BASE, budget_fill=1.0,
                                                         prune_isolated_nodes=True))

drift = results["base_nocap"].score - BEST_05_NOCAP
print(f"\nreproduction check: base_nocap {results['base_nocap'].score:.4f} "
      f"vs 05's 0.6671 (drift {drift:+.4f})")
if abs(drift) > 0.005:
    print("!! the baseline moved — something differs from 05; treat the rest with caution")
""")

code(r"""
print("=== prediction 1: pruning beats capping ===")
# A bare `>` calls a delta of 1e-9 a win. Per-movie spread here is +-0.14, so anything
# under a few thousandths is not a result -- report it as INCONCLUSIVE, not CONFIRMED.
MARGIN = 0.005
p, c = results["prune"], results["cap"]
d = p.score - c.score
verdict = ("CONFIRMED" if d > MARGIN else
           "FALSIFIED" if d < -MARGIN else f"INCONCLUSIVE (|delta| < {MARGIN})")
print(f"  prune {p.score:.4f}   cap {c.score:.4f}   delta {d:+.4f}")
print(f"  -> {verdict}")
print()
print(gate(c, p))

print("\n=== prediction 2: pruning is cheap in edge quality ===")
b = results["base_nocap"]
nb = sum(r["num_pred_nodes"] for r in b.rows.values())
npn = sum(r["num_pred_nodes"] for r in p.rows.values())
nc = sum(r["num_pred_nodes"] for r in c.rows.values())
dropped = (nb - npn) / nb
d_edge = p.summary["edge_jaccard"] - b.summary["edge_jaccard"]
c_edge = c.summary["edge_jaccard"] - b.summary["edge_jaccard"]
print(f"  nodes: base {nb:,} -> prune {npn:,} ({dropped:.1%} removed)")
print(f"         base {nb:,} -> cap   {nc:,} ({(nb-nc)/nb:.1%} removed)")
print(f"  edge_J cost:  prune {d_edge:+.4f}   cap {c_edge:+.4f}")
ok2 = dropped >= 0.05 and abs(d_edge) < 0.01
print(f"  -> {'CONFIRMED' if ok2 else 'FALSIFIED'}")
if not ok2 and abs(d_edge) >= 0.01:
    print("     Isolated nodes were NOT inert — they were winning bipartite matches.")
""")

md("""## 2. Tune the scales

The largest classical lever left: ~0.10 between our configuration and the public tuned
rung, on the same detector. Carried forward with whichever node-count treatment §1
**promoted** — reading the verdict, not the pooled number (`notes/11` §1).
""")

code(r"""
v_prune = gate(results["cap"], results["prune"])
if v_prune.promote:
    KEEP = dict(budget_fill=None, prune_isolated_nodes=True); KEEP_NAME = "prune"
else:
    v_cap = gate(results["base_nocap"], results["cap"])
    if v_cap.promote:
        KEEP = dict(budget_fill=1.0, prune_isolated_nodes=False); KEEP_NAME = "cap"
    else:
        KEEP = dict(budget_fill=None, prune_isolated_nodes=False); KEEP_NAME = "neither"
print(f"carrying forward: {KEEP_NAME}  (gate verdict, not the pooled score)\n")

SCALE_SETS = {
    "default":  [(1.5, 4.0), (2.5, 6.0)],
    "tight":    [(1.0, 2.5), (1.5, 4.0)],
    "wide":     [(2.0, 5.0), (3.0, 8.0)],
    "spread":   [(1.5, 4.0), (3.0, 8.0)],
    "verywide": [(2.5, 6.0), (4.0, 10.0)],
}
for name, sc in SCALE_SETS.items():
    if name == "default":
        results["scales_default"] = results["prune" if KEEP_NAME == "prune" else
                                            "cap" if KEEP_NAME == "cap" else "base_nocap"]
        print(f"{'scales_default':<26} (reusing the section 1 arm)")
        continue
    results[f"scales_{name}"] = run(f"scales_{name}",
        Config(detector="dog", min_separation_um=4.5, dog_rel_threshold=0.005,
               dog_scales=sc, **KEEP))
""")

code(r"""
print("=== prediction 3: some scale pair beats the default by >0.01 ===")
base_score = results["scales_default"].score
for name in SCALE_SETS:
    r = results[f"scales_{name}"]
    print(f"  {name:<10} {SCALE_SETS[name]}  -> {r.score:.4f}  ({r.score-base_score:+.4f})")
best_scale = max(SCALE_SETS, key=lambda n: results[f"scales_{n}"].score)
gain = results[f"scales_{best_scale}"].score - base_score
# prediction 3 already carries its own >0.01 margin, which is above the noise floor
print(f"\n  best: {best_scale} at {results[f'scales_{best_scale}'].score:.4f} ({gain:+.4f})")
print(f"  -> {'CONFIRMED' if gain > 0.01 else 'FALSIFIED'}")
if best_scale != "default":
    print()
    print(gate(results["scales_default"], results[f"scales_{best_scale}"]))

print("\n" + "=" * 66)
best_name = max(results, key=lambda k: results[k].score)
print(f"BEST THIS RUN: {best_name} -> {results[best_name].score:.4f}")
for lbl, v in (("05 best (sep4.5+cap, gate REJECTED)", BEST_05_CAP),
               ("04 (dog_matched)", 0.6760), ("03 (intensity)", 0.5790),
               ("public: plain DoG", 0.682), ("public: tuned scales", 0.791),
               ("public: multi-scale", 0.824)):
    print(f"  {lbl:<38} {v:.4f}")

payload = {"best": best_name, "keep": KEEP_NAME,
           "arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                        for kk, vv in v.summary.items()} for k, v in results.items()},
           "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
                     for k, v in results.items()},
           "scale_sets": {k: [list(p) for p in v] for k, v in SCALE_SETS.items()}}
(WORK / "prune_scale_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/prune_scale_results.json — send it back with the log.")
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")

for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        stripped = "\n".join("pass  # shell" if l.strip().startswith("!") else l
                             for l in src.splitlines())
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} syntax error: {e}\n---\n{src}")
print("all code cells parse as valid Python")
