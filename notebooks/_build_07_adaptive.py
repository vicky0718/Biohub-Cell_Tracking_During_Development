"""Build notebooks/07_adaptive.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/07_adaptive.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Experiment #6 — adapt density per dataset, and gate what `06` never gated

Two loose ends from `notes/12`.

**Nothing since `04` has passed the gate.** `05`'s 0.6957 and `06`'s wide scales were
rejected on the embryo split, and `06`'s headline **0.7072 (`prune + cap`) was never
compared to anything at all**. The best *gate-passing* configuration is still `04`'s
**0.6760**. Everything here is gated against it.

**The embryo split is the real obstacle, and it is three for three:**

| comparison | `44b6` | `6bba` |
|---|---|---|
| `05`: cap vs no cap | −0.0286 | +0.0389 |
| `06`: prune vs cap | +0.0378 | −0.0209 |
| `06`: wide vs default scales | −0.0597 | +0.0468 |

All three say the same thing: **`44b6` wants a denser detection field, `6bba` a sparser
one** — consistent with median budgets of ~327 vs ~97 cells/frame. There is no global
setting that serves both, and the hidden test is two *different* embryos.

`adaptive_separation` is the principled answer: pick `min_separation_um` **per dataset** so
the detector naturally emits about `estimated_number_of_nodes / T` per frame. Dense crops
get tight separation, sparse crops wide, with no embryo assumption anywhere. Unlike the cap
it hits the count by changing the detector rather than discarding its output — `06`
measured blind truncation at −0.0264 edge Jaccard while pruning *gained* +0.0065.

## Pre-registered

1. **`prune + cap` passes the gate against `04`'s 0.6760.** *Falsified if* it regresses
   either embryo — which would mean `06`'s headline was a pooled artefact.
2. **Adaptive separation beats the best fixed setting, and does it without an embryo
   split.** The specific claim is the fold deltas: *falsified if* `44b6` and `6bba` still
   move in opposite directions, meaning per-dataset budget is not the axis they disagree on.
3. **Pruning helps in every arm it is added to.** `notes/12` §1 says it is a
   matching-quality fix, not a budget tactic. *Falsified if* any paired prune/no-prune
   comparison here shows a loss.
""")

code(r"""
import subprocess, sys
def pip_install(pkgs):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
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

for need in ("adaptive_separation", "prune_isolated_nodes", "footprint"):
    if need not in Config.__dataclass_fields__:
        raise SystemExit(f"The uploaded snapshot lacks Config.{need} — re-upload the repo.")
print("config fields present: adaptive_separation, prune_isolated_nodes, footprint")

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data.")
TRAIN = COMP / "train"
CACHE = WORK / "cache"; CACHE.mkdir(exist_ok=True, parents=True)
train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})

SUBSET_SIZE = 60
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)
SUBSET = []
for pfx, names in sorted(by_prefix.items()):
    SUBSET += sorted(names, key=stable_key)[:round(SUBSET_SIZE * len(names) / len(train_names))]
SUBSET = sorted(SUBSET)
assert len(SUBSET) == 60, f"subset drifted ({len(SUBSET)})"

h = Harness(data_dir=TRAIN, cache_dir=CACHE)
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
prefixes = {f: {n.split("_")[0] for n in v} for f, v in folds.items()}
assert all(len(p) == 1 for p in prefixes.values()), "folds are NOT leave-one-embryo-out"
print("folds:", {f: len(v) for f, v in sorted(folds.items())}, prefixes)

BUDGETS = {n: estimated_total_nodes(TRAIN / f"{n}.zarr") for n in SUBSET}
BUDGETS = {k: v for k, v in BUDGETS.items() if v}
SCALES2 = [(1.5, 4.0), (2.5, 6.0)]
SCALES_WIDE = [(2.0, 5.0), (3.0, 8.0)]
GATED_CHAMPION = 0.6760      # 04, DoG sep 6.0, no cap, no prune -- the only gated config

def run(name, cfg):
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg, budgets=BUDGETS), arm=name, names=SUBSET, verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    print(f"{name:<24} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"recall={s['node_recall']:.3f}  nodes={n:>10,}  ratio={np.mean(ratios):+.3f}  "
          f"({time.time()-t0:.0f}s)", flush=True)
    return res

results = {}
""")

md("""## 1. The gated champion, and the composite `06` never tested""")

code(r"""
results["champion_04"] = run("champion_04",
    Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
           dog_scales=SCALES2, footprint="ball"))
drift = results["champion_04"].score - GATED_CHAMPION
print(f"\nreproduction: {results['champion_04'].score:.4f} vs 04's {GATED_CHAMPION:.4f} "
      f"(drift {drift:+.4f})")
if abs(drift) > 0.005:
    print("!! the champion moved — compare everything below with that in mind")

results["prune_plus_cap"] = run("prune_plus_cap",
    Config(detector="dog", min_separation_um=4.5, dog_rel_threshold=0.005,
           dog_scales=SCALES2, footprint="ball", budget_fill=1.0,
           prune_isolated_nodes=True))

print("\n=== prediction 1: does 06's headline actually pass the gate? ===")
print(gate(results["champion_04"], results["prune_plus_cap"]))
""")

md("""## 2. Adaptive per-dataset separation

The answer to the embryo split, if it is one. Pruning is on throughout — `notes/12` §1
found it *raises* edge Jaccard, so there is no arm where you would want it off.
""")

code(r"""
for tgt in (0.8, 1.0, 1.2):
    results[f"adaptive_{tgt}"] = run(f"adaptive_{tgt}",
        Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
               dog_scales=SCALES2, footprint="ball", adaptive_separation=True,
               adaptive_target=tgt, prune_isolated_nodes=True))

print("\n=== prediction 2: adaptive beats the best fixed setting, without an embryo split ===")
best_ad = max((k for k in results if k.startswith("adaptive_")), key=lambda k: results[k].score)
print(f"best adaptive arm: {best_ad} -> {results[best_ad].score:.4f}\n")
v = gate(results["champion_04"], results[best_ad])
print(v)
signs = [d > 0 for d in v.fold_deltas.values()]
split = len(set(signs)) > 1
print(f"\n  fold deltas: {({f: round(d, 4) for f, d in v.fold_deltas.items()})}")
print(f"  embryos still split: {split}  -> "
      f"{'FALSIFIED (per-dataset budget is not the axis)' if split else 'CONFIRMED'}")
""")

code(r"""
print("=== prediction 3: pruning helps wherever it is added ===")
results["adaptive_noprune"] = run("adaptive_noprune",
    Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
           dog_scales=SCALES2, footprint="ball", adaptive_separation=True,
           adaptive_target=float(best_ad.split("_")[1]), prune_isolated_nodes=False))
a, b = results["adaptive_noprune"], results[best_ad]
print(f"  adaptive without prune {a.score:.4f}   with prune {b.score:.4f}   "
      f"delta {b.score-a.score:+.4f}")
print(f"  edge_J {a.summary['edge_jaccard']:.4f} -> {b.summary['edge_jaccard']:.4f}")
print(f"  -> {'CONFIRMED' if b.score > a.score else 'FALSIFIED'}")

# does the wide-scale direction survive once density is adaptive?
results["adaptive_wide"] = run("adaptive_wide",
    Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
           dog_scales=SCALES_WIDE, footprint="ball", adaptive_separation=True,
           adaptive_target=float(best_ad.split("_")[1]), prune_isolated_nodes=True))
print("\nwide scales under adaptive density:")
print(gate(results[best_ad], results["adaptive_wide"]))
""")

code(r"""
print("=" * 66)
print(f"{'arm':<24} {'SCORE':>8} {'edge_J':>8} {'vs champion':>12} {'gate':>9}")
for k, r in sorted(results.items(), key=lambda kv: -kv[1].score):
    v = gate(results["champion_04"], r)
    tag = "-" if k == "champion_04" else ("PROMOTE" if v.promote else "reject")
    print(f"{k:<24} {r.score:>8.4f} {r.summary['edge_jaccard']:>8.4f} "
          f"{r.score - results['champion_04'].score:>+12.4f} {tag:>9}")

promoted = [k for k, r in results.items()
            if k != "champion_04" and gate(results["champion_04"], r).promote]
print(f"\narms that PASS the gate: {promoted or 'NONE'}")
if promoted:
    best = max(promoted, key=lambda k: results[k].score)
    # measure against the champion AS REPRODUCED IN THIS RUN, not the recorded constant --
    # otherwise any drift in the baseline is silently charged to the new arm.
    ref = results["champion_04"].score
    print(f"new gated champion: {best} -> {results[best].score:.4f} "
          f"({results[best].score - ref:+.4f} over the champion reproduced here, "
          f"which itself sits {ref - GATED_CHAMPION:+.4f} from 04's recorded {GATED_CHAMPION:.4f})")
else:
    print(f"gated champion unchanged: 04 at {GATED_CHAMPION:.4f}")

payload = {"arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                        for kk, vv in v.summary.items()} for k, v in results.items()},
           "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
                     for k, v in results.items()},
           "fold_deltas": {k: gate(results["champion_04"], v).fold_deltas
                           for k, v in results.items() if k != "champion_04"},
           "promoted": promoted}
(WORK / "adaptive_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/adaptive_results.json — send it back with the log.")
""")

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try: ast.parse("\n".join("pass" if l.strip().startswith("!") else l for l in src.splitlines()))
        except SyntaxError as e: raise SystemExit(f"cell {i}: {e}\n{src}")
print("all code cells parse as valid Python")
