"""Build notebooks/08_budget.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/08_budget.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Experiment #7 — can we read the node budget off the image?

`07` broke the embryo split and took CV to **0.7098**. `notes/13` §3 then found the
uncomfortable part: **two thirds of that gain is the node-budget multiplier, and the
budget is unreadable at test time.**

| `adaptive_1.2` − `champion_04` | |
|---|---|
| edge quality (micro edge_J 0.7053 → 0.7169) | +0.0111 |
| node-budget multiplier (0.9584 → 0.9901) | +0.0227 |
| total | +0.0338 |

`estimated_number_of_nodes` lives in GEFF metadata. **`test/` ships images only — 4
`.zarr`, zero `.geff`** (`notes/05` §0). Both `budget_fill` and `adaptive_separation` key
off that number, and with it missing `predict_dataset` falls back to the fixed separation —
i.e. straight back to `champion_04`. So `adaptive_1.2` scores 0.7098 on CV and would score
0.6760 on the leaderboard. The scorer still applies the multiplier from the hidden ground
truth; we are graded on a target we cannot see.

This notebook asks whether we can predict it from the pixels, and measures the one
configuration we could actually submit today.

## Pre-registered

1. **`champion_04 + prune` passes the gate over `champion_04`.** Pruning needs no
   metadata, so this is the best *submittable* number and we have never measured it.
   `notes/12` §1 measured pruning at +0.0065 edge Jaccard at sep 4.5; this asks whether it
   survives at sep 6.0. *Falsified if* it regresses either embryo.
2. **The budget is predictable across embryos** — median |relative error| **< 30 %**
   under leave-one-embryo-out, **and better than predicting the training embryo's median**.
   The null matters: if a constant does as well, the features are worthless.
   *Falsified* by either half.
3. **`adaptive_predicted` keeps at least half of the oracle gain**, measured as
   `(predicted − champion+prune) / (oracle − champion+prune)`. *Falsified* below 0.5.

A note on what the folds can and cannot say. There are two embryos, so
leave-one-embryo-out fits on one and tests on the other — a harsh test with n=2. It is
still the right one: the hidden set is two *different* embryos, so a model that only works
within an embryo is worth nothing here.
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
import pipeline.classical as pc
from pipeline.classical import Config, estimated_total_nodes, make_predictor

# 08 needs helpers that 07's snapshot did not have. Fail loudly rather than three
# hours later -- 05 lost a whole run to exactly this.
for need in ("budget_features", "open_movie", "load_frame"):
    if not hasattr(pc, need):
        raise SystemExit(f"The uploaded snapshot lacks pipeline.classical.{need} — re-upload the repo.")
budget_features = pc.budget_features
print("helpers present: budget_features, open_movie, load_frame")

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
assert len(BUDGETS) == 60, f"missing budgets for {60 - len(BUDGETS)} datasets"
SCALES2 = [(1.5, 4.0), (2.5, 6.0)]
BASE = dict(detector="dog", dog_rel_threshold=0.005, dog_scales=SCALES2, footprint="ball")
GATED_CHAMPION = 0.6760      # 04: DoG sep 6.0, no cap, no prune
ADAPTIVE_ORACLE = 0.7098     # 07: adaptive target 1.2 + prune, with the TRUE budget

results = {}
def run(name, cfg, budgets=None):
    t0 = time.time()
    res = h.evaluate(make_predictor(cfg, budgets=BUDGETS if budgets is None else budgets),
                     arm=name, names=SUBSET, verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    mult = s["adj_edge_jaccard"] / s["edge_jaccard"] if s["edge_jaccard"] else float("nan")
    print(f"{name:<24} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"mult={mult:.4f}  recall={s['node_recall']:.3f}  nodes={n:>10,}  "
          f"ratio={np.mean(ratios):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    results[name] = res
    return res
""")

md("""## 1. The best configuration we could actually submit

Everything promoted in `06` and `07` reads the node budget. Pruning does not — it only
drops predicted nodes that ended up with no edge, which is a property of our own output.
So `champion_04 + prune` is the honest submittable baseline, and it has never been scored.
""")

code(r"""
run("champion_04", Config(min_separation_um=6.0, **BASE))
drift = results["champion_04"].score - GATED_CHAMPION
print(f"\nreproduction: {results['champion_04'].score:.4f} vs 04's {GATED_CHAMPION:.4f} "
      f"(drift {drift:+.4f})")
if abs(drift) > 0.005:
    print("!! the champion moved — compare everything below with that in mind")

run("champion_plus_prune", Config(min_separation_um=6.0, prune_isolated_nodes=True, **BASE))

print("\n=== prediction 1: pruning survives at sep 6.0 and gates ===")
print(gate(results["champion_04"], results["champion_plus_prune"]))
""")

md("""## 2. Predicting the budget from the image

`pipeline.classical.budget_features` reads a few frames per dataset and returns detector
and intensity statistics — no `.geff` touched anywhere. Note the docstring's warning: the
**raw peak count runs backwards**, rising as the field empties, because
`dog_abs_percentile` floors the detector at a percentile of the volume. `frac_fg` and the
absolute-cut counts are the monotone ones. All are offered; the fit decides.
""")

code(r"""
t0 = time.time()
FEATS = {}
for i, n in enumerate(SUBSET):
    FEATS[n] = budget_features(TRAIN / f"{n}.zarr", Config(min_separation_um=6.0, **BASE),
                               frac_frames=(0.25, 0.5, 0.75), ref_seps=(4.0, 8.0, 16.0))
    if i % 10 == 0 or i == len(SUBSET) - 1:
        print(f"  features {i+1:>3}/{len(SUBSET)}  ({time.time()-t0:.0f}s)", flush=True)

FEAT_NAMES = ["n_sep4", "n_sep8", "n_sep16", "nstrong_sep4", "nstrong_sep8",
              "nstrong_sep16", "mean_int", "frac_fg"]

def design(names):
    # log1p on counts and log on the intensity fractions: the budget spans 20.8x across
    # datasets (recon §9), so the relationship is multiplicative, not additive.
    rows = []
    for n in names:
        f = FEATS[n]
        rows.append([1.0] + [np.log1p(f[k]) if k.startswith("n") else
                             np.log(max(f[k], 1e-6)) for k in FEAT_NAMES])
    return np.array(rows, float)

def target(names):
    # budget PER FRAME -- what adaptive_separation actually aims at.
    return np.array([np.log(BUDGETS[n] / FEATS[n]["T"]) for n in names], float)

print("\ncorrelation of each feature with log(budget/frame), pooled and per embryo:")
y_all = target(SUBSET)
X_all = design(SUBSET)
for j, k in enumerate(FEAT_NAMES, start=1):
    r_all = np.corrcoef(X_all[:, j], y_all)[0, 1]
    per = []
    for f, names in sorted(folds.items()):
        yy, xx = target(names), design(names)[:, j]
        per.append(f"{list(prefixes[f])[0]}={np.corrcoef(xx, yy)[0,1]:+.2f}")
    print(f"  {k:<14} pooled {r_all:+.3f}   " + "  ".join(per))
""")

code(r"""
def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]

# Leave-one-embryo-out: fit on the OTHER embryo only. With two embryos this is the
# harshest available test and the one that matches the hidden set.
pred_log, null_log = {}, {}
for f, held in sorted(folds.items()):
    train_names_f = [n for n in SUBSET if n not in held]
    beta = fit(design(train_names_f), target(train_names_f))
    yhat = design(held) @ beta
    med = float(np.median(target(train_names_f)))   # the null: the other embryo's median
    for n, v in zip(held, yhat):
        pred_log[n] = float(v)
        null_log[n] = med

def rel_err(pred):
    e = []
    for n in SUBSET:
        truth = BUDGETS[n] / FEATS[n]["T"]
        e.append(abs(np.exp(pred[n]) - truth) / truth)
    return np.array(e)

e_model, e_null = rel_err(pred_log), rel_err(null_log)
print("=== prediction 2: the budget is predictable across embryos ===")
print(f"  regression   median |rel err| {np.median(e_model):.1%}   mean {e_model.mean():.1%}")
print(f"  null (median) median |rel err| {np.median(e_null):.1%}   mean {e_null.mean():.1%}")
for f, held in sorted(folds.items()):
    idx = [SUBSET.index(n) for n in held]
    print(f"    {list(prefixes[f])[0]}: model {np.median(e_model[idx]):.1%}  "
          f"null {np.median(e_null[idx]):.1%}")
ok = np.median(e_model) < 0.30 and np.median(e_model) < np.median(e_null)
print(f"  -> {'CONFIRMED' if ok else 'FALSIFIED'} "
      f"(needs <30% AND better than the null)")

PRED_BUDGETS = {n: float(np.exp(pred_log[n]) * FEATS[n]["T"]) for n in SUBSET}
NULL_BUDGETS = {n: float(np.exp(null_log[n]) * FEATS[n]["T"]) for n in SUBSET}
print("\n  true vs predicted budget, five widest misses:")
worst = sorted(SUBSET, key=lambda n: -abs(np.log(PRED_BUDGETS[n] / BUDGETS[n])))[:5]
for n in worst:
    print(f"    {n:<22} true {BUDGETS[n]:>8,.0f}   predicted {PRED_BUDGETS[n]:>8,.0f}   "
          f"x{PRED_BUDGETS[n]/BUDGETS[n]:.2f}")
""")

md("""## 3. Does a predicted budget still buy the gain?

A 25 % error on the budget is not the same as a 25 % error on the score — the calibration
loop only uses the budget to pick a separation, and separation is a blunt knob. So measure
it, do not reason about it. Three arms: the oracle from `07`, the regression, and the null
(the training embryo's median budget applied to every dataset).
""")

code(r"""
ADAPT = dict(min_separation_um=6.0, adaptive_separation=True, adaptive_target=1.2,
             prune_isolated_nodes=True, **BASE)

run("adaptive_oracle", Config(**ADAPT))
d = results["adaptive_oracle"].score - ADAPTIVE_ORACLE
print(f"  reproduction of 07's adaptive_1.2: {results['adaptive_oracle'].score:.4f} "
      f"vs {ADAPTIVE_ORACLE:.4f} (drift {d:+.4f})")

run("adaptive_predicted", Config(**ADAPT), budgets=PRED_BUDGETS)
run("adaptive_null", Config(**ADAPT), budgets=NULL_BUDGETS)
""")

code(r"""
base = results["champion_plus_prune"].score
oracle_gain = results["adaptive_oracle"].score - base
pred_gain = results["adaptive_predicted"].score - base
null_gain = results["adaptive_null"].score - base
print("=== prediction 3: a predicted budget keeps at least half the oracle gain ===")
print(f"  champion+prune (submittable baseline) {base:.4f}")
print(f"  oracle budget    {results['adaptive_oracle'].score:.4f}  ({oracle_gain:+.4f})")
print(f"  predicted budget {results['adaptive_predicted'].score:.4f}  ({pred_gain:+.4f})")
print(f"  null budget      {results['adaptive_null'].score:.4f}  ({null_gain:+.4f})")
if oracle_gain <= 0:
    print("  -> VACUOUS: the oracle itself does not beat champion+prune, so there is "
          "no gain to retain. Prediction 3 cannot be scored; read prediction 1 instead.")
else:
    keep = pred_gain / oracle_gain
    print(f"  retention {keep:.0%}  -> {'CONFIRMED' if keep >= 0.5 else 'FALSIFIED'}")
    print(f"  regression over null: {results['adaptive_predicted'].score - results['adaptive_null'].score:+.4f}")
""")

code(r"""
# The gate, chained against the CURRENT champion rather than a fixed constant.
# 07 named adaptive_wide champion on the strength of a comparison against a
# two-experiments-old baseline, after failing the incremental gate (notes/13 §2).
print("=" * 82)
CHAIN = ["champion_04", "champion_plus_prune", "adaptive_predicted"]
# Walk the chain FIRST and record what each arm was actually gated against. An arm
# that fails does not become the reference for the next one -- the champion simply
# does not move. Printing "vs the previous row" instead would let a rejected arm
# launder the next one through an easier comparison, which is 07's mistake with a
# different shape.
champion, gated_against = "champion_04", {}
for arm in CHAIN[1:]:
    gated_against[arm] = champion
    if gate(results[champion], results[arm]).promote:
        champion = arm
# The budget-reading arms can never be submitted; show them against the submittable
# baseline for scale, and keep them out of the chain.
REFERENCE_ONLY = {"adaptive_oracle": "champion_plus_prune",
                  "adaptive_null": "champion_plus_prune"}

print(f"{'arm':<24} {'SCORE':>8} {'edge_J':>8} {'mult':>7} {'vs ref':>9} {'gate':>9}  ref")
for k, r in sorted(results.items(), key=lambda kv: -kv[1].score):
    ref = gated_against.get(k) or REFERENCE_ONLY.get(k)
    mult = r.summary["adj_edge_jaccard"] / r.summary["edge_jaccard"]
    if ref is None:
        print(f"{k:<24} {r.score:>8.4f} {r.summary['edge_jaccard']:>8.4f} {mult:>7.4f} "
              f"{'-':>9} {'-':>9}  (chain start)")
        continue
    v = gate(results[ref], r)
    tag = ("PROMOTE" if v.promote else "reject") if k in gated_against else "ref only"
    print(f"{k:<24} {r.score:>8.4f} {r.summary['edge_jaccard']:>8.4f} {mult:>7.4f} "
          f"{r.score - results[ref].score:>+9.4f} {tag:>9}  {ref}")

print(f"\nsubmittable champion: {champion} -> {results[champion].score:.4f}")
print("  (chained: gated against the standing champion, which only moves on a PROMOTE)")
for arm in REFERENCE_ONLY:
    print(f"  reads the node budget, so NOT submittable: {arm} {results[arm].score:.4f}")

payload = {
    "arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                 for kk, vv in v.summary.items()} for k, v in results.items()},
    "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
              for k, v in results.items()},
    "gates": {a: {"vs": b, "promote": bool(gate(results[b], results[a]).promote),
                  "fold_deltas": gate(results[b], results[a]).fold_deltas}
              for a, b in [("champion_plus_prune", "champion_04"),
                           ("adaptive_predicted", "champion_plus_prune"),
                           ("adaptive_oracle", "champion_plus_prune"),
                           ("adaptive_null", "champion_plus_prune"),
                           ("adaptive_predicted", "adaptive_null")]},
    "budget_pred": {"median_rel_err": float(np.median(e_model)),
                    "median_rel_err_null": float(np.median(e_null)),
                    "true": {n: float(BUDGETS[n]) for n in SUBSET},
                    "pred": PRED_BUDGETS},
    "features": FEATS,
    "champion": champion,
}
(WORK / "budget_results.json").write_text(json.dumps(payload, indent=2, default=str))
print(f"\nWrote {WORK}/budget_results.json — send it back with the log.")
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
