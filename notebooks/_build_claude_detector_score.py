"""Build notebooks/claude_detector_score.ipynb."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_detector_score.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

# One list; the header and the grading cell are both generated from it (notes/18 §1).
PREDICTIONS = [
    ("The learned detector beats the champion on the real metric.",
     "Every learned number so far is node recall, which is a proxy. The champion is "
     "adaptive_predicted at 0.7070 (notes/14). Falsified if no UNet arm passes the gate.",
     "beats_champion"),
    ("The best cap is 1.0x of predicted budget.",
     "notes/18 §5: the under-budget bonus tops out at x1.1 and cannot repay the recall "
     "given up at 0.5x, while 1.5x buys almost no recall for a 5% multiplier penalty. "
     "Falsified if 0.8x or 1.2x scores higher than 1.0x.",
     "cap_1x_best"),
    ("Recall gains survive as edge-Jaccard gains.",
     "notes/04 §7 says linking is solved given detections, so better detection should "
     "convert. Falsified if micro edge Jaccard fails to rise even where recall did.",
     "recall_converts"),
]

md(r"""
# Phase 1b — score the learned detector on the real metric

Everything measured so far is **node recall**, a proxy. This runs the learned detector
through the actual scorer, against the standing champion.

The chain to beat (`notes/14`): `04` DoG 0.6760 -> +prune 0.6896 -> **adaptive_predicted
0.7070**, which is also the configuration that scored **0.752 on the leaderboard**.

Weights come from the `claude_detector_train` kernel as a **kernel data source**, not a
download: `kaggleusercontent.com` is refused by the agent container's egress proxy
(`notes/17` §4), so files can never be fetched out of a kernel — only logs.

**Folds stay leave-one-embryo-out, and each fold is scored by the model that never saw
it.** `44b6` datasets are predicted by the model trained on `6bba`, and vice versa. That
is the whole point of training two models, and getting it backwards would silently
manufacture a result.

Everything downstream of detection is unchanged and already gated: the budget regression
from `notes/14` (10.7 % median error), pruning, Hungarian linking. Only the detector moves.
""")

md("## Pre-registered\n\n" + "\n".join(
    f"{i+1}. **{c}** {w}" for i, (c, w, _) in enumerate(PREDICTIONS)))

code(r"""
import subprocess, sys, time

def sh(*args, **kw):
    try:
        return subprocess.run(args, capture_output=True, text=True, **kw)
    except (FileNotFoundError, OSError) as e:
        return subprocess.CompletedProcess(args, 127, "", str(e))

def pip_install(pkgs, extra=()):
    r = sh(sys.executable, "-m", "pip", "install", "-q", *extra, *pkgs)
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r.returncode == 0

gpu = sh("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").stdout.strip()
print(f"accelerator: {gpu or 'NONE'}")
if "P100" in gpu:
    print("P100 -> installing torch with sm_60 kernels ...")
    t0 = time.time()
    ok = pip_install(["torch==2.5.1"],
                     extra=("--index-url", "https://download.pytorch.org/whl/cu121"))
    print(f"  torch replacement {'ok' if ok else 'FAILED'} ({time.time()-t0:.0f}s)")
print("installing geff + zarr ...")
pip_install(["geff", "zarr"])
""")

code(r"""
import sys, os, time, json, hashlib
from pathlib import Path
import numpy as np
import torch

WORK = Path("/kaggle/working"); WORK.mkdir(parents=True, exist_ok=True)
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__}  device {DEV}")
if DEV.type == "cuda":
    try:
        _w = torch.nn.Conv3d(1, 4, 3, padding=1).to(DEV)
        _ = _w(torch.randn(2, 1, 8, 8, 8, device=DEV)).sum().item()
        torch.cuda.synchronize(); print("  GPU smoke test passed")
    except Exception as e:
        raise SystemExit(f"GPU present but unusable: {type(e).__name__}: {str(e)[:200]}")

def find_dir(is_match, roots, max_depth=5):
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
    raise SystemExit("Could not find harness/ and pipeline/.")
sys.path.insert(0, str(REPO))

from harness import Harness, gate
import pipeline.classical as pc
if "prob_fn" not in pc.predict_dataset.__code__.co_varnames:
    raise SystemExit("The uploaded snapshot's predict_dataset has no prob_fn — re-upload.")
from pipeline.classical import (Config, budget_features, estimated_total_nodes,
                                make_predictor, predict_dataset)
from pipeline.unet import UNet3D, predict_volume

# Weights arrive as a KERNEL data source. Bounded scan -- a recursive glob over
# /kaggle/input walks every .zarr chunk and costs minutes (notes/17 §4).
def find_weights(root="/kaggle/input", max_depth=5):
    found, stack = set(), [(Path(root), 0)]
    while stack:
        d, depth = stack.pop()
        try:
            kids = list(os.scandir(d))
        except (PermissionError, OSError, FileNotFoundError):
            continue
        for e in kids:
            if e.is_file() and e.name.startswith("claude_unet_") and e.name.endswith(".pt"):
                # resolve(): a symlinked mount reaches the same file by two paths, and an
                # unresolved set would load it twice and report phantom duplicates.
                found.add(Path(e.path).resolve())
        if depth < max_depth:
            stack += [(Path(e.path), depth + 1) for e in kids
                      if e.is_dir() and not e.name.endswith((".zarr", ".geff"))]
    return sorted(found)

wpaths = find_weights()
print(f"weight files found: {[p.name for p in wpaths] or 'NONE'}")
if not wpaths:
    raise SystemExit("No claude_unet_*.pt found. Add the claude_detector_train kernel "
                     "as a data source (Add Input -> Notebook Output).")

# A training kernel may save several losses x both folds, and attaching a kernel as a
# data source attaches ALL of its outputs -- there is no way to pick a subset. So this
# SELECTS a loss rather than refusing, which an earlier version did on the strength of an
# instruction ("attach only the winning loss's files") that cannot actually be carried out.
FORCE_LOSS = None       # set to "masked" / "pu" to override the automatic choice

ckpts = {}              # (loss, train_emb) -> (path, checkpoint)
for wp in wpaths:
    ck = torch.load(wp, map_location="cpu")
    key = (ck.get("loss", "?"), ck["train_emb"])
    if key in ckpts:
        # Same loss AND same fold twice means two runs of the same thing are attached;
        # that IS ambiguous and the winner would be whichever the scan reached last.
        raise SystemExit(f"Two checkpoints for {key}: {ckpts[key][0].name} and {wp.name}. "
                         "Attach exactly one training kernel version.")
    ckpts[key] = (wp, ck)
    print(f"  found {wp.name}: loss={key[0]}, trained on {key[1]}, "
          f"best_recall={ck.get('best_recall')} @epoch {ck.get('best_epoch')}, "
          f"DoG {ck.get('dog_recall')}")

embryos_needed = {e for _, e in ckpts}
by_loss = {}
for (ln, emb), (wp, ck) in ckpts.items():
    by_loss.setdefault(ln, {})[emb] = (wp, ck)
complete = {ln: v for ln, v in by_loss.items() if set(v) == embryos_needed}
if not complete:
    raise SystemExit(f"No loss covers every embryo. Have: "
                     f"{ {ln: sorted(v) for ln, v in by_loss.items()} }")

def margin(v):
    # Mean held-out recall ABOVE DoG across folds. DoG differs per embryo (0.7696 vs
    # 0.8776), so raw recall is not comparable between folds and the margin is.
    return float(np.mean([ck.get("best_recall", 0.0) - ck.get("dog_recall", 0.0)
                          for _, ck in v.values()]))

ranked = sorted(complete, key=lambda ln: -margin(complete[ln]))
CHOSEN = FORCE_LOSS or ranked[0]
if CHOSEN not in complete:
    raise SystemExit(f"FORCE_LOSS={FORCE_LOSS!r} does not cover every embryo.")
print(f"\nloss ranking by mean margin over DoG: "
      + ", ".join(f"{ln} {margin(complete[ln]):+.4f}" for ln in ranked))
print(f"CHOSEN LOSS: {CHOSEN}"
      + (" (forced)" if FORCE_LOSS else " (best mean margin)"))
# One loss for BOTH folds. Picking per embryo independently would score a chimera --
# two different models stitched across the folds of one number.
MODELS = {}
for emb, (wp, ck) in sorted(complete[CHOSEN].items()):
    m = UNet3D(base=ck.get("base", 16), depth=ck.get("depth", 3))
    m.load_state_dict(ck["state_dict"]); m.eval().to(DEV)
    MODELS[emb] = m
    print(f"  using {wp.name} for datasets NOT from {emb}")
if len(MODELS) < 2:
    raise SystemExit(f"Need a model per embryo; got {sorted(MODELS)}")

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
TRAIN = COMP / "train"
CACHE = WORK / "cache"; CACHE.mkdir(exist_ok=True, parents=True)
train_names = sorted({p.stem for p in TRAIN.glob("*.zarr")} & {p.stem for p in TRAIN.glob("*.geff")})

SUBSET_SIZE = 60
def stable_key(n): return int(hashlib.sha1(n.encode()).hexdigest(), 16)
by_prefix = {}
for n in train_names:
    by_prefix.setdefault(n.split("_")[0], []).append(n)
SUBSET = []
for pfx, ns in sorted(by_prefix.items()):
    SUBSET += sorted(ns, key=stable_key)[:round(SUBSET_SIZE * len(ns) / len(train_names))]
SUBSET = sorted(SUBSET)
assert len(SUBSET) == 60, f"subset drifted ({len(SUBSET)})"

h = Harness(data_dir=TRAIN, cache_dir=CACHE)
folds = {}
for n in SUBSET:
    folds.setdefault(h.fold_of(n), []).append(n)
prefixes = {f: {n.split("_")[0] for n in v} for f, v in folds.items()}
assert all(len(p) == 1 for p in prefixes.values()), "folds are NOT leave-one-embryo-out"
print("folds:", {f: len(v) for f, v in sorted(folds.items())}, prefixes)

SCALES2 = [(1.5, 4.0), (2.5, 6.0)]
BASE = dict(detector="dog", dog_rel_threshold=0.005, dog_scales=SCALES2, footprint="ball")
CHAMPION_CV = 0.7070          # notes/14: adaptive_predicted, and the 0.752 LB submission
results = {}
""")

md("""## 1. The budget regression, refit here

`notes/14` §2: a constant budget scores **0.0882 below doing nothing**, so this is refit at
runtime rather than carried as coefficients. Same features and same leave-one-embryo-out
fit that measured 10.7 % median error.
""")

code(r"""
CFG_FEAT = Config(min_separation_um=6.0, **BASE)
FRAC_FRAMES, REF_SEPS = (0.25, 0.5, 0.75), (4.0, 8.0, 16.0)
FEAT_NAMES = ["n_sep4", "n_sep8", "n_sep16", "nstrong_sep4", "nstrong_sep8",
              "nstrong_sep16", "mean_int", "frac_fg"]

t0 = time.time()
FEATS, BUDGETS = {}, {}
for i, n in enumerate(SUBSET):
    FEATS[n] = budget_features(TRAIN / f"{n}.zarr", CFG_FEAT,
                               frac_frames=FRAC_FRAMES, ref_seps=REF_SEPS)
    BUDGETS[n] = estimated_total_nodes(TRAIN / f"{n}.zarr")
    if i % 15 == 0 or i == len(SUBSET) - 1:
        print(f"  features {i+1:>3}/{len(SUBSET)} ({time.time()-t0:.0f}s)", flush=True)

def design(ns):
    return np.array([[1.0] + [np.log1p(FEATS[n][k]) if k.startswith("n")
                              else np.log(max(FEATS[n][k], 1e-6)) for k in FEAT_NAMES]
                     for n in ns], float)

def target(ns):
    return np.array([np.log(BUDGETS[n] / FEATS[n]["T"]) for n in ns], float)

PRED_BUDGETS = {}
for f, held in sorted(folds.items()):
    tr = [n for n in SUBSET if n not in held]
    beta = np.linalg.lstsq(design(tr), target(tr), rcond=None)[0]
    for n, v in zip(held, design(held) @ beta):
        PRED_BUDGETS[n] = float(np.exp(v) * FEATS[n]["T"])
err = np.array([abs(PRED_BUDGETS[n] - BUDGETS[n]) / BUDGETS[n] for n in SUBSET])
print(f"budget regression: median |rel err| {np.median(err):.1%} "
      f"(notes/14 measured 10.7%)")
""")

md("""## 2. Score the arms

Each dataset is predicted by the model trained on **the other** embryo. The assertion
below is not decoration — swapping these would leak the training embryo into its own fold
and inflate every number in the run.
""")

code(r"""
OTHER = {e: [x for x in MODELS if x != e][0] for e in MODELS}
print("scoring map (dataset embryo -> model trained on):",
      {e: OTHER[e] for e in sorted(OTHER)})
for e in OTHER:
    assert OTHER[e] != e, f"{e} would be scored by a model that trained on it"

# Why a miss happened, not just that it did. A prediction 9um from a GT node is BOTH a
# false negative and a wasted detection, and it is fixed by resolution or sub-voxel
# refinement -- completely different work from a cell the detector never saw at all.
# Recall alone cannot tell those apart, and neither can a 2x2 confusion matrix: at a fixed
# cap, node precision is pinned to recall, and "true negative" is 262k background voxels
# per frame.
from harness.tracks import read_geff
from harness.purescore import match_nodes
NEAR_UM = 14.0        # 2x the metric's 7um match radius
DIAG = {}

def diagnose(name, data_dir, graph, scale):
    g = read_geff(Path(data_dir) / f"{name}.geff")
    if not len(g.t) or not len(graph.t):
        return
    matched = match_nodes(graph.t, graph.zyx, g.t, g.zyx, scale=scale, max_distance=7.0)
    hit = set(matched[matched >= 0].tolist())
    s = np.asarray(scale, float)
    near = far = 0
    for t in np.unique(g.t):
        gi = np.flatnonzero((g.t == t))
        miss = [i for i in gi if i not in hit]
        if not miss:
            continue
        pj = np.flatnonzero(graph.t == t)
        if not len(pj):
            far += len(miss); continue
        d = np.linalg.norm((g.zyx[miss][:, None] - graph.zyx[pj][None]) * s, axis=2).min(1)
        near += int((d <= NEAR_UM).sum()); far += int((d > NEAR_UM).sum())
    DIAG.setdefault(CURRENT_ARM, []).append(
        {"name": name, "n_gt": int(len(g.t)), "matched": len(hit),
         "near_miss": near, "far_miss": far, "n_pred": int(len(graph.t))})

CURRENT_ARM = ""

def make_unet_predictor(cfg, budgets):
    def _fn(name, data_dir):
        emb = name.split("_")[0]
        model = MODELS[OTHER[emb]]
        def prob_fn(vol):
            return predict_volume(model, vol, DEV)
        graph = predict_dataset(Path(data_dir) / name, cfg, verbose=False,
                                est_total_nodes=budgets.get(name), prob_fn=prob_fn)
        diagnose(name, data_dir, graph, SCALE_UM)
        return graph
    return _fn

def make_dog_predictor(cfg, budgets):
    base = make_predictor(cfg, budgets=budgets)
    def _fn(name, data_dir):
        graph = base(name, data_dir)
        diagnose(name, data_dir, graph, SCALE_UM)
        return graph
    return _fn

SCALE_UM = (1.625, 0.40625, 0.40625)     # GT lives in FULL-resolution voxels

def run(name, cfg, predictor=None):
    global CURRENT_ARM
    CURRENT_ARM = name
    t0 = time.time()
    fn = predictor if predictor is not None else make_dog_predictor(cfg, PRED_BUDGETS)
    res = h.evaluate(fn, arm=name, names=SUBSET, verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    ratios = [r["total_node_ratio"] for r in res.rows.values()
              if r["total_node_ratio"] == r["total_node_ratio"]]
    mult = s["adj_edge_jaccard"] / s["edge_jaccard"] if s["edge_jaccard"] else float("nan")
    print(f"{name:<20} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"mult={mult:.4f}  recall={s['node_recall']:.3f}  nodes={n:>9,}  "
          f"ratio={np.mean(ratios):+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    results[name] = res
    return res

# The champion, reproduced here so drift cannot be charged to the new arm.
run("champion", Config(min_separation_um=6.0, adaptive_separation=True,
                       adaptive_target=1.2, prune_isolated_nodes=True, **BASE))
drift = results["champion"].score - CHAMPION_CV
print(f"\nreproduction: {results['champion'].score:.4f} vs {CHAMPION_CV:.4f} "
      f"(drift {drift:+.4f})")
if abs(drift) > 0.005:
    print("!! the champion moved — read everything below against THIS number")
""")

code(r"""
# For a learned detector the density knob is the CAP, not the separation: the probability
# map is thresholded at a floor (unet_threshold=1e-6) and the per-frame budget does the
# work. notes/18 §5 puts the operating point at 1.0x; 0.8x and 1.2x bracket it.
for fill in (0.8, 1.0, 1.2):
    cfg = Config(min_separation_um=6.0, budget_fill=fill,
                 prune_isolated_nodes=True, **BASE)
    run(f"unet_cap{fill}", cfg, predictor=make_unet_predictor(cfg, PRED_BUDGETS))
""")

code(r"""
PREDICTIONS = """ + json.dumps([[c, w, k] for c, w, k in PREDICTIONS]) + r"""
ref = results["champion"]
print("=" * 84)
print(f"{'arm':<20}{'SCORE':>9}{'edge_J':>9}{'mult':>8}{'recall':>8}{'vs champ':>10}{'gate':>9}")
for k, r in sorted(results.items(), key=lambda kv: -kv[1].score):
    m = r.summary["adj_edge_jaccard"] / r.summary["edge_jaccard"]
    tag = "-" if k == "champion" else ("PROMOTE" if gate(ref, r).promote else "reject")
    print(f"{k:<20}{r.score:>9.4f}{r.summary['edge_jaccard']:>9.4f}{m:>8.4f}"
          f"{r.summary['node_recall']:>8.3f}{r.score - ref.score:>+10.4f}{tag:>9}")

# Edge confusion matrix, split. edge_jaccard = TP/(TP+FP+FN) hides which way it fails:
# FP >> FN means spurious links, FN >> FP means cells never found, and those want
# opposite fixes. Eight experiments have optimised the collapsed number without once
# looking at the split.
print("\n" + "=" * 84)
print(f"{'arm':<20}{'edge TP':>10}{'FP':>9}{'FN':>9}{'precision':>11}{'recall':>9}{'FP/FN':>8}")
for k, r in sorted(results.items(), key=lambda kv: -kv[1].score):
    tp = sum(x["edge_tp"] for x in r.rows.values())
    fp = sum(x["edge_fp"] for x in r.rows.values())
    fn = sum(x["edge_fn"] for x in r.rows.values())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    print(f"{k:<20}{tp:>10,}{fp:>9,}{fn:>9,}{prec:>11.4f}{rec:>9.4f}{fp/max(fn,1):>8.2f}")

print("\nnode misses: is the detector blind, or does it see the cell and miss the 7um radius?")
print(f"{'arm':<20}{'GT':>9}{'matched':>9}{'near<=14um':>12}{'far>14um':>10}{'near share':>12}")
for k in sorted(DIAG, key=lambda k: -results[k].score if k in results else 0):
    rows = DIAG[k]
    g = sum(x["n_gt"] for x in rows); mt = sum(x["matched"] for x in rows)
    nr = sum(x["near_miss"] for x in rows); fr = sum(x["far_miss"] for x in rows)
    print(f"{k:<20}{g:>9,}{mt:>9,}{nr:>12,}{fr:>10,}{nr/max(nr+fr,1):>12.1%}")
print("  a high near share means localisation, not detection -- resolution or sub-voxel")
print("  refinement, NOT more capacity. a high far share means cells never seen at all.")

print("\nper-dataset node recall spread (the pooled mean hides this):")
for k, r in sorted(results.items(), key=lambda kv: -kv[1].score):
    rc = sorted(x["node_recall"] for x in r.rows.values()
                if x["node_recall"] == x["node_recall"])
    if rc:
        q = lambda f: rc[min(len(rc) - 1, int(f * len(rc)))]
        print(f"  {k:<20} min {rc[0]:.3f}  p25 {q(.25):.3f}  median {q(.5):.3f}  "
              f"p75 {q(.75):.3f}  max {rc[-1]:.3f}")

unet = {k: v for k, v in results.items() if k.startswith("unet_")}
promoted = [k for k, v in unet.items() if gate(ref, v).promote]
best = max(unet, key=lambda k: unet[k].score)

verdicts = {
    "beats_champion": bool(promoted),
    "cap_1x_best": best == "unet_cap1.0",
    "recall_converts": (unet[best].summary["edge_jaccard"] > ref.summary["edge_jaccard"]
                        if unet[best].summary["node_recall"] > ref.summary["node_recall"]
                        else True),
}
for i, (claim, why, key) in enumerate(PREDICTIONS):
    print(f"\n=== prediction {i+1}: {claim} ===")
    print(f"    {why}")
    print(f"  -> {'CONFIRMED' if verdicts[key] else 'FALSIFIED'}")

print(f"\narms passing the gate: {promoted or 'NONE'}")
if promoted:
    champ = max(promoted, key=lambda k: results[k].score)
    v = gate(ref, results[champ])
    print(f"NEW CHAMPION: {champ} -> {results[champ].score:.4f} "
          f"({results[champ].score - ref.score:+.4f} over the champion reproduced here)")
    print(f"  fold deltas: { {f: round(d, 4) for f, d in v.fold_deltas.items()} }")
    print("  -> submittable; build claude_submit_unet against this configuration")
else:
    print(f"champion unchanged: {ref.score:.4f}. Do NOT submit a learned arm.")

payload = {"arms": {k: {kk: (None if isinstance(vv, float) and vv != vv else vv)
                        for kk, vv in v.summary.items()} for k, v in results.items()},
           "nodes": {k: int(sum(r["num_pred_nodes"] for r in v.rows.values()))
                     for k, v in results.items()},
           "fold_deltas": {k: gate(ref, v).fold_deltas for k, v in results.items()
                           if k != "champion"},
           "promoted": promoted, "verdicts": verdicts,
           "budget_median_rel_err": float(np.median(err)),
           "edge_confusion": {k: {"tp": sum(x["edge_tp"] for x in v.rows.values()),
                                  "fp": sum(x["edge_fp"] for x in v.rows.values()),
                                  "fn": sum(x["edge_fn"] for x in v.rows.values())}
                              for k, v in results.items()},
           "node_misses": DIAG}
blob = json.dumps(payload, indent=2, default=str)
(WORK / "claude_detector_score_results.json").write_text(blob)
print("\n===== RESULTS JSON BEGIN =====")
print(blob)
print("===== RESULTS JSON END =====")
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
