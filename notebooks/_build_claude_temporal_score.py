"""Build notebooks/claude_temporal_score.ipynb — score the temporal model end to end.

Derived from `_build_claude_detector_refine.py`. Four arms, and the two comparisons they
support are deliberately kept separate:

  r0_movie -> r0_perframe   what the train/serve normalisation skew was costing
  r0_perframe -> r1_perframe  what temporal input is worth, at MATCHED normalisation

Running r1 only against r0_movie would bundle the two and attribute both to the temporal
input — the notes/18 §1 mistake, made harder to spot because the number would look good.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_temporal_score.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

PREDICTIONS = [
    ("Fixing the normalisation skew is worth something.",
     "Until now every learned arm trained on a per-frame percentile rescale and was SERVED "
     "the whole-movie one -- 5.7x off in range on the test volumes. Measured as: "
     "r0_perframe scores above r0_movie. Falsified if it does not, which would mean the "
     "network is insensitive to input scale and the skew was never costing anything.",
     "norm_fix_helps"),
    ("Temporal input beats single-frame at matched normalisation, through coherence.",
     "**Registered AFTER the training run, which already measured this at the paired-recall "
     "level and found +0.0063 on one fold and -0.0152 on the other -- noise around zero. So "
     "this is now expected to FAIL, and is scored anyway because paired recall on 5-frame "
     "eval runs is not the leaderboard metric on full movies, and the two have already "
     "disagreed once (notes/21 §1).** Measured as: r1_perframe beats r0_perframe on SCORE "
     "**and** on temporal position, on the datasets both radii can score out-of-fold. "
     "Falsified if the score moves without the position moving -- that would mean something "
     "other than coherence produced it.",
     "temporal_beats_single"),
    ("It is still not enough to pass the champion gate.",
     "notes/22 projected 0.7006 at full DoG-parity coherence from a 0.6556 baseline -- "
     "short of the champion's 0.7128 edge Jaccard unless edge PRECISION moves too. "
     "Recorded so that a pass draws scrutiny rather than celebration. Falsified if any "
     "learned arm scores above 0.7070.",
     "still_short"),
]

md(r"""
# Step 1, scored — temporal input and the normalisation fix, separated

`claude_temporal_train` produced checkpoints at `temporal_radius` 0 and 1, selected on
**paired recall**. This scores them on the real metric, against the champion.

## Four arms, and why not three

| arm | radius | `prob_input_norm` | datasets | what it isolates |
|---|---|---|---|---|
| `champion` | — | — | all 60 | drift control, reproduces CV 0.7070 in-run |
| `r1_movie` | 1 | `movie` | all 60 | the notes/21–22 serving path, reproduced |
| `r1_perframe` | 1 | `per_frame` | all 60 | **minus `r1_movie` = the normalisation skew** |
| `r0_perframe` | 0 | `per_frame` | subset | **vs `r1_perframe` = temporal input** |

Two changes landed between `notes/22` and here: the temporal input, and a train/serve
normalisation fix. Measuring them against a single baseline would bundle them and credit
both to whichever is named — the `notes/18` §1 failure, and *harder* to catch here because
a bundled number looks better than either change deserves. So each comparison moves exactly
one thing.

**Why `r1` carries the normalisation comparison rather than `r0`.** The training run's
sub-DoG guard refused to save `r0/44b6` (paired 0.8529 against DoG's 0.8550 — a −0.0021
tie), so `r=0` has an out-of-fold model for only one embryo. `r=1` covers both, so it is
the radius that can hold the full-subset comparison. The temporal comparison then runs on
the datasets where **both** radii are out-of-fold, with `r1_perframe` re-summarised over
exactly those datasets so neither side sees data the other did not. Reusing the `6bba`
model on `6bba` data would have kept all 60 datasets and leaked; a smaller honest number
is worth more than a larger contaminated one.

*(That guard firing on a control is a design error carried over from `notes/20`, where it
existed to stop unusable weights — 0.2654 against DoG's 0.7696 — reaching a scorer. A
control is a measurement, not a shipment, and should be saved regardless. Fixed in the
training notebook for future runs; not worth re-running 1.5 h of GPU to regenerate here.)*

## What the normalisation bug was

Every training notebook stored `dog_response(load_frame(...))[0]` as its input tensor — a
**per-frame** percentile rescale. `predict_dataset` handed `prob_fn` the raw `load_frame`
output, normalised by **whole-movie** quantiles from the zarr attrs. Different
distributions: on the test volumes the two differ by 5.7× in range.

So every learned arm ever scored through this pipeline — `notes/21`, `notes/22`, and the
submission now sitting on the leaderboard — was served an input distribution it had never
trained on. Those runs are internally consistent with each other, because they all had it.
`r0_movie` reproduces that path so the comparison is exact rather than remembered.

## The gate

`adaptive_predicted`, CV **0.7070**, the configuration behind the 0.752 leaderboard score.
It is reproduced in-run so drift is charged to the champion and not to the new arm.
""")

md("## Pre-registered\n\n" + "\n".join(
    f"{i}. **{c}** {w}" for i, (c, w, _) in enumerate(PREDICTIONS, 1)))

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
import sys, os, gc, time, json, hashlib
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
for field in ("refine", "temporal_radius", "prob_input_norm"):
    if field not in pc.Config.__dataclass_fields__:
        raise SystemExit(f"The uploaded snapshot's Config has no `{field}` — re-upload. "
                         "Without it the arms below are not what they claim to be.")
if "prob_fn" not in pc.predict_dataset.__code__.co_varnames:
    raise SystemExit("The uploaded snapshot's predict_dataset has no prob_fn — re-upload.")
from pipeline.classical import (Config, budget_features, estimated_total_nodes,
                                make_predictor, predict_dataset)
from pipeline.detector import paired_recall
from pipeline.unet import UNet3D, predict_volume
from harness.purescore import match_nodes
from harness.tracks import read_geff

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
            if e.is_file() and e.name.startswith("claude_temporal_r") and e.name.endswith(".pt"):
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
    raise SystemExit("No claude_temporal_r*.pt found. Add the claude_temporal_train kernel "
                     "as a data source (Add Input -> Notebook Output).")

ckpts = {}              # (radius, train_emb) -> (path, checkpoint)
for wp in wpaths:
    ck = torch.load(wp, map_location="cpu")
    key = (int(ck["temporal_radius"]), ck["train_emb"])
    if key in ckpts:
        raise SystemExit(f"Two checkpoints for {key}: {ckpts[key][0].name} and {wp.name}. "
                         "Attach exactly one training kernel version.")
    ckpts[key] = (wp, ck)
    print(f"  {wp.name}: r={key[0]}, trained on {key[1]}, "
          f"paired={ck.get('best_paired'):.4f} node={ck.get('best_recall'):.4f} "
          f"@epoch {ck.get('best_epoch')} (DoG paired {ck.get('dog_paired'):.4f})")

by_radius = {}
for (rad, emb), v in ckpts.items():
    by_radius.setdefault(rad, {})[emb] = v
embryos_needed = {e for _, e in ckpts}
if 1 not in by_radius or set(by_radius[1]) != embryos_needed:
    raise SystemExit(f"r=1 must cover every fold; have "
                     f"{ {r: sorted(v) for r, v in by_radius.items()} }")

# A model may only score datasets from an embryo it did NOT train on, so a radius covers
# exactly the embryos it has an "other" model for. The training run's sub-DoG guard
# refused to save r0/44b6 (paired 0.8529 vs DoG 0.8550, a -0.0021 tie), which leaves r=0
# able to score only the 44b6 datasets.
#
# Rather than drop the temporal comparison or -- worse -- reuse the 6bba model on 6bba
# data and leak, each comparison runs on the datasets where BOTH of its arms are
# out-of-fold:
#
#   normalisation (r1_movie vs r1_perframe)   full subset, r=1 covers both folds
#   temporal      (r0_perframe vs r1_perframe) only the embryos r=0 can score
#
# The temporal delta is then computed by re-summarising the FULL r1_perframe run over
# that same restricted set, so both sides of it see identical datasets.
def coverage(rad):
    return {e for e in embryos_needed if any(o != e for o in by_radius.get(rad, {}))}

COVER = {rad: coverage(rad) for rad in by_radius}
TEMPORAL_EMBRYOS = COVER.get(0, set()) & COVER.get(1, set())
print(f"\nscoreable embryos by radius: { {r: sorted(v) for r, v in COVER.items()} }")
print(f"temporal comparison restricted to: {sorted(TEMPORAL_EMBRYOS) or 'NONE'}")
if not TEMPORAL_EMBRYOS:
    print("!! no embryo has an out-of-fold model at BOTH radii — the temporal delta "
          "cannot be measured here, only the normalisation one.")

MODELS = {}             # radius -> {embryo -> model}
for rad, per_emb in sorted(by_radius.items()):
    MODELS[rad] = {}
    for emb, (wp, ck) in sorted(per_emb.items()):
        m = UNet3D(base=ck.get("base", 16), depth=ck.get("depth", 3),
                   in_ch=ck.get("in_ch", 1 if rad == 0 else 3))
        m.load_state_dict(ck["state_dict"]); m.eval().to(DEV)
        MODELS[rad][emb] = m
        print(f"  r={rad}: {wp.name} scores datasets NOT from {emb}")

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
BASELINE_CV = 0.6490          # notes/22: unet_cap1.2_norefine, the arm to beat
results = {}
""")

md("""## 1. The budget regression, refit here

`notes/14` §2: a constant budget scores **0.0882 below doing nothing**, so this is refit at
runtime rather than carried as coefficients. Same features and same leave-one-embryo-out
fit that measured 10.7 % median error in `08`, and reproduced exactly in `notes/21` and
`notes/22`.
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
    if (i + 1) % 20 == 0:
        print(f"  features {i+1}/{len(SUBSET)}  ({time.time()-t0:.0f}s)", flush=True)

def design(rows):
    return np.array([[1.0] + [np.log1p(f[k]) if k.startswith("n") else
                              np.log(max(f[k], 1e-6)) for k in FEAT_NAMES]
                     for f in rows], float)

usable = [n for n in SUBSET if BUDGETS.get(n)]
X = design([FEATS[n] for n in usable])
y = np.array([np.log(BUDGETS[n] / max(1.0, FEATS[n]["T"])) for n in usable])
pref = np.array([n.split("_")[0] for n in usable])

PRED_BUDGETS, errs = {}, []
for g in sorted(set(pref)):
    m = pref == g
    b_g = np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]
    e = np.exp(X[m] @ b_g) * np.array([FEATS[n]["T"] for n, k in zip(usable, m) if k])
    for n, v in zip([n for n, k in zip(usable, m) if k], e):
        PRED_BUDGETS[n] = float(v)
    errs.append(np.abs(e / np.array([BUDGETS[n] for n, k in zip(usable, m) if k]) - 1))
pooled = np.concatenate(errs)
print(f"\nleave-one-embryo-out budget error: median {np.median(pooled):.1%} "
      f"(08 measured 10.7%, notes/21-22 reproduced it exactly)")
""")

md("""## 2. Score the arms

`diagnose` collects the near/far miss split and the paired-recall accounting per dataset,
so a result arrives with a mechanism rather than as a bare number. That is what turned
`notes/21`'s gate failure into the most informative run in the project.
""")

code(r"""
SCALE_UM = (1.625, 0.40625, 0.40625)     # GT lives in FULL-resolution voxels
NEAR_UM = 14.0                           # 2x the 7um match radius
DIAG, CURRENT_ARM = {}, ""

def diagnose(name, data_dir, graph, s):
    g = read_geff(Path(data_dir) / f"{name}.geff")
    if not len(g.t) or not len(graph.t):
        return
    matched = match_nodes(graph.t, graph.zyx, g.t, g.zyx, scale=s, max_distance=7.0)
    hit = set(matched[matched >= 0].tolist())
    near = far = 0
    for t in np.unique(g.t):
        gi = np.flatnonzero(g.t == t)
        miss = [i for i in gi if i not in hit]
        if not miss:
            continue
        pj = np.flatnonzero(graph.t == t)
        if not len(pj):
            far += len(miss); continue
        d = np.linalg.norm((g.zyx[miss][:, None] - graph.zyx[pj][None]) * s, axis=2).min(1)
        near += int((d <= NEAR_UM).sum()); far += int((d > NEAR_UM).sum())
    # The same coherence measure the training run selected on, now on FULL movies rather
    # than short runs -- so the number that chose the checkpoint and the number that
    # explains the score are the same quantity.
    pr = paired_recall(graph.t, graph.zyx, g.t, g.zyx, g.edges, s)
    DIAG.setdefault(CURRENT_ARM, []).append(
        {"name": name, "n_gt": int(len(g.t)), "matched": len(hit),
         "near_miss": near, "far_miss": far, "n_pred": int(len(graph.t)),
         "paired": pr["paired"], "position": pr["position"],
         "pair_edges": pr["n_edges"]})

def make_unet_predictor(cfg, budgets, radius):
    def _fn(name, data_dir):
        emb = name.split("_")[0]
        other = [e for e in MODELS[radius] if e != emb]
        model = MODELS[radius][other[0] if other else emb]
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

def run(name, cfg, predictor=None, names=None):
    global CURRENT_ARM
    CURRENT_ARM = name
    t0 = time.time()
    fn = predictor if predictor is not None else make_dog_predictor(cfg, PRED_BUDGETS)
    res = h.evaluate(fn, arm=name, names=names or SUBSET, verbose=False)
    s = res.summary
    n = sum(r["num_pred_nodes"] for r in res.rows.values())
    mult = s["adj_edge_jaccard"] / s["edge_jaccard"] if s["edge_jaccard"] else float("nan")
    rows = DIAG.get(name, [])
    tot_e = sum(r["pair_edges"] for r in rows)
    pos = (sum(r["position"] * r["pair_edges"] for r in rows
               if r["position"] == r["position"]) / tot_e) if tot_e else float("nan")
    print(f"{name:<16} SCORE={s['score']:.4f}  edge_J={s['edge_jaccard']:.4f}  "
          f"mult={mult:.4f}  recall={s['node_recall']:.3f}  nodes={n:>9,}  "
          f"position={pos:+.1%}  ({time.time()-t0:.0f}s)", flush=True)
    results[name] = res
    results[name].position = pos
    return res

# The champion, reproduced here so drift cannot be charged to the new arms.
run("champion", Config(min_separation_um=6.0, adaptive_separation=True,
                       adaptive_target=1.2, prune_isolated_nodes=True, **BASE))
drift = results["champion"].score - CHAMPION_CV
print(f"\nreproduction: {results['champion'].score:.4f} vs {CHAMPION_CV:.4f} "
      f"(drift {drift:+.4f})")
if abs(drift) > 0.005:
    print("!! the champion moved — read everything below against THIS number")
""")

code(r"""
# budget_fill=1.2 and refine=False are notes/22's best learned configuration, held fixed
# across every learned arm so the only things that move are the two under test.
LEARNED = dict(min_separation_um=6.0, budget_fill=1.2, refine=False,
               prune_isolated_nodes=True, **BASE)

# Normalisation, on the FULL subset. r=1 rather than r=0 because only r=1 covers both
# folds -- the two arms differ in prob_input_norm and in nothing else, which is what makes
# this an isolated measurement of the train/serve skew.
for tag, norm in (("r1_movie", "movie"), ("r1_perframe", "per_frame")):
    cfg = Config(temporal_radius=1, prob_input_norm=norm, **LEARNED)
    run(tag, cfg, predictor=make_unet_predictor(cfg, PRED_BUDGETS, 1))

# Temporal, on the embryos r=0 can score out-of-fold.
TEMPORAL_NAMES = sorted(n for n in SUBSET if n.split("_")[0] in TEMPORAL_EMBRYOS)
print(f"\ntemporal comparison on {len(TEMPORAL_NAMES)} of {len(SUBSET)} datasets "
      f"({sorted(TEMPORAL_EMBRYOS)})")
if TEMPORAL_NAMES:
    cfg0 = Config(temporal_radius=0, prob_input_norm="per_frame", **LEARNED)
    run("r0_perframe", cfg0, predictor=make_unet_predictor(cfg0, PRED_BUDGETS, 0),
        names=TEMPORAL_NAMES)
""")

code(r"""
from harness.harness import summarise

# Re-summarise the FULL r1_perframe run over the temporal subset, so both sides of the
# temporal delta are scored on identical datasets. Re-running r=1 on the subset would
# produce the same graphs at extra cost; restricting the rows is the same measurement.
def sub_score(tag, names):
    rows = [r for n, r in results[tag].rows.items() if n in set(names)]
    return summarise(rows)["score"] if rows else float("nan")

TEMPORAL_NAMES = sorted(n for n in SUBSET if n.split("_")[0] in TEMPORAL_EMBRYOS)
if TEMPORAL_NAMES and "r0_perframe" in results:
    r1_sub = sub_score("r1_perframe", TEMPORAL_NAMES)
    r0_sub = results["r0_perframe"].score
    print(f"on the {len(TEMPORAL_NAMES)} temporal-comparable datasets:")
    print(f"  r0_perframe {r0_sub:.4f}   r1_perframe {r1_sub:.4f}   "
          f"temporal delta {r1_sub - r0_sub:+.4f}")
else:
    r1_sub = r0_sub = float("nan")
    print("temporal delta not measurable from the attached checkpoints")
""")

md("""## 3. Grade the pre-registered predictions""")

code(r"""
PREDICTIONS = """ + json.dumps([[c, w, k] for c, w, k in PREDICTIONS]) + r"""
ref = results["champion"]

def S(tag): return results[tag].score if tag in results else float("nan")
def P(tag): return getattr(results[tag], "position", float("nan")) if tag in results else float("nan")

print(f"{'arm':<16} {'SCORE':>8} {'edge_J':>8} {'recall':>7} {'position':>10} {'vs champ':>9}")
print("-" * 62)
for tag in ("champion", "r1_movie", "r1_perframe", "r0_perframe"):
    if tag not in results:
        continue
    s = results[tag].summary
    note = "  (subset)" if tag == "r0_perframe" else ""
    print(f"{tag:<16} {s['score']:>8.4f} {s['edge_jaccard']:>8.4f} "
          f"{s['node_recall']:>7.3f} {P(tag):>9.1%} "
          f"{s['score'] - ref.score:>+9.4f}{note}")

# Normalisation: full subset, both arms r=1, differing ONLY in prob_input_norm.
d_norm = S("r1_perframe") - S("r1_movie")
# Temporal: restricted to the datasets both radii can score out-of-fold, with r=1
# re-summarised over exactly those datasets so neither side sees data the other did not.
d_temp = r1_sub - r0_sub
d_pos = P("r1_perframe") - P("r0_perframe")
print(f"\nDECOMPOSED — the two changes measured separately, on matched datasets:")
print(f"  normalisation fix   r1_movie -> r1_perframe    {d_norm:+.4f}   "
      f"({len(SUBSET)} datasets)")
print(f"  temporal input      r0_perframe -> r1_perframe {d_temp:+.4f}   "
      f"({len(TEMPORAL_NAMES)} datasets, position {d_pos:+.1%})")
print(f"\n  notes/22 baseline (unet_cap1.2_norefine): {BASELINE_CV:.4f}")
print(f"  best full-subset learned arm here:        "
      f"{max(S('r1_movie'), S('r1_perframe')):.4f}")

# Only full-subset arms can be compared to the champion; r0_perframe ran on a subset and
# its number is not commensurable with a 60-dataset score.
best_learned = max(S("r1_movie"), S("r1_perframe"))
VERDICTS = {
    "norm_fix_helps": bool(d_norm > 0),
    # BOTH clauses. A score gain without a position gain means something other than
    # coherence produced it, and notes/21's mechanism would still be unconfirmed -- so
    # this must not read as a confirmation.
    "temporal_beats_single": bool(d_temp > 0 and d_pos > 0),
    "still_short": bool(best_learned <= ref.score),
}

print()
for i, (claim, why, key) in enumerate(PREDICTIONS, 1):
    print(f"{i}. {'CONFIRMED' if VERDICTS[key] else 'FALSIFIED':<10} {claim}")

if not VERDICTS["still_short"]:
    print(f"\n*** A LEARNED ARM PASSED THE GATE: {best_learned:.4f} > {ref.score:.4f}. "
          "Prediction 3 was written to make this draw scrutiny rather than celebration — "
          "check the champion's drift above before believing it.")
if VERDICTS["temporal_beats_single"] is False and d_temp > 0:
    print(f"\n!! The score moved {d_temp:+.4f} but temporal position did NOT "
          f"({d_pos:+.1%}). Whatever produced the gain, it was not the mechanism "
          "notes/21 identified — do not report it as confirmation of that mechanism.")

blob = json.dumps({
    "scores": {t: results[t].score for t in results},
    "summaries": {t: dict(results[t].summary) for t in results},
    "position": {t: P(t) for t in results},
    "deltas": {"normalisation": d_norm, "temporal": d_temp, "position": d_pos},
    "temporal_subset": {"names": TEMPORAL_NAMES, "r0": r0_sub, "r1": r1_sub},
    "champion_cv": CHAMPION_CV, "champion_drift": drift,
    "baseline_cv": BASELINE_CV, "verdicts": VERDICTS,
    "diag": {k: v for k, v in DIAG.items()},
}, indent=2, default=float)
(WORK / "claude_temporal_score_results.json").write_text(blob)
print("\nwrote claude_temporal_score_results.json")
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
