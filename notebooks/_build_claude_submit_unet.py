"""Build notebooks/claude_submit_unet.ipynb — the learned detector's LB calibration point.

Derived from `_build_09_submission.py`, which is the artefact that actually scored 0.752.
Everything that notebook solves about a scored rerun (no network, wheelhouse fallback,
globbed test names, runtime budget regression, streamed CSV, time guard) is kept verbatim.
Three things change: the detector, the density knob, and the checkpoint plumbing.
"""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/claude_submit_unet.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Submission — the learned detector, `pu` ensemble at cap 1.2x

**This submission is expected to score below the banked 0.752.** It is being made anyway,
deliberately, and the reason is worth stating before any of the code.

## Why a submission that is predicted to lose

The standing policy is *gated configurations only*: nothing reaches the leaderboard unless
it beat the champion in cross-validation. The best learned arm did not. `notes/22`:

| arm | CV score | vs champion |
|---|---|---|
| champion — `adaptive_predicted` (**LB 0.752**) | **0.7070** | — |
| `unet_cap1.2_norefine` — what this notebook ships | 0.6490 | **−0.0580** |

So this run is a **deliberate, owner-authorised exception**, and it buys one specific
thing: **the CV→LB transfer function for the learned path.**

We know one point of that function — champion CV 0.7070 → LB 0.752, an offset of
**+0.045**. We do not know whether that offset holds for a learned detector, and we need
it: the next run (temporal input) will produce a CV number, and the decision to submit it
or not rests entirely on projecting that number to the leaderboard. Guessing the offset
would make that decision unfounded. One measurement removes the guess.

If the offset transfers, this scores about **0.694**. That is the null hypothesis.

## Which weights, and why an ensemble

`claude_detector_earlystop` trained one model per embryo under leave-one-embryo-out, and
each was scored only by the fold it never saw. The hidden test set is **two unseen
embryos**, so neither model has a familiarity advantage over the other — both are "the
other embryo's model" with respect to any test crop.

`notes/20` §1 argued the CV is *pessimistic* about a real submission for exactly this
reason: it trains on one embryo where the submission can use both. §3 recorded the
consequence — "training on both embryos is the single largest untested lever ... it cannot
be validated with two embryos, but it can be *submitted* and measured on the leaderboard."

**This notebook averages the two `pu` probability maps.** That is not the same as one model
trained on both embryos, but it is the available proxy, and it is what that note points at.

It does mean the measured offset bundles two effects — CV→LB transfer and any ensemble
bonus. That is acceptable *because they will always be applied together*: the temporal run
will ensemble the same way, so the combined offset is precisely the transfer function that
run needs. Two unknowns that never need separating do not need separating.

Cost: inference runs twice. Measured 215 ms per 64³ volume on Kaggle CPU, so ~2.4 h for a
200-dataset test set instead of ~1.2 h. Both fit inside twelve hours.

## The configuration, and one thing that is NOT in it

`Config(min_separation_um=6.0, budget_fill=1.2, refine=False, prune_isolated_nodes=True)`

- **`budget_fill=1.2`, not `adaptive_separation`.** For a learned detector the density knob
  is the **cap**: the probability map is thresholded at a floor (`unet_threshold=1e-6`) and
  the per-frame budget selects the strongest peaks. Adaptive separation is a DoG-side
  control and does nothing useful here. 1.2× was the best of 0.8/1.0/1.2 in `notes/21`.
- **`refine=False`.** `notes/22` measured the intensity refinement costing the learned path
  **+0.0038 of score and 7.9 points of temporal coherence**, because it jitters peak
  positions frame to frame with intensity noise. Leaving it on would ship a known defect.
- **`detector="dog"` is still set** and is deliberately inert: `prob_fn` overrides the
  detection function, and DoG is left in the config so this run's `Config` is
  byte-identical to the one that measured 0.6490. Drift between the measured arm and the
  submitted arm would destroy the calibration this notebook exists to buy.

## What a scored rerun forbids

Same three constraints `09` was built around — no `geff`, no `polars` on the writing path,
`test/` globbed rather than hardcoded — plus one more:

**`torch` must come from the image.** There is no torch wheel in the wheelhouse and no
network to fetch one. The image ships torch 2.10+cu128, whose CUDA kernels are sm_70+ and
therefore useless on Kaggle's P100 — but that is a **GPU-only** problem. This notebook runs
on **CPU**, where the same build works fine. It never calls `.cuda()` and never asks for a
GPU.
""")

code(r"""
# No network install anywhere in this notebook. This cell reports what the image has.
import sys, os, time, platform, subprocess
from pathlib import Path
print(f"python {platform.python_version()}")

def probe():
    have, gone = {}, []
    for mod in ("numpy", "scipy", "zarr", "torch", "polars", "geff", "pandas"):
        try:
            m = __import__(mod)
            have[mod] = getattr(m, "__version__", "?")
        except ImportError:
            gone.append(mod)
    return have, gone

have, missing = probe()
for mod, ver in have.items():
    print(f"  {mod:<8} {ver}")
for mod in missing:
    print(f"  {mod:<8} MISSING")

# torch is NOT in this list: the wheelhouse has no torch wheel and could not sanely carry
# one. It has to come from the image, and it does. It is checked separately below.
WHEELABLE = ("numpy", "scipy", "zarr")
if any(m in missing for m in WHEELABLE):
    # Fallback: install from an attached wheelhouse dataset. --no-index means pip never
    # touches the network, so this works with internet off. See 10_wheelhouse.ipynb.
    # Bounded scan, NOT Path.glob("**/*.whl") -- that walks into every .zarr and takes
    # ~3 minutes on this mount, because each dataset is thousands of chunk files.
    def find_wheelhouses(root="/kaggle/input", max_depth=4):
        found, stack = set(), [(Path(root), 0)]
        while stack:
            d, depth = stack.pop()
            try:
                kids = list(os.scandir(d))
            except (PermissionError, OSError, FileNotFoundError):
                continue
            if any(e.is_file() and e.name.endswith(".whl") for e in kids):
                found.add(d)
            if depth < max_depth:
                stack += [(Path(e.path), depth + 1) for e in kids
                          if e.is_dir() and not e.name.endswith((".zarr", ".geff"))]
        return sorted(found)

    t_scan = time.time()
    wheelhouses = find_wheelhouses()
    print(f"\nwheelhouse directories found ({time.time()-t_scan:.1f}s): "
          f"{[str(w) for w in wheelhouses] or 'NONE'}")
    for w in wheelhouses:
        r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                            f"--find-links={w}", *[m for m in missing if m in WHEELABLE]],
                           capture_output=True, text=True)
        print(f"  install from {w}: {'ok' if r.returncode == 0 else 'failed'}")
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
    have, missing = probe()

still = [m for m in WHEELABLE if m in missing]
if still:
    raise SystemExit(
        f"{', '.join(still)} not importable, and internet is off in a scored rerun so "
        "PyPI is unreachable. Build a wheelhouse dataset with 10_wheelhouse.ipynb "
        "(internet ON) and attach it as an input."
    )
if "torch" in missing:
    raise SystemExit(
        "torch is not importable. It ships in the Kaggle image and cannot be installed "
        "here -- the wheelhouse carries no torch wheel and there is no network in a "
        "scored rerun. If the image has genuinely dropped torch, this whole approach "
        "needs a different delivery route; DO NOT paper over it with a DoG fallback, "
        "which would silently submit a different model than the one this run measures."
    )
print("\nrequired packages present:",
      ", ".join(f"{m} {have[m]}" for m in (*WHEELABLE, "torch")))
""")

code(r"""
import os, time, json, math
from pathlib import Path
import numpy as np
import torch

T_START = time.time()
TIME_BUDGET_S = 10.5 * 3600     # 12 h hard cap; leave 1.5 h of headroom
WORK = Path("/kaggle/working")

# CPU, explicitly and unconditionally. Kaggle's free GPU is a P100 (sm_60) and the image's
# torch ships kernels for sm_70+, so CUDA here reports "available" and then every launch
# dies -- notes/19 §5, which cost a run. Inference is 215 ms per 64^3 volume on CPU, which
# is affordable, so there is nothing to gain by touching the GPU even if one appeared.
DEV = torch.device("cpu")
torch.set_grad_enabled(False)
try:
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except Exception as e:
    print(f"  (could not set thread count: {e})")
print(f"torch {torch.__version__}  device {DEV}  threads {torch.get_num_threads()}")

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
    raise SystemExit("Could not find harness/ and pipeline/. Add the project dataset as an input.")
sys.path.insert(0, str(REPO))

import pipeline.classical as pc
from pipeline.classical import (Config, budget_features, build_graph,
                                estimated_total_nodes, predict_dataset)
from pipeline.unet import UNet3D, predict_volume
from harness.csvout import check_graph, write_submission

# Guards against an out-of-date snapshot. Each one is a real failure that has happened:
# a missing prob_fn silently runs DoG, and a missing `refine` field silently ships the
# jitter notes/22 measured. Both would produce a plausible-looking wrong number.
for need in ("budget_features", "open_movie"):
    if not hasattr(pc, need):
        raise SystemExit(f"Snapshot lacks pipeline.classical.{need} — re-upload the repo.")
if "prob_fn" not in pc.predict_dataset.__code__.co_varnames:
    raise SystemExit("The uploaded snapshot's predict_dataset has no prob_fn — re-upload.")
if "refine" not in pc.Config.__dataclass_fields__:
    raise SystemExit("The uploaded snapshot's Config has no `refine` field — re-upload.")

COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "test").glob("*.zarr")), ["/kaggle/input"])
if COMP is None:
    raise SystemExit("Could not find the competition data (a dir with train/ and test/).")
TRAIN, TEST = COMP / "train", COMP / "test"

# GLOBBED, never hardcoded: the rerun swaps test/ for the real hidden set.
TEST_NAMES = sorted(p.stem for p in TEST.glob("*.zarr"))
TRAIN_NAMES = sorted(p.stem for p in TRAIN.glob("*.zarr"))
if not TEST_NAMES:
    raise SystemExit(f"No .zarr datasets under {TEST}")
print(f"competition: {COMP}")
print(f"  train: {len(TRAIN_NAMES)} datasets   test: {len(TEST_NAMES)} datasets")
print(f"  first three test names: {TEST_NAMES[:3]}")
""")

md("""## 1. Load the checkpoints

Weights arrive as a **kernel data source** (`claude_detector_earlystop`). Attaching a kernel
attaches *all* of its outputs, so the selection happens here at load time rather than by
choosing what to attach — an earlier notebook refused on an instruction ("attach only the
winning loss") that cannot actually be carried out.

The loss is chosen by **mean held-out recall above DoG**, not raw recall: DoG's own recall
differs per embryo (0.7696 on `6bba`, 0.8776 on `44b6`), so raw numbers are not comparable
across folds and the margin is. `notes/20` measured `pu` at **+0.0751** against `masked`'s
+0.0123, and `pu` is the only loss whose checkpoints cover both embryos at all — the
sub-DoG guard correctly refused to save `masked`/`44b6`.
""")

code(r"""
# Bounded scan -- a recursive glob over /kaggle/input walks every .zarr chunk (notes/17 §4).
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
    raise SystemExit("No claude_unet_*.pt found. Add the claude_detector_earlystop kernel "
                     "as a data source (Add Input -> Notebook Output).")

FORCE_LOSS = None       # set to "masked" / "pu" to override the automatic choice

ckpts = {}              # (loss, train_emb) -> (path, checkpoint)
for wp in wpaths:
    ck = torch.load(wp, map_location="cpu")
    key = (ck.get("loss", "?"), ck["train_emb"])
    if key in ckpts:
        raise SystemExit(f"Two checkpoints for {key}: {ckpts[key][0].name} and {wp.name}. "
                         "Attach exactly one training kernel version.")
    ckpts[key] = (wp, ck)
    print(f"  {wp.name}: loss={key[0]}, trained on {key[1]}, "
          f"best_recall={ck.get('best_recall')} @epoch {ck.get('best_epoch')}, "
          f"DoG {ck.get('dog_recall')}")

by_loss = {}
for (ln, emb), (wp, ck) in ckpts.items():
    by_loss.setdefault(ln, {})[emb] = (wp, ck)
embryos_needed = {e for _, e in ckpts}
complete = {ln: v for ln, v in by_loss.items() if set(v) == embryos_needed}
if not complete:
    raise SystemExit(f"No loss covers every embryo. Have: "
                     f"{ {ln: sorted(v) for ln, v in by_loss.items()} }")

def margin(v):
    return float(np.mean([ck.get("best_recall", 0.0) - ck.get("dog_recall", 0.0)
                          for _, ck in v.values()]))

ranked = sorted(complete, key=lambda ln: -margin(complete[ln]))
CHOSEN = FORCE_LOSS or ranked[0]
if CHOSEN not in complete:
    raise SystemExit(f"FORCE_LOSS={FORCE_LOSS!r} does not cover every embryo.")
print(f"\nloss ranking by mean margin over DoG: "
      + ", ".join(f"{ln} {margin(complete[ln]):+.4f}" for ln in ranked))
print(f"CHOSEN LOSS: {CHOSEN}" + (" (forced)" if FORCE_LOSS else " (best mean margin)"))

# ENSEMBLE, not selection: the test embryos are unseen by BOTH models, so neither is the
# designated "other embryo" model the way CV had one. See the header.
ENSEMBLE = []
for emb, (wp, ck) in sorted(complete[CHOSEN].items()):
    m = UNet3D(base=ck.get("base", 16), depth=ck.get("depth", 3))
    m.load_state_dict(ck["state_dict"])
    m.eval().to(DEV)
    ENSEMBLE.append((emb, m))
    print(f"  ensembling {wp.name} (trained on {emb})")
if len(ENSEMBLE) < 2:
    raise SystemExit(f"Expected two folds to ensemble; got {[e for e, _ in ENSEMBLE]}")

# Averaging PROBABILITIES, not logits: peaks_from_prob thresholds and collapses on the
# probability scale, and a logit mean would let one saturated model dominate a region the
# other calls empty.
def prob_fn(vol):
    acc = None
    for _, m in ENSEMBLE:
        p = predict_volume(m, vol, DEV)
        acc = p if acc is None else acc + p
    return acc / len(ENSEMBLE)

# Time one call so the projection below uses this machine's speed, not a remembered one.
try:
    _probe = np.random.default_rng(0).random((64, 64, 64)).astype(np.float32)
    _t = time.time(); _ = prob_fn(_probe); SEC_PER_VOLUME = time.time() - _t
    print(f"\nensemble inference: {SEC_PER_VOLUME*1000:.0f} ms per 64^3 volume "
          f"({len(ENSEMBLE)} models, CPU)")
except Exception as e:
    SEC_PER_VOLUME = 0.43
    print(f"\n!! inference probe failed ({e}); assuming {SEC_PER_VOLUME*1000:.0f} ms/volume")
""")

md("""## 2. Size the run before starting it

`09` projected from DoG timings. This notebook replaces DoG detection with two forward
passes per frame, so the per-dataset cost is measured above and projected here rather than
carried over. The fit size is still the only discretionary term — every test dataset has
to be predicted.
""")

code(r"""
SEC_PER_DETECTION = 1.3      # 08: 540 detections in 700 s (budget features, DoG-side)
SEC_PER_LINK = 20.0          # 08: pipeline minus detection

FRAC_FRAMES = (0.25, 0.5, 0.75)
REF_SEPS = (4.0, 8.0, 16.0)
FEAT_NAMES = ["n_sep4", "n_sep8", "n_sep16", "nstrong_sep4", "nstrong_sep8",
              "nstrong_sep16", "mean_int", "frac_fg"]

# Frames per dataset, read rather than assumed: it sets the inference bill directly.
try:
    _arr, _attrs, _scale, _vox, _lo, _hi = pc.open_movie(TEST / f"{TEST_NAMES[0]}.zarr",
                                                         Config())
    T_PER_DATASET = int(_arr.shape[0])
    del _arr
except Exception as e:
    T_PER_DATASET = 25
    print(f"!! could not read frame count ({e}); assuming {T_PER_DATASET}")
print(f"frames per test dataset: {T_PER_DATASET}")

def project(n_fit, n_det):
    feat = (n_fit + len(TEST_NAMES)) * n_det * SEC_PER_DETECTION
    infer = len(TEST_NAMES) * T_PER_DATASET * SEC_PER_VOLUME
    link = len(TEST_NAMES) * SEC_PER_LINK
    return feat + infer + link

N_FIT = len(TRAIN_NAMES)
n_det = len(FRAC_FRAMES) * len(REF_SEPS)
print(f"projection: budget features {(N_FIT+len(TEST_NAMES))*n_det*SEC_PER_DETECTION/3600:.1f} h"
      f" + inference {len(TEST_NAMES)*T_PER_DATASET*SEC_PER_VOLUME/3600:.1f} h"
      f" + linking {len(TEST_NAMES)*SEC_PER_LINK/3600:.1f} h"
      f" = {project(N_FIT, n_det)/3600:.1f} h")

# 08 reached 10.7% median budget error fitting on ~30 datasets per embryo, so there is room
# to cut the fit before the model degrades. Inference and linking are not discretionary.
while project(N_FIT, n_det) > TIME_BUDGET_S and N_FIT > 40:
    N_FIT = max(40, int(N_FIT * 0.75))
    print(f"  trimming the fit to {N_FIT} train datasets -> {project(N_FIT, n_det)/3600:.1f} h")
if project(N_FIT, n_det) > TIME_BUDGET_S:
    print(f"  !! still projects {project(N_FIT, n_det)/3600:.1f} h against a "
          f"{TIME_BUDGET_S/3600:.1f} h budget. Inference dominates and cannot be trimmed "
          "without changing the model. Proceeding; the guard in cell 3 keeps the "
          "submission valid if it runs out.")

# Spread the fit across the corpus rather than taking a prefix, so both embryos are in it
# whatever the naming turns out to be.
FIT_NAMES = ([TRAIN_NAMES[i] for i in np.linspace(0, len(TRAIN_NAMES) - 1, N_FIT).astype(int)]
             if N_FIT < len(TRAIN_NAMES) else list(TRAIN_NAMES))
FIT_NAMES = sorted(set(FIT_NAMES))
print(f"\nfitting the budget regression on {len(FIT_NAMES)} train datasets")
""")

md("""## 3. Fit the budget regression on `train/`

Unchanged from `09`, and load-bearing for a different reason here. With `budget_fill=1.2`
the predicted budget *is* the density control — it sets the per-frame cap directly, where
in `09` it only steered the adaptive separation search. A bad budget is therefore worse
here, not better.

`notes/14` §2: a single constant budget applied to every dataset scores **0.0882 below
doing nothing at all**. The regression is fit at runtime on whatever `train/` the rerun
mounts; a hardcoded coefficient vector is the failure mode this cell exists to avoid.
""")

code(r"""
# detector="dog" is INERT -- prob_fn overrides detection. It is kept so this Config is
# identical to the one that measured 0.6490 in notes/22, because any drift between the
# measured arm and the submitted arm destroys the calibration this notebook is buying.
CFG = Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
             dog_scales=[(1.5, 4.0), (2.5, 6.0)], footprint="ball",
             budget_fill=1.2, refine=False, prune_isolated_nodes=True)
print(f"config: budget_fill={CFG.budget_fill}  refine={CFG.refine}  "
      f"unet_threshold={CFG.unet_threshold}  prune={CFG.prune_isolated_nodes}")
assert CFG.refine is False, "refine must be off -- notes/22 measured it costing 0.0038"
assert not CFG.adaptive_separation, "adaptive separation is a DoG-side control"

t0 = time.time()
fit_rows, fit_y, fit_used = [], [], []
for i, n in enumerate(FIT_NAMES):
    b = estimated_total_nodes(TRAIN / f"{n}.zarr")
    if not b:
        continue
    f = budget_features(TRAIN / f"{n}.zarr", CFG, frac_frames=FRAC_FRAMES, ref_seps=REF_SEPS)
    fit_rows.append(f)
    fit_y.append(math.log(b / max(1.0, f["T"])))
    fit_used.append(n)
    if len(fit_used) % 20 == 0:
        print(f"  fit features {len(fit_used):>3}/{len(FIT_NAMES)}  "
              f"({time.time()-t0:.0f}s)", flush=True)
print(f"  fit features {len(fit_used)}/{len(FIT_NAMES)} usable  ({time.time()-t0:.0f}s)")

if len(fit_used) < 20:
    raise SystemExit(
        f"Only {len(fit_used)} train datasets exposed estimated_number_of_nodes. The "
        "budget regression cannot be fit, and notes/14 §2 measured a constant budget at "
        "-0.0882 against doing nothing — so falling back to one would make this "
        "submission worse than the previous champion. Stopping instead."
    )

def design(rows):
    return np.array([[1.0] + [np.log1p(f[k]) if k.startswith("n") else
                              np.log(max(f[k], 1e-6)) for k in FEAT_NAMES]
                     for f in rows], float)

X, y = design(fit_rows), np.array(fit_y, float)
BETA = np.linalg.lstsq(X, y, rcond=None)[0]
resid = np.exp(X @ BETA) / np.exp(y)
print(f"\nin-sample budget error: median {np.median(np.abs(resid-1)):.1%}  "
      f"mean {np.mean(np.abs(resid-1)):.1%}")

# Held-out estimate: split by name prefix, which is the embryo. This mirrors 08's
# leave-one-embryo-out (10.7% median) and is the number that predicts the hidden set.
pref = np.array([n.split("_")[0] for n in fit_used])
groups = sorted(set(pref))
if len(groups) > 1:
    errs = []
    for g in groups:
        m = pref == g
        if m.sum() < 5 or (~m).sum() < 5:
            continue
        b_g = np.linalg.lstsq(X[~m], y[~m], rcond=None)[0]
        e = np.abs(np.exp(X[m] @ b_g) / np.exp(y[m]) - 1)
        errs.append(e)
        print(f"  held out {g}: median {np.median(e):.1%}  (n={int(m.sum())})")
    if errs:
        pooled = np.concatenate(errs)
        print(f"  pooled leave-one-embryo-out median {np.median(pooled):.1%} "
              f"(08 measured 10.7%, and notes/21-22 reproduced it exactly)")
else:
    print("  only one name prefix in train — no held-out estimate available")
""")

md("""## 4. Predict, check, and stream to CSV

One dataset at a time: features → predicted budget → **ensemble forward pass** → peaks →
link → prune → rows. Never more than one graph in memory, and `check_graph` runs on each
before it is written, so a malformed graph is reported here rather than silently repaired
by the scorer.
""")

code(r"""
def predict_test():
    t_run, starved = time.time(), False
    for i, n in enumerate(TEST_NAMES, 1):
        # A partial submission scores 0 on the datasets it skips. A submission that never
        # finishes scores nothing at all, so past the budget we keep emitting rows and
        # stop paying for predictions.
        if starved or time.time() - T_START > TIME_BUDGET_S:
            if not starved:
                print(f"!! time budget exhausted at dataset {i} of {len(TEST_NAMES)}; "
                      "the rest are written empty so the run still produces a valid "
                      "submission", flush=True)
                starved = True
            yield n, build_graph(np.zeros((0, 4)), [])
            continue
        f = budget_features(TEST / f"{n}.zarr", CFG, frac_frames=FRAC_FRAMES,
                            ref_seps=REF_SEPS)
        budget = float(np.exp(design([f])[0] @ BETA) * f["T"])
        tr = predict_dataset(TEST / f"{n}.zarr", CFG, verbose=False,
                             est_total_nodes=budget, prob_fn=prob_fn)
        elapsed = time.time() - t_run
        print(f"[{i:>3}/{len(TEST_NAMES)}] {n:<24} budget~{budget:>9,.0f}  "
              f"{tr.n_nodes:>7,} nodes  {tr.n_edges:>7,} edges  "
              f"({elapsed:.0f}s, projected total {elapsed/i*len(TEST_NAMES)/3600:.1f}h)",
              flush=True)
        yield n, tr

SUB = WORK / "submission.csv"
summary = write_submission(predict_test(), SUB, verbose=False)
print(f"\n{summary['rows']:,} rows | {summary['datasets']} datasets | "
      f"{summary['nodes']:,} nodes | {summary['edges']:,} edges")
print(f"total elapsed {(time.time()-T_START)/3600:.2f} h")
if summary["problems"]:
    print(f"\n!! {len(summary['problems'])} problems the scorer would silently repair:")
    for p in summary["problems"][:20]:
        print(f"   {p}")
else:
    print("no malformed-graph problems found")
""")

code(r"""
# Read the file back. Everything above worked on objects; this checks the artefact.
with SUB.open() as fh:
    header = fh.readline().strip().split(",")
    n_rows = sum(1 for _ in fh)
print(f"header: {header}")
print(f"rows (excluding header): {n_rows:,}")

expected = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x",
            "source_id", "target_id"]
assert header == expected, f"column mismatch\n  got      {header}\n  expected {expected}"
assert n_rows == summary["rows"], f"row count drifted: {n_rows} vs {summary['rows']}"
assert set(summary["names"]) == set(TEST_NAMES), "not every test dataset was written"
print(f"\nOK: {SUB} covers all {len(TEST_NAMES)} test datasets with the expected schema.")
print(f"size: {SUB.stat().st_size/1e6:.1f} MB")
""")

md("""## 5. What this run is worth, stated before the score arrives

The projection below is written into the output so the leaderboard number is read against
a prediction rather than rationalised after the fact.
""")

code(r"""
CHAMPION_CV, CHAMPION_LB = 0.7070, 0.752
THIS_CV = 0.6490                      # notes/22, unet_cap1.2_norefine
OFFSET = CHAMPION_LB - CHAMPION_CV    # +0.0450, the one point of the transfer function

print(f"champion:  CV {CHAMPION_CV:.4f} -> LB {CHAMPION_LB:.4f}   (offset {OFFSET:+.4f})")
print(f"this arm:  CV {THIS_CV:.4f} -> LB {THIS_CV + OFFSET:.4f} IF the offset transfers")
print(f"           and the ensemble adds nothing beyond it.\n")
print("How to read the result:")
print(f"  ~{THIS_CV + OFFSET:.3f}  the offset transfers, the ensemble bought nothing.")
print( "         Use +0.045 to project the temporal run. Expected outcome.")
print(f"  >{THIS_CV + OFFSET + 0.02:.3f}  notes/20 §1 is right that leave-one-embryo-out is")
print( "         pessimistic. The temporal run should ensemble too, and project with")
print( "         the larger combined offset measured here.")
print(f"  <{THIS_CV + OFFSET - 0.02:.3f}  the learned path loses more to the hidden set than the")
print( "         classical one does. That RAISES the CV bar a temporal model must")
print( "         clear before it is worth submitting.")
print(f"\n  In every case 0.752 stands as the banked score; this cannot lower it.")

(WORK / "claude_submit_unet_summary.json").write_text(json.dumps(
    {k: v for k, v in summary.items() if k != "names"} |
    {"n_test": len(TEST_NAMES), "n_fit": len(fit_used),
     "loss": CHOSEN, "ensemble": [e for e, _ in ENSEMBLE],
     "budget_fill": CFG.budget_fill, "refine": CFG.refine,
     "sec_per_volume": SEC_PER_VOLUME,
     "this_cv": THIS_CV, "projected_lb": THIS_CV + OFFSET,
     "beta": BETA.tolist(), "feat_names": FEAT_NAMES,
     "hours": (time.time() - T_START) / 3600}, indent=2))
print("\nwrote claude_submit_unet_summary.json")
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
