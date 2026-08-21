"""Build notebooks/09_submission.ipynb — the first real submission."""
import ast, json
from pathlib import Path
OUT = Path("/workspace/biohub-cell_tracking_during_development/notebooks/09_submission.ipynb")
CELLS = []
def md(src): CELLS.append({"cell_type":"markdown","metadata":{},"source":src.strip("\n").splitlines(keepends=True)})
def code(src): CELLS.append({"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.strip("\n").splitlines(keepends=True)})

md(r"""
# Submission — `adaptive_predicted`, CV 0.7070

Eight experiments, no leaderboard score. This is the one that fixes that.

The configuration is the gated chain from `notes/14`: DoG detection at
`min_separation_um` chosen **per dataset** so the detector emits about 1.2x that
dataset's node budget per frame, isolated nodes pruned, Hungarian linking.

| step | CV | gate |
|---|---|---|
| `04` DoG sep 6.0 | 0.6760 | chain start |
| + prune isolated nodes | 0.6896 | PROMOTE +0.0136 |
| + adaptive density from a **predicted** budget | **0.7070** | PROMOTE +0.0174 |

## What makes this notebook different from `01`–`08`

Those ran interactively with internet on. **A scored rerun has internet off**, so this
notebook installs nothing — if a package is not already in the image, it is not
available. Three consequences shaped the code:

- **No `geff`.** It was only ever read for `estimated_number_of_nodes`. That number is
  now predicted from the image, and where train's true budgets are needed to fit the
  regression, `estimated_total_nodes` reads the `.geff`'s zarr attrs directly.
- **No `polars`** on the writing path. `harness.csvout` streams the CSV with the stdlib.
- **`test/` is swapped at rerun**, so dataset names are globbed, never hardcoded, and
  `max_frames` is never set.

That leaves `numpy`, `scipy` and `zarr`. **numpy and scipy ship in the Kaggle image;
`zarr` does not** — measured 2026-08-21 on the current image (python 3.12.13, numpy
2.0.2, scipy 1.16.3), where `import zarr` fails. An older probe notebook said otherwise,
but Kaggle rotates images and that reading is now stale (`notes/07`).

**So this notebook needs a wheelhouse dataset attached.** Build it once with
`10_wheelhouse.ipynb` (internet ON), save its output as a Dataset, and add that Dataset
as an input here. Cell 1 then installs from it with `--no-index`, which touches no
network and so works in a scored rerun. Downloading the wheels *on Kaggle* is what
guarantees they match its python and platform — wheels built anywhere else may not load.

## The one thing that must not be got wrong

`notes/14` §2: applying a single constant budget to every dataset scores **0.0882 below
doing nothing at all**. The regression has to be fit here, at runtime, on whatever
`train/` the rerun mounts. A hardcoded coefficient vector is the failure mode this
notebook is built to avoid.
""")

code(r"""
# No network install anywhere in this notebook. This cell reports what the image has.
import sys, os, time, platform, subprocess
from pathlib import Path
print(f"python {platform.python_version()}")

def probe():
    have, gone = {}, []
    for mod in ("numpy", "scipy", "zarr", "polars", "geff", "pandas"):
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

REQUIRED = ("numpy", "scipy", "zarr")
if any(m in missing for m in REQUIRED):
    # Fallback, expected never to fire: install from an attached wheelhouse dataset.
    # --no-index means pip never touches the network, so this works with internet off.
    # See the header for how to build the dataset (pip download, ON Kaggle, so the
    # wheels match its python and platform).
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
                            f"--find-links={w}", *[m for m in missing if m in REQUIRED]],
                           capture_output=True, text=True)
        print(f"  install from {w}: {'ok' if r.returncode == 0 else 'failed'}")
        if r.returncode != 0:
            print(r.stdout[-1500:]); print(r.stderr[-1500:])
    have, missing = probe()

still = [m for m in REQUIRED if m in missing]
if still:
    raise SystemExit(
        f"{', '.join(still)} not importable, and internet is off in a scored rerun so "
        "PyPI is unreachable. Build a wheelhouse dataset as described in the header "
        "(pip download, on Kaggle, with internet on) and attach it as an input."
    )
print("\nrequired packages present:", ", ".join(f"{m} {have[m]}" for m in REQUIRED))
print("not required by this notebook:",
      [m for m in ("polars", "geff", "pandas") if m in missing] or "none missing")
""")

code(r"""
import os, time, json, math
from pathlib import Path
import numpy as np

T_START = time.time()
TIME_BUDGET_S = 10.5 * 3600     # 12 h hard cap; leave 1.5 h of headroom
WORK = Path("/kaggle/working")

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
from harness.csvout import check_graph, write_submission
for need in ("budget_features", "open_movie"):
    if not hasattr(pc, need):
        raise SystemExit(f"Snapshot lacks pipeline.classical.{need} — re-upload the repo.")

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

sample = COMP / "sample_submission.csv"
if sample.exists():
    with sample.open() as fh:
        header = fh.readline().strip().split(",")
        named = {ln.split(",")[1] for ln in fh if "," in ln}
    print(f"  sample_submission columns: {header}")
    extra, absent = named - set(TEST_NAMES), set(TEST_NAMES) - named
    if extra or absent:
        print(f"  !! sample names differ from test/*.zarr — only in sample: {sorted(extra)[:5]}, "
              f"only on disk: {sorted(absent)[:5]}")
    else:
        print("  sample_submission names match test/*.zarr exactly")
""")

md("""## 1. Size the run before starting it

The forum's "scoring timeout" threads are really prediction-time threads — thread 724917,
answered by a grandmaster: *"this is not caused by scoring... All the other time goes into
constructing your prediction."* `08` measured 26–46 s per dataset for the pipeline and
1.3 s per detection for features, so the cost is predictable. Project it, and if the
projection does not fit, spend less on the part that is discretionary — the size of the
regression fit — rather than discovering the cap at hour twelve.
""")

code(r"""
SEC_PER_DETECTION = 1.3      # 08: 540 detections in 700 s
SEC_PER_DATASET = 45.0       # 08: 26 s (fixed sep) .. 46 s (adaptive, worst arm)

FRAC_FRAMES = (0.25, 0.5, 0.75)
REF_SEPS = (4.0, 8.0, 16.0)
FEAT_NAMES = ["n_sep4", "n_sep8", "n_sep16", "nstrong_sep4", "nstrong_sep8",
              "nstrong_sep16", "mean_int", "frac_fg"]

def project(n_fit, n_det):
    return (n_fit + len(TEST_NAMES)) * n_det * SEC_PER_DETECTION + \
           len(TEST_NAMES) * SEC_PER_DATASET

N_FIT = len(TRAIN_NAMES)
n_det = len(FRAC_FRAMES) * len(REF_SEPS)
print(f"projection with the validated feature set ({n_det} detections/dataset, "
      f"fit on all {N_FIT}): {project(N_FIT, n_det)/3600:.1f} h")

# The fit size is the only discretionary term -- the pipeline has to see every test
# dataset. 08 reached 10.7% median error fitting on ~30 datasets per embryo, so there is
# room to cut here before the model degrades.
while project(N_FIT, n_det) > TIME_BUDGET_S and N_FIT > 40:
    N_FIT = max(40, int(N_FIT * 0.75))
    print(f"  trimming the fit to {N_FIT} train datasets -> {project(N_FIT, n_det)/3600:.1f} h")
if project(N_FIT, n_det) > TIME_BUDGET_S:
    print(f"  !! still projects {project(N_FIT, n_det)/3600:.1f} h against a "
          f"{TIME_BUDGET_S/3600:.1f} h budget. The pipeline itself dominates; nothing "
          "feature-side can fix that. Proceeding, but expect this to be tight.")

# Spread the fit across the corpus rather than taking a prefix, so both embryos are in it
# whatever the naming turns out to be.
FIT_NAMES = ([TRAIN_NAMES[i] for i in np.linspace(0, len(TRAIN_NAMES) - 1, N_FIT).astype(int)]
             if N_FIT < len(TRAIN_NAMES) else list(TRAIN_NAMES))
FIT_NAMES = sorted(set(FIT_NAMES))
print(f"\nfitting the budget regression on {len(FIT_NAMES)} train datasets")
""")

md("""## 2. Fit the budget regression on `train/`

`estimated_number_of_nodes` for the training datasets comes out of each `.geff`'s zarr
attrs — no `geff` package involved. If a rerun mounts a `train/` without ground truth,
this cell says so and the notebook stops, because a constant budget is worse than useless
(`notes/14` §2).
""")

code(r"""
CFG = Config(detector="dog", min_separation_um=6.0, dog_rel_threshold=0.005,
             dog_scales=[(1.5, 4.0), (2.5, 6.0)], footprint="ball",
             adaptive_separation=True, adaptive_target=1.2, prune_isolated_nodes=True)

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
              f"(08 measured 10.7% on its 60-dataset subset)")
else:
    print("  only one name prefix in train — no held-out estimate available")
""")

md("""## 3. Predict, check, and stream to CSV

One dataset at a time: features → predicted budget → detect → link → prune → rows. Never
more than one graph in memory, and `check_graph` runs on each before it is written, so a
malformed graph is reported here rather than silently repaired by the scorer.
""")

code(r"""
def predict_test():
    t_run, starved = time.time(), False
    for i, n in enumerate(TEST_NAMES, 1):
        # A partial submission scores 0 on the datasets it skips. A submission that
        # never finishes scores nothing at all, so past the budget we keep emitting
        # rows and stop paying for predictions.
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
                             est_total_nodes=budget)
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

(WORK / "submission_summary.json").write_text(json.dumps(
    {k: v for k, v in summary.items() if k != "names"} |
    {"n_test": len(TEST_NAMES), "n_fit": len(fit_used),
     "beta": BETA.tolist(), "feat_names": FEAT_NAMES,
     "hours": (time.time() - T_START) / 3600}, indent=2))
print("wrote submission_summary.json — send it back with the log.")
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
