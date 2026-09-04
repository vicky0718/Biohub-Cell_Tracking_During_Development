"""Build notebooks/claude_linkgeom.ipynb — are our LINKING gates as wrong as the division ones?

`notes/57` found `pipeline/divisions.py`'s geometry gates rejecting **88% of real
divisions**. They were adopted from a public notebook's published constants and never
checked against data we hold. The same class of constant governs linking, and none of these
has ever been measured against the ground-truth displacement distribution:

    pipeline/repair.py   close_gaps       max_um = 5.75    (bridges a 2-frame hole)
                         cap_edge_length  max_um = 14.0    (drops over-long edges)
                         linefit_smooth   max_shift_um = 3.2

`notes/58`: hengck23 posted GT link statistics from 63,751 links and flagged his own
extremes as suspicious (`min dz -37, max +35  ###???`). He also warns separately that GT
tracks *freeze after the same frame indices* and that some annotations are interpolated
between labelled frames. So this run measures two things at once — where the gates should
sit, and whether the tail is real motion or bad labels.

Same CPU-only subprocess structure as `claude_divgeom`, for the reason recorded there: the
pack wheels replace numpy on disk and the notebook process already holds it loaded, so all
numeric work runs in a fresh interpreter and the parent never imports numpy.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claude_linkgeom.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Are our linking gates as wrong as the division ones were?

`notes/57` measured `pipeline/divisions.py`'s gates against ground truth and found them
rejecting **88.7%** and **86.8%** of real divisions. They came from a public notebook's
published constants and were never checked. **The same class of constant governs linking:**

```
pipeline/repair.py   close_gaps       max_um = 5.75    bridges a 2-frame hole
                     cap_edge_length  max_um = 14.0    drops over-long edges
                     linefit_smooth   max_shift_um = 3.2
```

None has been measured against the GT displacement distribution.

`notes/58` (hengck23, 63,751 links) gives a reference shape and flags its own extremes:

```
percentiles [50,90,95,99]   dz [1, 2, 2, 4]   dy [0.5, 1.25, 1.75, 3.0]   dx [0.5, 1.5, 2.0, 3.75]
min dz -37,  max +35   ###???        <- his flag
```

He separately warns that GT tracks **freeze after the same frame indices** and that some
annotations are **interpolated** between labelled frames. So the tail matters twice over:
it sets the gates, and it says whether the labels are trustworthy.

## Pre-registered predictions

1. **`close_gaps`'s 5.75 µm is too tight.** It bridges a **2-frame** hole, so the right
   comparison is against **2×** a single-frame displacement. More than 10% of real
   two-frame spans exceed 5.75 µm.
2. **`cap_edge_length`'s 14.0 µm is safe** — it drops under 1% of real single-frame links.
   If it drops more, we have been deleting real edges at the top of the chain.
3. **The distribution has a hard tail.** p99 of single-frame displacement is more than 4×
   the median — the signature hengck23 flagged, and the reason a mean-based gate misleads.
4. **Zero-displacement links exist in bulk** (>2% exactly 0.0 µm), which is the fingerprint
   of frozen frames or interpolated annotation rather than real motion.
5. **The two embryos agree** — median single-frame displacement within 1 µm. Otherwise a
   global gate cannot serve both, as `notes/57` found for divisions.

*Every figure is reported per embryo as well as pooled. `notes/49`: the test set is a third
pair of embryos and a pooled number says nothing about it.*
""")

code(r"""
# numpy is imported AFTER the wheel install, never before, and all numeric work runs in a
# subprocess. See notes/56-57: the pack wheels replace numpy on disk while the notebook
# process already holds it loaded, and papermill imports numpy before any cell runs.
import os, subprocess, sys, time
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
        print(r.stdout[-1200:]); print(r.stderr[-1200:])
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

REPO = find_dir(lambda p: (p / "harness").is_dir() and (p / "pipeline").is_dir(),
                ["/kaggle/input"])
COMP = find_dir(lambda p: (p / "train").is_dir() and (p / "test").is_dir()
                and any((p / "train").glob("*.zarr")), ["/kaggle/input"])
PACK = find_dir(lambda p: (p / "repo").is_dir() and (p / "weights").is_dir(), ["/kaggle/input"])
for lbl, v in (("repo", REPO), ("comp", COMP), ("pack", PACK)):
    print(f"  {lbl:<6} {v}")
if None in (REPO, COMP, PACK):
    raise SystemExit("missing mount")

ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")
probe = sh(sys.executable, "-c",
           "import numpy, zarr, geff; print('fresh interpreter ok, numpy', numpy.__version__)")
print(probe.stdout.strip() or probe.stderr.strip()[-700:])
if probe.returncode != 0:
    raise SystemExit("the dependency stack does not import in a fresh interpreter")
print(f"setup {time.time()-T_START:.0f}s")
""")

md("""## 1. Measure every ground-truth link, in a fresh interpreter""")

code(r'''
WORKER = WORK / "run_linkgeom.py"
WORKER.write_text("""
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("__REPO__"); TRAIN = Path("__COMP__") / "train"
sys.path.insert(0, str(REPO))
from harness.tracks import read_geff, read_scale

names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / (p.stem + ".zarr")).exists())
print(len(names), "training datasets", flush=True)

T0 = time.time()
OUT, FAILED = [], []
for i, name in enumerate(names, 1):
    try:
        gt = read_geff(TRAIN / (name + ".geff"))
        sc = np.asarray(read_scale(TRAIN / (name + ".zarr")), float)
    except Exception as e:
        FAILED.append(name + ": " + type(e).__name__ + " " + str(e))
        print("  !! " + FAILED[-1], flush=True)
        if len(FAILED) >= 3 and not OUT:
            raise SystemExit("first 3 datasets all failed: " + FAILED[0])
        continue
    if len(gt.edges) == 0:
        continue
    e = np.asarray(gt.edges)
    um = np.asarray(gt.zyx, float) * sc
    t = np.asarray(gt.t)
    dt = t[e[:, 1]] - t[e[:, 0]]
    d = um[e[:, 1]] - um[e[:, 0]]
    dist = np.linalg.norm(d, axis=1)
    keep = dt == 1
    OUT.append(dict(name=name, embryo=name.split("_")[0],
                    n_edges=int(len(e)), n_dt1=int(keep.sum()),
                    dist=[float(x) for x in dist[keep]],
                    dz=[float(x) for x in d[keep][:, 0]],
                    dy=[float(x) for x in d[keep][:, 1]],
                    dx=[float(x) for x in d[keep][:, 2]],
                    n_zero=int((dist[keep] < 1e-9).sum()),
                    dt_counts={str(int(k)): int(v) for k, v in
                               zip(*np.unique(dt, return_counts=True))}))
    if i % 50 == 0:
        print("  " + str(i) + "/" + str(len(names)) + "  "
              + str(sum(r["n_dt1"] for r in OUT)) + " links  "
              + str(int(time.time() - T0)) + "s", flush=True)

json.dump(OUT, open("/kaggle/working/linkgeom.json", "w"), default=float)
print("\\n" + str(sum(r["n_dt1"] for r in OUT)) + " single-frame links across "
      + str(len(OUT)) + " datasets in " + str(int(time.time() - T0)) + "s", flush=True)
""".replace("__REPO__", str(REPO)).replace("__COMP__", str(COMP)))

t0 = time.time()
proc = subprocess.Popen([sys.executable, "-u", str(WORKER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
print(f"\nworker exited {rc} after {time.time()-t0:.0f}s")
if rc != 0:
    raise SystemExit(f"worker failed ({rc})")
''')

md("""## 2. Where the gates should sit""")

code(r'''
GRADER = WORK / "grade_linkgeom.py"
GRADER.write_text("""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("__REPO__")
sys.path.insert(0, str(REPO))
R = json.load(open("/kaggle/working/linkgeom.json"))
dist = np.concatenate([np.asarray(r["dist"]) for r in R if r["dist"]])
dz = np.concatenate([np.asarray(r["dz"]) for r in R if r["dz"]])
dy = np.concatenate([np.asarray(r["dy"]) for r in R if r["dy"]])
dx = np.concatenate([np.asarray(r["dx"]) for r in R if r["dx"]])
n_zero = sum(r["n_zero"] for r in R)
print(str(len(dist)) + " single-frame GT links across " + str(len(R)) + " datasets")

dtc = {}
for r in R:
    for k, v in r["dt_counts"].items():
        dtc[k] = dtc.get(k, 0) + v
print("dt histogram: " + str(dict(sorted(dtc.items(), key=lambda x: int(x[0])))))

def dist_row(label, v):
    q = np.percentile(np.abs(v), [50, 90, 95, 99])
    print("  %-14s median %6.2f  p90 %6.2f  p95 %6.2f  p99 %6.2f  max %7.2f"
          % (label, q[0], q[1], q[2], q[3], np.abs(v).max()))

print("\\nsingle-frame displacement (um):")
dist_row("euclidean", dist)
dist_row("|dz|", dz); dist_row("|dy|", dy); dist_row("|dx|", dx)

p = np.percentile(dist, [50, 90, 95, 99])
print("\\nOUR GATES vs the data:")
print("  %-42s%10s%9s" % ("gate", "rejects", "pct"))
def rej(lbl, v, cap):
    n = int((v > cap).sum())
    print("  %-42s%10d%8.2f%%" % (lbl, n, 100.0 * n / max(len(v), 1)))
    return n / max(len(v), 1)
# close_gaps bridges a TWO-frame hole, so compare against 2x a single-frame step.
two = dist * 2.0
r_gap = rej("close_gaps max_um=5.75  (vs 2-frame span)", two, 5.75)
r_cap = rej("cap_edge_length max_um=14.0 (1-frame)", dist, 14.0)
r_sm  = rej("linefit_smooth max_shift_um=3.2 (1-frame)", dist, 3.2)

print("\\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)
ok1 = r_gap > 0.10
print("\\n1. close_gaps 5.75um is too tight for a 2-frame span (>10% rejected)")
print("   rejects %.1f%% of real two-frame spans  ->  %s"
      % (100 * r_gap, "PASS -- widen it" if ok1 else "FAIL -- it is wide enough"))
ok2 = r_cap < 0.01
print("\\n2. cap_edge_length 14.0um is safe (<1% of real links dropped)")
print("   rejects %.2f%%  ->  %s" % (100 * r_cap, "PASS" if ok2 else "FAIL -- deleting real edges"))
ok3 = p[3] > 4.0 * max(p[0], 1e-9)
print("\\n3. the distribution has a hard tail (p99 > 4x median)")
print("   median %.2f  p99 %.2f  ratio %.1fx  ->  %s"
      % (p[0], p[3], p[3] / max(p[0], 1e-9), "PASS" if ok3 else "FAIL"))
ok4 = n_zero / max(len(dist), 1) > 0.02
print("\\n4. zero-displacement links exist in bulk (>2%) -- frozen/interpolated GT")
print("   %d of %d exactly 0.0um  (%.2f%%)  ->  %s"
      % (n_zero, len(dist), 100.0 * n_zero / max(len(dist), 1), "PASS" if ok4 else "FAIL"))
if ok4:
    print("   hengck23 (notes/58): GT tracks freeze after the same frame indices, and some")
    print("   annotations are interpolated between labelled frames. Confirmed here.")

print("\\n5. the two embryos agree (median within 1um)")
meds = {}
for e in sorted(set(r["embryo"] for r in R)):
    v = np.concatenate([np.asarray(r["dist"]) for r in R if r["embryo"] == e and r["dist"]])
    meds[e] = (float(np.median(v)), len(v))
    print("   %-8s n=%-8d median %.2f  p99 %.2f" % (e, len(v), meds[e][0],
                                                    float(np.percentile(v, 99))))
vals = [m for m, _ in meds.values()]
ok5 = len(vals) > 1 and (max(vals) - min(vals)) <= 1.0
print("   spread %.2fum  ->  %s" % (max(vals) - min(vals), "PASS" if ok5 else "FAIL"))

print("\\n" + "=" * 78)
print(str(sum([ok1, ok2, ok3, ok4, ok5])) + "/5 passed   (n = " + str(len(dist)) + " links)")
print("RECOMMENDED: close_gaps max_um >= %.1f  (p95 of 2-frame spans)"
      % float(np.percentile(two, 95)))
print("             cap_edge_length max_um >= %.1f  (p99 of 1-frame)"
      % float(np.percentile(dist, 99)))
print("=" * 78)
""".replace("__REPO__", str(REPO)))

proc = subprocess.Popen([sys.executable, "-u", str(GRADER)],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line.rstrip(), flush=True)
rc = proc.wait()
if rc != 0:
    raise SystemExit(f"grader failed ({rc})")
''')

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")
