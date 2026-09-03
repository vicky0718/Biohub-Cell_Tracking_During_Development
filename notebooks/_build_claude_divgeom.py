"""Build notebooks/claude_divgeom.ipynb — check the division-geometry claim against our GT.

`nusrati/0-938`'s diff over `0-936` is entirely division geometry, and states two numbers as
fact in its own comments:

    SAFE_DIV_MAX_UM         7.0 -> 9.0   "Ground-truth parent-daughter links reach 10.4um;
                                          a 7um cap rejected ~25% of real links."
    SAFE_DIV_SISTER_MAX_UM  12.0 -> 14.0 "Ground-truth divisions have sister separations up
                                          to 13.7um (median 10.4, p90 13.0), so a 12um cap
                                          rejected ~29% of real divisions."

`pipeline/divisions.py` carries exactly those two gates -- `max_um` (parent to new daughter)
and `sister_max_um` (daughter to daughter) -- and defaults them to **4.5 and 6.8**, tighter
than even `0-936`. This project has never swept them against ground-truth statistics;
`notes/50` closed the division direction on the ILP weight axis alone. The claim is about
data we hold, so it is checkable rather than something to trust.

CPU only, no model, no GPU.

**All numeric work runs in a SUBPROCESS.** Three earlier versions of this notebook died
because the pack wheels replace numpy on disk while the notebook process already has numpy
loaded, leaving it inconsistent (`_center` missing from `numpy._core.umath`, then
`_blas_supports_fpe` missing from `_multiarray_umath`). Moving the parent's own `import
numpy` after the install was not enough -- papermill and IPython import numpy before any
cell runs. Every working notebook in this repo writes a worker script and launches a fresh
interpreter; this one now does the same, and the parent never imports numpy at all.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claude_divgeom.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# What do real divisions actually look like?

`nusrati/0-938` beats `0-936` on division geometry alone, and states two numbers as fact:

```
SAFE_DIV_MAX_UM         7.0 -> 9.0    "GT parent-daughter links reach 10.4um;
                                       a 7um cap rejected ~25% of real links"
SAFE_DIV_SISTER_MAX_UM  12.0 -> 14.0  "GT sister separations up to 13.7um
                                       (median 10.4, p90 13.0); a 12um cap
                                       rejected ~29% of real divisions"
```

`pipeline/divisions.py` carries **exactly those two gates**, and defaults them to
**`max_um=4.5`, `sister_max_um=6.8`** — tighter than even `0-936`'s 7.0/12.0. This project
has never swept them against ground truth: `notes/50` closed divisions on the ILP weight
axis, having found `div_J` 0.1154 identical across eight post-processing chains. The
geometry gates were never in that grid.

This is a claim about data we hold, so it does not need to be trusted.

**Caveat carried into the reading:** `notes/43` counted **151 division events across 199
datasets**. Every quantile below rests on that n, and a "p90" over ~150 points is a handful
of events. Read it with n attached, not as a smooth distribution.

## Pre-registered predictions

1. **The parent-daughter claim reproduces.** Max GT parent-daughter separation within 1 µm
   of 10.4, and a 7 µm cap rejects 20–30% of real links.
2. **The sister claim reproduces.** Median within 1 µm of 10.4, p90 within 1 µm of 13.0, max
   within 1 µm of 13.7, and a 12 µm cap rejects 25–35% of divisions.
3. **Our gates are too tight.** `pipeline/divisions.py`'s defaults reject more than 10% of
   real divisions on at least one axis.
4. **The two embryos agree** — sister medians within 2 µm, otherwise one global gate cannot
   serve both and the fix is per-embryo rather than a value.

*If 1 and 2 fail, their numbers do not describe this training set and the 0.936 -> 0.938
step is not the geometry they claim. Worth knowing before porting anything.*
""")

code(r"""
import os, subprocess, sys, time
from pathlib import Path
# NOTE: this cell deliberately does NOT import numpy. The pack wheels replace numpy on
# disk, and anything already holding it goes inconsistent. All numeric work happens in the
# worker below, in a fresh interpreter started AFTER the install.

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

md("""## 1. Measure every ground-truth division, in a fresh interpreter""")

code(r'''
WORKER = WORK / "run_divgeom.py"
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
REC, FAILED = [], []
for i, name in enumerate(names, 1):
    try:
        gt = read_geff(TRAIN / (name + ".geff"))
        sc = np.asarray(read_scale(TRAIN / (name + ".zarr")), float)
    except Exception as e:
        FAILED.append(name + ": " + type(e).__name__ + " " + str(e))
        print("  !! " + FAILED[-1], flush=True)
        if len(FAILED) >= 3 and not REC:
            raise SystemExit("first 3 datasets all failed: " + FAILED[0])
        continue
    if len(gt.edges) == 0:
        continue
    edges = np.asarray(gt.edges)
    src = edges[:, 0]
    out_deg = np.bincount(src, minlength=len(gt.t))
    um = np.asarray(gt.zyx, float) * sc
    for p in np.flatnonzero(out_deg >= 2):
        kids = edges[src == p][:, 1]
        if len(kids) != 2:
            REC.append(dict(name=name, n_kids=int(len(kids))))
            continue
        a, b = int(kids[0]), int(kids[1])
        pa = float(np.linalg.norm(um[a] - um[p]))
        pb = float(np.linalg.norm(um[b] - um[p]))
        REC.append(dict(name=name, n_kids=2, pd_a=pa, pd_b=pb, pd_max=max(pa, pb),
                        ss=float(np.linalg.norm(um[a] - um[b])),
                        dt=int(gt.t[a] - gt.t[p]),
                        asym=abs(pa - pb) / max((pa + pb) / 2.0, 1e-9)))
    if i % 50 == 0:
        print("  " + str(i) + "/" + str(len(names)) + "  " + str(len(REC))
              + " divisions  " + str(int(time.time() - T0)) + "s", flush=True)

json.dump(REC, open("/kaggle/working/divgeom.json", "w"), default=float)
print("\\n" + str(len(REC)) + " GT divisions across " + str(len(names))
      + " datasets in " + str(int(time.time() - T0)) + "s", flush=True)
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

md("""## 2. The distributions, and what our gates would reject""")

code(r'''
GRADER = WORK / "grade_divgeom.py"
GRADER.write_text("""
import json, sys
from pathlib import Path
import numpy as np

REPO = Path("__REPO__")
sys.path.insert(0, str(REPO))
REC = json.load(open("/kaggle/working/divgeom.json"))
binary = [r for r in REC if r.get("n_kids") == 2]
print(str(len(REC)) + " divisions, " + str(len(binary)) + " binary ("
      + str(len(REC) - len(binary)) + " with !=2 children)")
if not binary:
    raise SystemExit("no binary divisions found")

pd_max = np.array([r["pd_max"] for r in binary])
ss = np.array([r["ss"] for r in binary])
dt = np.array([r["dt"] for r in binary])
asym = np.array([r["asym"] for r in binary])

def dist(label, v):
    q = np.percentile(v, [50, 75, 90, 95])
    print("  %-22s n=%-5d min %6.2f  median %6.2f  p75 %6.2f  p90 %6.2f  p95 %6.2f  max %6.2f"
          % (label, len(v), v.min(), q[0], q[1], q[2], q[3], v.max()))

print("\\ndistances (um):")
dist("parent->daughter (max)", pd_max)
dist("sister<->sister", ss)
print("  frame gap dt: " + str(np.bincount(dt[dt >= 0]).tolist())
      + "   asymmetry median %.3f" % float(np.median(asym)))

print("\\nwhat each gate REJECTS:")
print("  %-30s%10s%8s%9s" % ("gate", "rejects", "of", "pct"))
for lbl, v, caps in (("max_um (parent->daughter)", pd_max, (4.5, 7.0, 9.0, 11.0, 14.0)),
                     ("sister_max_um", ss, (6.8, 11.0, 12.0, 14.0, 16.0))):
    for c in caps:
        n = int((v > c).sum())
        print("  %-30s%10d%8d%8.1f%%" % (lbl + " > " + str(c), n, len(v), 100.0 * n / len(v)))

print("\\nTHEIR CLAIMS vs OURS:")
claims = [("parent-daughter max ~10.4um", float(pd_max.max()), 10.4, 1.0),
          ("7um rejects ~25% of links", float((pd_max > 7.0).mean() * 100), 25.0, 8.0),
          ("sister median ~10.4um", float(np.median(ss)), 10.4, 1.0),
          ("sister p90 ~13.0um", float(np.percentile(ss, 90)), 13.0, 1.0),
          ("sister max ~13.7um", float(ss.max()), 13.7, 1.0),
          ("12um rejects ~29% of divisions", float((ss > 12.0).mean() * 100), 29.0, 8.0)]
ok = []
for lbl, got, want, tol in claims:
    hit = abs(got - want) <= tol
    ok.append(hit)
    print("  %-34s ours %7.2f  theirs %6.1f  %s"
          % (lbl, got, want, "MATCH" if hit else "DIFFERS"))

print("\\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)
ok1 = ok[0] and ok[1]
ok2 = ok[2] and ok[3] and ok[4] and ok[5]
print("\\n1. the parent-daughter claim reproduces  ->  " + ("PASS" if ok1 else "FAIL"))
print("2. the sister claim reproduces  ->  " + ("PASS" if ok2 else "FAIL"))

import inspect
from pipeline import divisions as DV
cur_max = cur_sis = None
try:
    sig = inspect.signature(DV.DivisionParams)
    for k, p in sig.parameters.items():
        if "sister" in k and isinstance(p.default, (int, float)):
            cur_sis = float(p.default)
        elif "max_um" in k and isinstance(p.default, (int, float)):
            cur_max = float(p.default)
except Exception as e:
    print("  (could not read DivisionParams: " + str(e) + ")")
print("\\n3. our gates are too tight (>10% of real divisions rejected)")
print("   pipeline/divisions.py defaults: max_um=" + str(cur_max)
      + "  sister_max_um=" + str(cur_sis))
rej = []
if cur_max: rej.append(("max_um", float((pd_max > cur_max).mean())))
if cur_sis: rej.append(("sister_max_um", float((ss > cur_sis).mean())))
for k, v in rej:
    print("   %s rejects %.1f%% of real divisions" % (k, 100.0 * v))
ok3 = any(v > 0.10 for _, v in rej)
print("   ->  " + ("PASS -- loosen them" if ok3 else "FAIL -- already wide enough"))

print("\\n4. the two embryos agree (sister medians within 2um)")
meds = {}
for e in sorted(set(r["name"].split("_")[0] for r in binary)):
    v = np.array([r["ss"] for r in binary if r["name"].startswith(e)])
    if len(v):
        meds[e] = (float(np.median(v)), len(v))
for e, (m, n) in meds.items():
    print("   %-8s n=%-5d sister median %.2f" % (e, n, m))
vals = [m for m, _ in meds.values()]
ok4 = len(vals) > 1 and (max(vals) - min(vals)) <= 2.0
print("   spread %.2fum  ->  %s" % (max(vals) - min(vals), "PASS" if ok4 else "FAIL"))
if not ok4:
    print("   One global gate cannot serve both embryos; the fix is per-embryo.")

print("\\n" + "=" * 78)
print(str(sum([ok1, ok2, ok3, ok4])) + "/4 passed   (n = " + str(len(binary))
      + " binary divisions)")
if ok3:
    print("RECOMMENDED GATES: max_um >= %.0f, sister_max_um >= %.0f  (p99 of real divisions)"
          % (np.ceil(np.percentile(pd_max, 99)), np.ceil(np.percentile(ss, 99))))
else:
    print("Our gates already admit essentially every real division; the 0.936->0.938 step")
    print("is not reachable by loosening ours, and something else in their diff carries it.")
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
