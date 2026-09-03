"""Build notebooks/claude_divgeom.ipynb — check the division-geometry claim against our GT.

`nusrati/0-938`'s diff over `0-936` is entirely division geometry, and it states two
numbers as fact in its own comments:

    SAFE_DIV_MAX_UM         7.0 -> 9.0   "Ground-truth parent-daughter links reach 10.4um;
                                          a 7um cap rejected ~25% of real links."
    SAFE_DIV_SISTER_MAX_UM  12.0 -> 14.0 "Ground-truth divisions have sister separations up
                                          to 13.7um (median 10.4, p90 13.0), so a 12um cap
                                          rejected ~29% of real divisions."

`pipeline/divisions.py` carries exactly those two gates — `max_um` (parent to new daughter)
and `sister_max_um` (daughter to daughter) — and this project has **never swept them against
ground-truth statistics**. `notes/50` closed the division direction on the ILP weight axis
alone, having measured `div_J` 0.1154 identical across eight post-processing chains. The
geometry gates were never in that grid.

This is a claim about data we hold, so it is checkable without trusting the source. CPU
only, no model, no GPU — it reads the training `.geff` files and measures the distributions
directly. `notes/43` measured 151 division events across 199 datasets, so the sample is
small and the answer must be reported with that n attached rather than as a smooth quantile.

The output is the gate values our own divisions.py should use, or evidence that the claim
does not hold on the full training set.
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

`pipeline/divisions.py` carries **exactly those two gates** — `max_um` and `sister_max_um` —
and this project has never swept them against ground truth. `notes/50` closed divisions on
the ILP weight axis, having found `div_J` 0.1154 identical across eight post-processing
chains; the geometry gates were never in that grid.

This is a claim about data we hold, so it does not need to be trusted. CPU only.

**Caveat carried into the reading:** `notes/43` counted **151 division events across 199
datasets**. Every quantile below rests on that n, and a "p90" over ~150 points is a handful
of events. Report it with n attached; do not read it as a smooth distribution.

## Pre-registered predictions

1. **The parent-daughter claim reproduces.** Max GT parent-daughter separation is within
   1 µm of 10.4, and a 7 µm cap rejects 20–30% of real links.
2. **The sister claim reproduces.** Median sister separation within 1 µm of 10.4, p90 within
   1 µm of 13.0, max within 1 µm of 13.7, and a 12 µm cap rejects 25–35% of divisions.
3. **Our current gates are too tight.** `pipeline/divisions.py`'s defaults reject more than
   10% of real divisions on at least one of the two axes.
4. **The two embryos agree.** `44b6` and `6bba` give medians within 2 µm of each other —
   otherwise a single global gate cannot serve both and the fix is per-embryo, not a value.

*If 1 and 2 fail, their numbers do not describe this training set and the 0.936 -> 0.938
step is not the geometry they claim it is. That is worth knowing before porting anything.*
""")

code(r"""
# numpy is imported AFTER the wheel install, never before. v2 imported it at the top of
# this cell; pip then replaced numpy on disk and the already-loaded module went
# inconsistent -- `ImportError: cannot import name '_center' from numpy._core.umath` in a
# later cell. Every other notebook in this repo installs first and imports after.
import os, subprocess, sys, time, json
from pathlib import Path

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
print("repo:", REPO, "\ncomp:", COMP, "\npack:", PACK)
if REPO is None or COMP is None or PACK is None:
    raise SystemExit("missing mount")

# read_geff needs `geff`, which is not in the base image. v1 skipped this block and every
# one of the 199 datasets failed with ModuleNotFoundError while the mounts were fine.
ok = pip_install([str(p) for p in sorted((PACK / "wheels").glob("*.whl"))],
                 extra=("--no-index", f"--find-links={PACK/'wheels'}"))
print(f"pack wheels {'ok' if ok else 'FAILED'}")
probe = sh(sys.executable, "-c",
           "import numpy, zarr, geff; print('geff ok, numpy', numpy.__version__)")
print(probe.stdout.strip() or probe.stderr.strip()[-600:])
if probe.returncode != 0:
    raise SystemExit("geff does not import in a fresh interpreter")

import numpy as np      # first import in this process, and only now that pip has finished
sys.path.insert(0, str(REPO))
TRAIN = COMP / "train"

from harness.tracks import read_geff, read_scale
names = sorted(p.stem for p in TRAIN.glob("*.geff") if (TRAIN / (p.stem + ".zarr")).exists())
print(f"{len(names)} training datasets")
""")

md("""## 1. Measure every ground-truth division""")

code(r"""
T0 = time.time()
REC = []          # one row per GT division
FAILED = []
for i, name in enumerate(names, 1):
    try:
        gt = read_geff(TRAIN / f"{name}.geff")
        sc = np.asarray(read_scale(TRAIN / f"{name}.zarr"), float)
    except Exception as e:
        FAILED.append(f"{name}: {type(e).__name__} {e}")
        print(f"  !! {FAILED[-1]}")
        # Fail FAST. v1 caught per-dataset and ground through all 199 identical
        # ModuleNotFoundErrors before the analysis cell died on an empty record.
        if len(FAILED) >= 3 and not REC:
            raise SystemExit(f"first {len(FAILED)} datasets all failed: {FAILED[0]}")
        continue
    if len(gt.edges) == 0:
        continue
    src = np.asarray(gt.edges)[:, 0]
    out_deg = np.bincount(src, minlength=len(gt.t))
    parents = np.flatnonzero(out_deg >= 2)
    um = np.asarray(gt.zyx, float) * sc
    for p in parents:
        kids = np.asarray(gt.edges)[src == p][:, 1]
        if len(kids) != 2:            # >2 children is not a binary division; record and skip
            REC.append(dict(name=name, n_kids=int(len(kids)), pd_max=float("nan"),
                            ss=float("nan"), dt=-1))
            continue
        a, b = int(kids[0]), int(kids[1])
        pd_a = float(np.linalg.norm(um[a] - um[p]))
        pd_b = float(np.linalg.norm(um[b] - um[p]))
        REC.append(dict(name=name, n_kids=2,
                        pd_a=pd_a, pd_b=pd_b, pd_max=max(pd_a, pd_b),
                        ss=float(np.linalg.norm(um[a] - um[b])),
                        dt=int(gt.t[a] - gt.t[p]),
                        asym=abs(pd_a - pd_b) / max((pd_a + pd_b) / 2, 1e-9)))
    if i % 40 == 0:
        print(f"  {i}/{len(names)}  {len(REC)} divisions  {time.time()-T0:.0f}s", flush=True)

json.dump(REC, open("/kaggle/working/divgeom.json", "w"), default=float)
print(f"\n{len(REC)} GT divisions across {len(names)} datasets in {time.time()-T0:.0f}s")
""")

md("""## 2. The distributions, and what our gates would reject""")

code(r"""
import numpy as np
REC = json.load(open("/kaggle/working/divgeom.json"))
binary = [r for r in REC if r.get("n_kids") == 2]
print(f"{len(REC)} divisions, {len(binary)} binary  "
      f"({len(REC) - len(binary)} with !=2 children)")
if not binary:
    raise SystemExit("no binary divisions found")

pd_max = np.array([r["pd_max"] for r in binary])
ss = np.array([r["ss"] for r in binary])
dt = np.array([r["dt"] for r in binary])
asym = np.array([r["asym"] for r in binary])

def dist(label, v):
    q = np.percentile(v, [50, 75, 90, 95, 99])
    print(f"  {label:<22} n={len(v):<5} min {v.min():>6.2f}  median {q[0]:>6.2f}  "
          f"p75 {q[1]:>6.2f}  p90 {q[2]:>6.2f}  p95 {q[3]:>6.2f}  max {v.max():>6.2f}")

print("\ndistances (um):")
dist("parent->daughter (max)", pd_max)
dist("sister<->sister", ss)
print(f"\n  frame gap dt: {np.bincount(dt[dt >= 0]).tolist()}   asymmetry median {np.median(asym):.3f}")

print("\nwhat each gate REJECTS:")
print(f"  {'gate':<28}{'rejects':>10}{'of':>7}{'pct':>9}")
for lbl, v, caps in (("max_um (parent->daughter)", pd_max, (5.0, 7.0, 9.0, 11.0, 14.0)),
                     ("sister_max_um", ss, (7.0, 11.0, 12.0, 14.0, 16.0))):
    for c in caps:
        n = int((v > c).sum())
        print(f"  {lbl + ' > ' + str(c):<28}{n:>10}{len(v):>7}{n / len(v):>9.1%}")

print("\nTHEIR CLAIMS vs OURS:")
claims = [("parent-daughter max ~10.4um", pd_max.max(), 10.4, 1.0),
          ("7um rejects ~25% of links", (pd_max > 7.0).mean() * 100, 25.0, 8.0),
          ("sister median ~10.4um", float(np.median(ss)), 10.4, 1.0),
          ("sister p90 ~13.0um", float(np.percentile(ss, 90)), 13.0, 1.0),
          ("sister max ~13.7um", ss.max(), 13.7, 1.0),
          ("12um rejects ~29% of divisions", (ss > 12.0).mean() * 100, 29.0, 8.0)]
ok = []
for lbl, got, want, tol in claims:
    hit = abs(got - want) <= tol
    ok.append(hit)
    print(f"  {lbl:<34} ours {got:>7.2f}  theirs {want:>6.1f}  "
          f"{'MATCH' if hit else 'DIFFERS'}")

print("\n" + "=" * 78)
print("PREDICTION GRADING")
print("=" * 78)
ok1 = ok[0] and ok[1]
print(f"\n1. the parent-daughter claim reproduces  ->  {'PASS' if ok1 else 'FAIL'}")
ok2 = ok[2] and ok[3] and ok[4] and ok[5]
print(f"2. the sister claim reproduces  ->  {'PASS' if ok2 else 'FAIL'}")

# our own defaults, read from the module rather than retyped
sys.path.insert(0, str(REPO))
import inspect
from pipeline import divisions as DV
sig = inspect.signature(DV.DivisionParams) if hasattr(DV, "DivisionParams") else None
cur_max = cur_sis = None
if sig:
    for k, p in sig.parameters.items():
        if "sister" in k and isinstance(p.default, (int, float)): cur_sis = float(p.default)
        elif "max_um" in k and isinstance(p.default, (int, float)): cur_max = float(p.default)
print(f"\n3. our gates are too tight (>10% of real divisions rejected)")
print(f"   pipeline/divisions.py defaults: max_um={cur_max}  sister_max_um={cur_sis}")
rej = []
if cur_max: rej.append(("max_um", (pd_max > cur_max).mean()))
if cur_sis: rej.append(("sister_max_um", (ss > cur_sis).mean()))
for k, v in rej: print(f"   {k} rejects {v:.1%} of real divisions")
ok3 = any(v > 0.10 for _, v in rej)
print(f"   ->  {'PASS -- loosen them' if ok3 else 'FAIL -- our gates are already wide enough'}")

print("\n4. the two embryos agree (medians within 2um)")
meds = {}
for e in sorted({r["name"].split("_")[0] for r in binary}):
    v = np.array([r["ss"] for r in binary if r["name"].startswith(e)])
    if len(v): meds[e] = (float(np.median(v)), len(v))
for e, (m, n) in meds.items(): print(f"   {e}  n={n:<5} sister median {m:.2f}")
vals = [m for m, _ in meds.values()]
ok4 = len(vals) > 1 and (max(vals) - min(vals)) <= 2.0
print(f"   spread {max(vals) - min(vals):.2f}um  ->  {'PASS' if ok4 else 'FAIL'}")
if not ok4:
    print("   A single global gate cannot serve both embryos; the fix is per-embryo.")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4])}/4 passed   (n = {len(binary)} binary divisions)")
if ok3:
    hi_pd = float(np.percentile(pd_max, 99)); hi_ss = float(np.percentile(ss, 99))
    print(f"RECOMMENDED GATES: max_um >= {np.ceil(hi_pd):.0f}, "
          f"sister_max_um >= {np.ceil(hi_ss):.0f}  (p99 of real divisions)")
else:
    print("Our gates already admit essentially every real division; the 0.936->0.938 step")
    print("is not reachable by loosening ours, and something else in their diff carries it.")
print("=" * 78)
""")

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
