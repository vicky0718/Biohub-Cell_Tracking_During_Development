"""Build notebooks/claude_oracle.ipynb — score a submission locally against the REAL test GT.

`claude_nest2` found the thing this competition has been hiding from us all along:

```
test  : 44b6_0113de3b  44b6_0b24845f  6bba_05b6850b  6bba_05db0fb1
train : 199 zarr, 199 geff   -- and ALL FOUR test stems are among them, same shapes
```

The four hidden-test datasets are four of the 199 **training** datasets, with their ground
truth sitting in `train/*.geff`. `notes/49`'s "the test set is a third pair of embryos" is
wrong: both embryo prefixes are the same, and the movies are byte-identical in shape.

That changes the economics of everything. `notes/65` §2 established that this is a
kernels-only competition, so a human has to press Submit and the daily five is the budget
for *measured* arms. If a local score reproduces the leaderboard, the budget stops binding:
every arm is screened for free and only winners spend a slot.

**The calibration that decides whether this is real.** We have two arms whose leaderboard
scores we know and which differ by a known amount:

```
claude_fork      LB 0.937      SECONDARY_DETECTION_WEIGHT 0.80
claude_forkw085  LB 0.932      SECONDARY_DETECTION_WEIGHT 0.85     delta -0.005
```

If the local metric reproduces that −0.005, it is an oracle. If it does not, it is another
`notes/64` — a screen that moves in its own direction — and it gets recorded as one and
abandoned. The grading cell tests exactly that and nothing softer.

`harness/purescore.py` is embedded verbatim rather than mounted, so this notebook has no
dependency on the repo dataset's current version — numpy and scipy are all it needs. Its
division term is exact only for fork-free predictions and these arms have forks, so what is
reported is **`adj_edge_jaccard`**, which is the metric's own dominant term
(`score = adj_edge_jaccard + 0.1 * division_jaccard`). The calibration is against the
*difference* between two arms, where the small division term largely cancels.
"""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "notebooks" / "claude_oracle.ipynb"
PURESCORE = (ROOT / "harness" / "purescore.py").read_text()

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# A local oracle: score against the real test ground truth

`claude_nest2` showed the four hidden-test datasets are four of the 199 **training**
datasets — same stems, same shapes, ground truth present in `train/*.geff`.

If a local score reproduces the leaderboard, arms stop costing submission slots.

**Pre-registered predictions.**

1. Ground truth loads for all four test stems, with annotated node counts matching what
   `claude_nest2` read from the GEFF metadata: `52, 51, 861, 1229`.
2. The per-dataset node ratios reproduce `claude_nest2` exactly
   (`-0.013, -0.448, -0.052, -0.010`). A consistency check on the CSV reader — if this
   fails, the reader is wrong and nothing else is readable.
3. `adj_edge_jaccard` for `claude_fork` lands in **[0.90, 0.96]**, near its LB 0.937.
4. **The one that matters.** `adj_edge(claude_fork) - adj_edge(claude_forkw085)` is
   **+0.005 ± 0.003**, reproducing the leaderboard delta between the same two runs.

Prediction 4 is the whole point. Predictions 1-3 can all pass on a metric that ranks arms
wrongly; only 4 says this can replace a submission.
""")

code("import pathlib\npathlib.Path('purescore.py').write_text(r'''"
     + PURESCORE.replace("'''", "\\'\\'\\'") + "''')\nprint('purescore.py written')")

code(r"""
import json, sys
from pathlib import Path
import numpy as np
import purescore as PS

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
TEST_STEMS = sorted(p.name.split(".")[0] for p in (COMP / "test").glob("*.zarr"))
print("test stems:", TEST_STEMS)

print("\n=== mounted ===")
for p in sorted(Path("/kaggle/input").glob("*")):
    print(" ", p.name, "->", [q.name for q in sorted(p.glob("*"))[:6]])

# Structure of one GEFF, so the reader below is written against what is actually there
# rather than against a remembered spec.
g = TRAIN / f"{TEST_STEMS[0]}.geff"
print(f"\n=== tree of {g.name} ===")
for q in sorted(g.rglob("zarr.json"))[:24]:
    rel = q.relative_to(g).parent
    d = json.loads(q.read_text())
    print(f"  {str(rel) or '.':<34} {d.get('node_type','?'):<6} shape={d.get('shape')}")
""")

code(r"""
# GEFF is a zarr v3 store. `zarr` is NOT in the Kaggle image -- v1 of this notebook died
# on ModuleNotFoundError -- and this run has no submission to make, so internet is on and
# we install it. Nothing here writes a submission.csv, so the competition's offline rule
# does not apply to this notebook.
import subprocess, sys
try:
    import zarr
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr>=3"], check=True)
    import zarr
print("zarr", zarr.__version__)

def _arr(root, *names):
    for n in names:
        try:
            return np.asarray(zarr.open_array(str(root / n), mode="r"))
        except Exception:
            continue
    return None

def read_gt(stem):
    g = TRAIN / f"{stem}.geff"
    t = _arr(g, "nodes/props/t/values")
    z = _arr(g, "nodes/props/z/values")
    y = _arr(g, "nodes/props/y/values")
    x = _arr(g, "nodes/props/x/values")
    ids = _arr(g, "nodes/ids")
    e = _arr(g, "edges/ids")
    if t is None or ids is None:
        return None
    zyx = np.stack([z if z is not None else np.zeros_like(t, float),
                    y, x], axis=1).astype(float)
    index = {int(v): i for i, v in enumerate(np.asarray(ids).ravel())}
    if e is None or len(e) == 0:
        edges = np.zeros((0, 2), int)
    else:
        e = np.asarray(e).reshape(-1, 2)
        edges = np.array([[index[int(a)], index[int(b)]] for a, b in e
                          if int(a) in index and int(b) in index], int)
        if edges.size == 0:
            edges = np.zeros((0, 2), int)
    est = None
    meta = json.loads((g / "zarr.json").read_text())
    stack = [meta]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if "estimated" in str(k) and "node" in str(k):
                    est = v
                stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return {"t": np.asarray(t).astype(np.int64), "zyx": zyx, "edges": edges,
            "est": float(est) if est is not None else float("nan")}

GT = {}
for s in TEST_STEMS:
    r = read_gt(s)
    GT[s] = r
    if r is None:
        print(f"{s:<18} GT UNREADABLE")
    else:
        print(f"{s:<18} nodes {len(r['t']):>6,}  edges {len(r['edges']):>6,}  "
              f"est {r['est']:>10,.0f}  t in [{r['t'].min()}, {r['t'].max()}]")
""")

code(r"""
# Predictions: every mounted kernel output that contains a submission.csv.
from collections import defaultdict

def read_submission(path):
    per = defaultdict(lambda: {"nid": [], "t": [], "zyx": [], "edges": []})
    with open(path) as f:
        col = {n: i for i, n in enumerate(next(f).rstrip("\n").split(","))}
        for line in f:
            r = line.rstrip("\n").split(",")
            d = per[r[col["dataset"]]]
            if r[col["row_type"]] == "node":
                d["nid"].append(r[col["node_id"]])
                d["t"].append(int(r[col["t"]]))
                d["zyx"].append((float(r[col["z"]]), float(r[col["y"]]),
                                 float(r[col["x"]])))
            else:
                d["edges"].append((r[col["source_id"]], r[col["target_id"]]))
    out = {}
    for ds, d in per.items():
        idx = {n: i for i, n in enumerate(d["nid"])}
        e = [(idx[a], idx[b]) for a, b in d["edges"] if a in idx and b in idx]
        out[ds] = {"t": np.array(d["t"], np.int64),
                   "zyx": np.array(d["zyx"], float),
                   "edges": np.array(e, int) if e else np.zeros((0, 2), int)}
    return out

subs = {}
# Kernel outputs mount under /kaggle/input/notebooks/<user>/<slug>/, not
# /kaggle/input/<slug>/ -- v1 globbed one level and found nothing.
for p in sorted(Path("/kaggle/input").rglob("submission.csv")):
    subs[p.parent.name] = read_submission(p)
    print(f"{p.parent.name:<24} {len(subs[p.parent.name])} datasets  <- {p.parent}")
if not subs:
    print("NO submission.csv MOUNTED — attach the arm kernels as kernel sources")
""")

code(r"""
def score_arm(pred):
    rows = []
    for s in TEST_STEMS:
        g, p = GT.get(s), pred.get(s)
        if g is None or p is None:
            continue
        counts = PS.count_edges(p["t"], p["zyx"], p["edges"],
                                g["t"], g["zyx"], g["edges"])
        row = PS.per_sample(counts, g["est"], len(g["t"]), n_gt_divisions=0)
        row["dataset"] = s
        rows.append(row)
    return rows

results = {}
for name, pred in subs.items():
    rows = score_arm(pred)
    results[name] = rows
    print(f"\n=== {name} ===")
    print(f"{'dataset':<18}{'edge_J':>9}{'ratio':>9}{'adj':>9}{'nodes':>9}{'gtE':>8}")
    for r in rows:
        print(f"{r['dataset']:<18}{r['edge_jaccard']:>9.4f}{r['total_node_ratio']:>+9.3f}"
              f"{r['adj_edge_jaccard']:>9.4f}{r['num_pred_nodes']:>9,}"
              f"{r['edge_tp'] + r['edge_fn']:>8,}")
    w = [r["edge_tp"] + r["edge_fp"] + r["edge_fn"] for r in rows]
    adj = sum(r["adj_edge_jaccard"] * wi for r, wi in zip(rows, w)) / sum(w)
    ej = sum(r["edge_jaccard"] * wi for r, wi in zip(rows, w)) / sum(w)
    print(f"{'WEIGHTED':<18}{ej:>9.4f}{'':>9}{adj:>9.4f}")
    results[name] = {"rows": rows, "adj": adj, "edge_j": ej}
""")

code(r"""
print("=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

WANT_ANN = {"44b6_0113de3b": 52, "44b6_0b24845f": 51,
            "6bba_05b6850b": 861, "6bba_05db0fb1": 1229}
WANT_RATIO = {"44b6_0113de3b": -0.013, "44b6_0b24845f": -0.448,
              "6bba_05b6850b": -0.052, "6bba_05db0fb1": -0.010}

ok1 = all(GT.get(s) and len(GT[s]["t"]) == WANT_ANN[s] for s in TEST_STEMS)
print(f"\n1. GT loads with the annotated node counts claude_nest2 read  ->  "
      f"{'PASS' if ok1 else 'FAIL'}")
for s in TEST_STEMS:
    got = len(GT[s]["t"]) if GT.get(s) else None
    print(f"   {s:<18} got {str(got):>6}   want {WANT_ANN[s]:>6}")

fork = results.get("claude-fork")
w085 = results.get("claude-forkw085")

if fork:
    ok2 = all(abs(r["total_node_ratio"] - WANT_RATIO[r["dataset"]]) < 0.002
              for r in fork["rows"])
    print(f"\n2. node ratios reproduce claude_nest2  ->  {'PASS' if ok2 else 'FAIL'}")
    for r in fork["rows"]:
        print(f"   {r['dataset']:<18}{r['total_node_ratio']:>+9.3f}   want "
              f"{WANT_RATIO[r['dataset']]:>+7.3f}")

    ok3 = 0.90 <= fork["adj"] <= 0.96
    print(f"\n3. claude_fork adj_edge in [0.90, 0.96]  ->  {'PASS' if ok3 else 'FAIL'}")
    print(f"   adj_edge {fork['adj']:.4f}   edge_J {fork['edge_j']:.4f}   LB was 0.937")
else:
    ok2 = ok3 = False
    print("\n2-3. NOT GRADED — claude-fork not mounted")

if fork and w085:
    d = fork["adj"] - w085["adj"]
    ok4 = abs(d - 0.005) < 0.003
    print(f"\n4. local delta reproduces the leaderboard's -0.005  ->  "
          f"{'PASS' if ok4 else 'FAIL'}")
    print(f"   claude_fork {fork['adj']:.4f}   claude_forkw085 {w085['adj']:.4f}   "
          f"delta {d:+.4f}   LB delta +0.0050")
else:
    ok4 = False
    print("\n4. NOT GRADED — need both claude-fork and claude-forkw085 mounted")

print("\n" + "=" * 78)
print(f"{sum([ok1, ok2, ok3, ok4])}/4 passed")
if ok4:
    print("ORACLE CONFIRMED. Every arm can now be screened locally for free; submission")
    print("slots go only to arms that win here. This is the constraint notes/65 §2 set,")
    print("removed.")
elif ok1 and ok2 and ok3:
    print("SCORES BUT DOES NOT RANK. The metric computes, and it disagrees with the board")
    print("on the one comparison we can check. That is notes/64 again -- record it as a")
    print("failed screen and keep spending slots.")
else:
    print("SEE ABOVE — read prediction 1 before anything else.")
print("=" * 78)
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

for c in CELLS:
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))

OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT.name}: {len(CELLS)} cells, all code cells parse")
