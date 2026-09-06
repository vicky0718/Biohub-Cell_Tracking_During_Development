"""Build notebooks/claude_nest2.ipynb — is a test dataset also a TRAIN dataset? CPU only.

`claude_nest` went looking for the metric's `N_est` in the test mount and found no `.geff`
there at all — only four `.zarr` image stores. On the way past, its listing showed something
much more interesting:

```
TEST   44b6_0113de3b.zarr   44b6_0b24845f.zarr   6bba_05b6850b.zarr   6bba_05db0fb1.zarr
TRAIN  ... 44b6_0b24845f.geff ...
```

**`44b6_0b24845f` appears in both.** And every test name carries a `44b6` or `6bba` prefix —
the same two embryo prefixes as training.

If that name collision is a real collision rather than a coincidence of hashes, three things
this project has treated as settled are wrong at once:

1. **`notes/49`'s "third pair of embryos".** The reasoning behind per-embryo grading, and
   behind distrusting every train-side screen, rests on the test set being unseen animals.
   The prefixes say otherwise.
2. **`N_est` is unavailable.** `claude_nest` closed the node-budget direction because the
   test mount has no GEFF. If the same dataset's GEFF sits in `train/`, its
   `estimated_number_of_nodes` is readable and the metric's multiplier becomes computable
   for at least part of the test set.
3. **Ground truth for a test dataset is readable**, which would make a genuine local
   validator possible for the first time — `notes/24`'s unknowable `split_0` membership
   stops mattering if we can score directly against a dataset the leaderboard also scores.

That is a lot to hang on one line of a directory listing, so this notebook checks it
properly rather than assuming it.

**Pre-registered predictions.**

1. At least one of the four test stems has a `.geff` of the same name under `train/`.
2. Where it does, the `.zarr` under `test/` and the one under `train/` have the **same time
   extent and shape** — i.e. it is the same movie, not a different crop that happens to
   share a hash.
3. `estimated_number_of_nodes` is much larger than the annotated node count in the same
   GEFF (the annotations are sparse; `MEMORY.md` records 20-150x more predictions than
   annotations), which is what makes `N_est` a population estimate rather than a label count.
4. With `N_est` known for the overlapping datasets, `claude_fork`'s node counts give a real
   ratio and a real adjustment factor.

Prediction 2 is the one that decides everything. A shared name with a different time extent
means the organisers re-cropped, and only prediction 1 survives.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claude_nest2.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Is a test dataset also a train dataset?

`claude_nest` found no `.geff` under `test/` — but its listing showed `44b6_0b24845f` under
**both** `test/` and `train/`, and all four test stems carry the same `44b6` / `6bba` embryo
prefixes as training.

If that holds, `notes/49`'s "the test set is a third pair of embryos" is wrong, the metric's
`N_est` is readable for part of the test set, and a real local validator becomes possible.

Predictions: (1) a name collision exists; (2) the colliding `.zarr`s have the same time
extent and shape, so it is the same movie; (3) `estimated_number_of_nodes` far exceeds the
annotated node count; (4) `claude_fork`'s output gives a real adjustment factor.
""")

code(r"""
import json
from pathlib import Path

ROOT = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TRAIN, TEST = ROOT / "train", ROOT / "test"

test_stems = sorted(p.name.split(".")[0] for p in TEST.glob("*.zarr"))
train_zarr = sorted(p.name.split(".")[0] for p in TRAIN.glob("*.zarr"))
train_geff = sorted(p.name.split(".")[0] for p in TRAIN.glob("*.geff"))

print(f"test  : {len(test_stems)} zarr   {test_stems}")
print(f"train : {len(train_zarr)} zarr, {len(train_geff)} geff")
print(f"train prefixes: {sorted({s.split('_')[0] for s in train_zarr})}")
print(f"test  prefixes: {sorted({s.split('_')[0] for s in test_stems})}")

overlap_geff = [s for s in test_stems if s in set(train_geff)]
overlap_zarr = [s for s in test_stems if s in set(train_zarr)]
print(f"\ntest stems with a train .geff : {overlap_geff}")
print(f"test stems with a train .zarr : {overlap_zarr}")
""")

code(r"""
# Same movie, or same hash on a different crop? Compare the zarr metadata directly.
def zmeta(p):
    out = {}
    j = p / "zarr.json"
    if j.exists():
        d = json.loads(j.read_text())
        out["node_type"] = d.get("node_type")
        out["attrs_keys"] = sorted((d.get("attributes") or {}).keys())
    # the pixel array lives under `0`; its own zarr.json carries the shape
    a = p / "0" / "zarr.json"
    if a.exists():
        d = json.loads(a.read_text())
        out["shape"] = d.get("shape")
        out["dtype"] = d.get("data_type")
        out["chunks"] = ((d.get("chunk_grid") or {}).get("configuration") or {}).get(
            "chunk_shape")
    return out

for s in test_stems:
    t = zmeta(TEST / f"{s}.zarr")
    r = zmeta(TRAIN / f"{s}.zarr") if (TRAIN / f"{s}.zarr").exists() else None
    same = (r is not None and t.get("shape") == r.get("shape"))
    print(f"{s:<18} test shape={t.get('shape')}  train shape="
          f"{r.get('shape') if r else '(absent)'}   same={same}")
""")

code(r"""
# estimated_number_of_nodes and the annotated node count, from the train GEFF.
def geff_meta(stem):
    p = TRAIN / f"{stem}.geff"
    if not p.exists():
        return None
    d = json.loads((p / "zarr.json").read_text())
    g = ((d.get("attributes") or {}).get("geff") or {})
    est = None
    stack = [d]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if "estimated" in str(k) and "node" in str(k):
                    est = v
                stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    axes = {a.get("name"): (a.get("min"), a.get("max")) for a in (g.get("axes") or [])}
    # node count: the ids array's own zarr.json shape
    n_ann = None
    for cand in ("nodes/ids/zarr.json", "nodes/props/t/values/zarr.json"):
        c = p / cand
        if c.exists():
            n_ann = json.loads(c.read_text()).get("shape")
            break
    return {"est": est, "t_range": axes.get("t"), "n_annotated": n_ann}

rows = {}
for s in test_stems:
    m = geff_meta(s)
    rows[s] = m
    print(f"{s:<18} {m}")
""")

code(r"""
FORK_NODES = {"6bba_05db0fb1": 69112, "44b6_0113de3b": 25425,
              "44b6_0b24845f": 18089, "6bba_05b6850b": 6033}

print("=" * 78)
print("PREDICTION GRADING")
print("=" * 78)

ok1 = bool(overlap_geff)
print(f"\n1. a test stem has a train .geff  ->  {'PASS' if ok1 else 'FAIL'}")
print(f"   {overlap_geff or 'none'}")

same_shape = []
for s in overlap_zarr:
    t = zmeta(TEST / f"{s}.zarr").get("shape")
    r = zmeta(TRAIN / f"{s}.zarr").get("shape")
    same_shape.append(t == r)
ok2 = bool(same_shape) and all(same_shape)
print(f"\n2. the colliding zarrs are the same movie  ->  "
      f"{'PASS' if ok2 else 'FAIL' if same_shape else 'NOT GRADED (no zarr overlap)'}")
if not ok2 and same_shape:
    print("   Same name, different pixels: the organisers re-cropped and the collision is")
    print("   a hash coincidence. notes/49 stands and nothing below is usable.")

ests = {s: m["est"] for s, m in rows.items() if m and isinstance(m.get("est"), (int, float))}
ok3 = False
if ests:
    print("\n3. N_est far exceeds the annotated node count")
    for s, e in ests.items():
        n = rows[s].get("n_annotated")
        n0 = n[0] if isinstance(n, list) and n else None
        ratio = (e / n0) if n0 else float("nan")
        print(f"   {s:<18} est {e:>10,.0f}   annotated {str(n0):>8}   {ratio:>7.1f}x")
        ok3 = ok3 or (n0 is not None and e > 3 * n0)
    print(f"   ->  {'PASS' if ok3 else 'FAIL'}")
else:
    print("\n3. NOT GRADED — no estimated_number_of_nodes read")

if ests:
    print("\n4. claude_fork's adjustment factor on the datasets we can now see")
    print(f"   {'dataset':<18}{'N_pred':>10}{'N_est':>12}{'ratio':>10}{'factor':>10}")
    tp = te = 0.0
    for s in sorted(ests):
        n, e = FORK_NODES[s], ests[s]
        tp += n; te += e
        r = (n - e) / e
        print(f"   {s:<18}{n:>10,}{e:>12,.0f}{r:>+10.3f}{1 - 0.1 * r:>10.4f}")
    R = (tp - te) / te
    print(f"   {'subtotal':<18}{tp:>10,.0f}{te:>12,.0f}{R:>+10.3f}{1 - 0.1 * R:>10.4f}")
    print(f"\n   At edge_J ~ 0.92 that factor is worth {0.92 * (1 - 0.1 * R) - 0.92:+.4f}.")
    if R > 0:
        print("   OVER budget: pruning toward N_est pays twice -- a larger factor AND")
        print("   fewer false edges. This is the first unclaimed term we have found.")
    else:
        print("   UNDER budget already: the bonus is being collected, and further pruning")
        print("   trades edge_J against a factor above 1. Prices the direction near zero.")
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
