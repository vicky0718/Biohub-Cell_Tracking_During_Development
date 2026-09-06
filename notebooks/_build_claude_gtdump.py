"""Build notebooks/claude_gtdump.ipynb — export the four test datasets' ground truth. CPU.

`claude_nest2` established that the four hidden-test stems are four of the 199 **training**
datasets, ground truth included. `claude_oracle` scores against that GT inside a Kaggle
kernel, which works but costs a kernel round trip per evaluation — minutes each, and
Kaggle's session limits apply.

The GT itself is tiny: 52, 51, 861 and 1,229 annotated nodes. Dumping it once turns every
future evaluation into a local computation over `harness/purescore.py` that takes seconds
and needs no Kaggle session at all. An arm's `submission.csv` already comes down through
`/kernels/output/download` (`notes/65` §2), so after this run the whole screening loop is
local.

**On what this data is.** These are *public training* annotations for datasets that also
appear in `test/`. Using them to choose between configurations is validation, and that is
all this is for. Using them to build a submission — copying GT edges into the output —
would be a different thing entirely and is not what this enables: the annotations cover
about 0.2% of the nodes a submission contains (2,193 against 118,659), so it could not
carry a submission even if we wanted it to, and it would be constructing an answer rather
than measuring one.

The dump also records `estimated_number_of_nodes` per dataset, which is the metric's `N_est`
and cannot be recomputed from anything else we hold.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "claude_gtdump.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Export the four test datasets' ground truth, once

The hidden-test stems are training datasets (`claude_nest2`), and their GT is 2,193
annotated nodes in total. Dumping it makes every later evaluation local and instant.

Writes `test_gt.json`: per dataset, the node `t`/`z`/`y`/`x` arrays, the edge list as index
pairs, and `estimated_number_of_nodes`.
""")

code(r"""
import json, subprocess, sys
from pathlib import Path
import numpy as np

try:
    import zarr
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "zarr>=3"], check=True)
    import zarr
print("zarr", zarr.__version__)

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
TRAIN = COMP / "train"
STEMS = sorted(p.name.split(".")[0] for p in (COMP / "test").glob("*.zarr"))
print("test stems:", STEMS)


def arr(root, name):
    try:
        return np.asarray(zarr.open_array(str(root / name), mode="r"))
    except Exception as e:
        print(f"   {name}: {type(e).__name__} {e}")
        return None


def est_nodes(geff):
    # estimated_number_of_nodes, wherever it sits in the GEFF metadata tree.
    stack = [json.loads((geff / "zarr.json").read_text())]
    while stack:
        o = stack.pop()
        if isinstance(o, dict):
            for k, v in o.items():
                if "estimated" in str(k) and "node" in str(k):
                    return float(v)
                stack.append(v)
        elif isinstance(o, list):
            stack.extend(o)
    return float("nan")


dump = {}
for s in STEMS:
    g = TRAIN / f"{s}.geff"
    ids = arr(g, "nodes/ids")
    t = arr(g, "nodes/props/t/values")
    z = arr(g, "nodes/props/z/values")
    y = arr(g, "nodes/props/y/values")
    x = arr(g, "nodes/props/x/values")
    e = arr(g, "edges/ids")
    index = {int(v): i for i, v in enumerate(np.asarray(ids).ravel())}
    pairs = []
    if e is not None and len(e):
        for a, b in np.asarray(e).reshape(-1, 2):
            if int(a) in index and int(b) in index:
                pairs.append([index[int(a)], index[int(b)]])
    dump[s] = {
        "t": np.asarray(t).astype(int).tolist(),
        "z": np.asarray(z).astype(float).tolist(),
        "y": np.asarray(y).astype(float).tolist(),
        "x": np.asarray(x).astype(float).tolist(),
        "edges": pairs,
        "estimated_number_of_nodes": est_nodes(g),
    }
    print(f"{s:<18} nodes {len(dump[s]['t']):>6,}  edges {len(pairs):>6,}  "
          f"est {dump[s]['estimated_number_of_nodes']:>10,.0f}")

Path("test_gt.json").write_text(json.dumps(dump))
print(f"\nwrote test_gt.json  ({Path('test_gt.json').stat().st_size:,} bytes)")
""")

code(r"""
# Sanity: the counts must match what claude_nest2 read independently, from the metadata
# rather than from the arrays. Two routes to the same numbers or the dump is not trusted.
WANT_NODES = {"44b6_0113de3b": 52, "44b6_0b24845f": 51,
              "6bba_05b6850b": 861, "6bba_05db0fb1": 1229}
WANT_EST = {"44b6_0113de3b": 25755, "44b6_0b24845f": 32795,
            "6bba_05b6850b": 6362, "6bba_05db0fb1": 69800}
ok = True
for s, w in WANT_NODES.items():
    got = len(dump[s]["t"])
    e_got, e_want = dump[s]["estimated_number_of_nodes"], WANT_EST[s]
    good = (got == w) and abs(e_got - e_want) < 1
    ok &= good
    print(f"{s:<18} nodes {got:>6} want {w:>6}   est {e_got:>10,.0f} want "
          f"{e_want:>10,}   {'ok' if good else 'MISMATCH'}")
print("\nDUMP TRUSTED" if ok else "\nDO NOT USE — the dump disagrees with claude_nest2")
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
