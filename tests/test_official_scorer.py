"""Regression test: a FORKING prediction must score through the official scorer.

    CELLMOT_REPO=/path/to/kaggle-cell-tracking-competition python tests/test_official_scorer.py

Skips cleanly when the official scorer or tracksdata is unavailable -- it cannot run on
the Kaggle submission image, which is the whole reason `purescore` exists.

Every arm in this project so far has been fork-free, so `Harness.score_graph`'s
`use_official` branch had never executed. The first forking prediction -- the public
pack's model, which predicts divisions -- hit KeyError('match_node_id') because the branch
converted to tracksdata twice: `evaluate` writes its matching onto the graphs it is given,
and `node_recall` reads it back.
"""
import shutil, sys, tempfile
from pathlib import Path
import numpy as np, polars as pl, tracksdata as td, zarr

REPO = Path("/workspace/biohub-cell_tracking_during_development")
sys.path.insert(0, str(REPO))
from harness import Harness
from harness.tracks import Tracks
from harness import harness as _H

if _H._official() is None:
    print("SKIP: official scorer not importable "
          "(set CELLMOT_REPO and install tracksdata). This path cannot be exercised "
          "on the Kaggle submission image, which is why purescore exists.")
    raise SystemExit(0)

SCALE = (1.625, 0.40625, 0.40625)
tmp = Path(tempfile.mkdtemp(prefix="forktest_"))
T, N = 4, 6
pos = np.column_stack([np.full(N, 8.0), np.linspace(20, 80, N), np.linspace(20, 80, N)])

# ---- ground truth: N tracks, one real division at t=2 on track 0
g = td.graph.InMemoryGraph()
for k in ("z", "y", "x"):
    g.add_node_attr_key(k, pl.Float64, 0.0)
prev, ids = [None] * N, {}
for t in range(T):
    for i in range(N):
        nid = g.add_node({"t": t, "z": float(pos[i][0]),
                          "y": float(pos[i][1]), "x": float(pos[i][2])})
        ids[(t, i)] = nid
        if prev[i] is not None:
            g.add_edge(prev[i], nid, {})
        prev[i] = nid
# a genuine fork in the GT: node 0 at t=1 also parents a new node at t=2
extra = g.add_node({"t": 2, "z": float(pos[0][0]), "y": float(pos[0][1] + 3),
                    "x": float(pos[0][2] + 3)})
g.add_edge(ids[(1, 0)], extra, {})
gp = tmp / "emb_0.geff"
g.to_geff(str(gp))
zg = zarr.open_group(str(gp), mode="a")
ex = dict(zg.attrs.get("geff", {})); ex.setdefault("extra", {})["estimated_number_of_nodes"] = float(N * T)
zg.attrs["geff"] = ex

mv = np.zeros((T, 16, 96, 96), np.float32)
z = zarr.open_group(str(tmp / "emb_0.zarr"), mode="w")
z.create_array("0", shape=mv.shape, dtype=mv.dtype, chunks=(1, 16, 96, 96))
z["0"][:] = mv
z.attrs["multiscales"] = [{"datasets": [{"path": "0", "coordinateTransformations":
                          [{"type": "scale", "scale": [1.0, *SCALE]}]}]}]
z.attrs["image_statistics"] = {"quantiles": {"0.001": 0.0, "0.999": 1.0}}

# ---- prediction WITH a fork
coords, edges = [], []
idx = {}
for t in range(T):
    for i in range(N):
        idx[(t, i)] = len(coords)
        coords.append([t, pos[i][0], pos[i][1], pos[i][2]])
for t in range(T - 1):
    for i in range(N):
        edges.append((idx[(t, i)], idx[(t + 1, i)]))
fork_child = len(coords)
coords.append([2, pos[0][0], pos[0][1] + 3, pos[0][2] + 3])
edges.append((idx[(1, 0)], fork_child))          # node (1,0) now has out-degree 2
pred = Tracks(np.array([c[0] for c in coords], float),
              np.array([c[1:] for c in coords], float),
              np.array(edges, int))

fails = []
def check(label, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {label}" + (f" — {detail}" if detail else ""))
    if not ok: fails.append(label)

check("the prediction really forks", pred.has_forks(), f"{pred.n_divisions} division(s)")

h = Harness(data_dir=tmp, cache_dir=None)
try:
    row = h.score_graph("emb_0", pred)
    ok, err = True, ""
except Exception as e:
    row, ok, err = None, False, f"{type(e).__name__}: {e}"
check("a forking prediction scores without raising", ok, err)
if row:
    print("   row:", {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in sorted(row.items())})
    num = [k for k, v in row.items() if isinstance(v, (int, float)) and v == v]
    check("it returned real numeric metrics", len(num) >= 4, f"{len(num)} numeric fields")
    nr = row.get("node_recall")
    check("node recall was computed, not silently zeroed",
          nr is not None and nr > 0, f"node_recall={nr}")

shutil.rmtree(tmp, ignore_errors=True)
print("\n" + "=" * 60)
if fails:
    print(f"{len(fails)} FAILURE(S): {fails}"); sys.exit(1)
print("forking predictions score correctly through the official scorer")
