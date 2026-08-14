"""Build biohub/notebooks/01_recon.ipynb from cell sources defined here."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_recon.ipynb"

CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Biohub Cell Tracking — Recon

Answers the open questions in `notes/02-metric-findings.md`, **without loading a single
image**: everything here reads GEFF ground truth and zarr metadata only, so it runs on CPU
in a few minutes.

What it produces:

1. Dataset inventory (train/test, frames, image shapes) and the official fold splits.
2. Annotation density — how much of the embryo is actually labelled, and the
   `estimated_number_of_nodes` node budget the scorer measures us against.
3. Motion statistics — inter-frame displacement in µm, which sets the linking radius.
4. Confusability — nearest-neighbour spacing vs the 7 µm match cutoff.
5. Division counts — is the `0.1 · division_jaccard` term worth any effort.
6. **The linking-only ceiling**: feed the GT nodes back in as perfect detections, link them
   by nearest neighbour, and score with the official scorer. This splits the problem into
   "how much of the score is detection" vs "how much is linking".

Everything lands in `recon_summary.json` at the end.
""")

code(r"""
# --- deps -------------------------------------------------------------------
# tracksdata reads .geff and provides the official matching code.
# Requires internet ON in the notebook settings (Settings -> Internet).
!pip install -q tracksdata 2>&1 | tail -2

import subprocess, sys, os
from pathlib import Path

CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if not CELLMOT.exists():
    subprocess.run(
        ["git", "clone", "--depth", "1",
         "https://github.com/royerlab/kaggle-cell-tracking-competition.git", str(CELLMOT)],
        check=False,
    )
if not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    raise SystemExit(
        "Could not fetch the official baseline repo.\n"
        "Turn Internet ON in notebook settings, or attach the repo as a Kaggle dataset "
        "and point CELLMOT at it. We use it ONLY for the official scorer, so that our "
        "local numbers are the same numbers the leaderboard computes."
    )
sys.path.insert(0, str(CELLMOT / "src"))
print("cellmot repo:", CELLMOT)
""")

code(r"""
import json, warnings
from collections import Counter

import numpy as np
import polars as pl
import zarr
import tracksdata as td
from geff import GeffMetadata

warnings.filterwarnings("ignore")
K = td.DEFAULT_ATTR_KEYS

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():  # some mounts drop the "competitions" level
    alt = Path("/kaggle/input/biohub-cell-tracking-during-development")
    COMP = alt if alt.exists() else COMP
TRAIN, TEST = COMP / "train", COMP / "test"
print("competition mount:", COMP, "| exists:", COMP.exists())
print("train:", TRAIN.exists(), "| test:", TEST.exists())
if COMP.exists():
    print("top level:", sorted(p.name for p in COMP.iterdir())[:20])

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)  # z, y, x microns/px
MATCH_UM = 7.0                             # scorer's node-match cutoff


def load_geff(path):
    r = td.graph.IndexedRXGraph.from_geff(str(path))
    return r[0] if isinstance(r, tuple) else r


def zarr_info(zpath):
    '''(T, Z, Y, X) shape and (z, y, x) scale from zarr metadata only.'''
    g = zarr.open_group(str(zpath), mode="r")
    attrs = dict(g.attrs)
    shape = tuple(g["0"].shape)
    scale = DEFAULT_SCALE
    if "multiscales" in attrs:
        tr = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr.get("type") == "scale":
            scale = tuple(tr["scale"][-3:])
    return shape, scale, attrs


def estimated_nodes(geff_path):
    try:
        meta = GeffMetadata.read(str(geff_path))
        v = (meta.extra or {}).get("estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception as e:
        print("  (no geff metadata:", type(e).__name__, e, ")")
        return float("nan")
""")

md("## 1. Inventory")

code(r"""
train_names = sorted(p.stem for p in TRAIN.glob("*.zarr")) if TRAIN.exists() else []
test_names  = sorted(p.stem for p in TEST.glob("*.zarr"))  if TEST.exists()  else []
print(f"{len(train_names)} train datasets, {len(test_names)} test datasets\n")
print("train:", train_names)
print("test: ", test_names)

# Official 5-fold splits ship with the data; use them so our CV matches the organisers'.
splits = None
for cand in (TRAIN / "dataset_splits.json", COMP / "dataset_splits.json"):
    if cand.exists():
        splits = json.loads(cand.read_text())
        print(f"\nfound official splits at {cand}: {len(splits)} folds")
        for i, f in enumerate(splits):
            print(f"  fold {i}: {len(f.get('train', []))} train / {len(f.get('test', []))} test")
        break
if splits is None:
    print("\nNo dataset_splits.json found - we build our own deterministic folds later.")
""")

code(r"""
rows = []
for name in train_names:
    shape, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    na = gt.node_attrs()
    n_est = estimated_nodes(TRAIN / f"{name}.geff")
    out_deg = np.asarray(gt.out_degree(gt.node_ids()))
    rows.append(dict(
        name=name, T=shape[0], Z=shape[1], Y=shape[2], X=shape[3],
        scale_z=scale[0], scale_y=scale[1], scale_x=scale[2],
        gt_nodes=gt.num_nodes(), gt_edges=gt.num_edges(),
        divisions=int((out_deg == 2).sum()),
        t_min=int(na[K.T].min()), t_max=int(na[K.T].max()),
        est_total_nodes=n_est,
    ))
    print(f"{name:<28} T={shape[0]:>4} shape={shape[1:]}  gt_nodes={gt.num_nodes():>7} "
          f"gt_edges={gt.num_edges():>7} div={int((out_deg==2).sum()):>5} est_total={n_est:,.0f}")

inv = pl.DataFrame(rows)
inv
""")

md("## 2. Annotation density and the node budget\n\nHow much of the embryo is labelled, and what `N_total` the `×(1 − 0.1·ratio)` term measures us against.")

code(r"""
d = inv.with_columns(
    (pl.col("gt_nodes") / pl.col("est_total_nodes")).alias("annotated_frac"),
    (pl.col("gt_nodes") / pl.col("T")).alias("gt_nodes_per_frame"),
    (pl.col("est_total_nodes") / pl.col("T")).alias("est_cells_per_frame"),
)
print(d.select("name", "T", "gt_nodes", "est_total_nodes",
               "annotated_frac", "gt_nodes_per_frame", "est_cells_per_frame"))

af = d["annotated_frac"].drop_nans().drop_nulls()
if len(af):
    print(f"\nAnnotated fraction: min={af.min():.4f} median={af.median():.4f} max={af.max():.4f}")
    print(f"=> roughly 1 cell in {1/af.median():.0f} is annotated (median dataset)")
print("\nThis is THE number behind 'false positives are free': the scorer only ever sees "
      "edges touching that annotated minority.")
print("\nNode budget: to sit at ratio 0 we must emit ~est_total_nodes detections per dataset:")
print(d.select("name", "est_total_nodes", "est_cells_per_frame"))
""")

md("## 3. Motion — how far does a cell move between frames?\n\nSets the linking search radius, and tells us whether the true match is even the nearest neighbour.")

code(r"""
def edge_displacements(gt, scale):
    '''Physical (um) displacement for every GT edge, plus dt sanity check.'''
    na = gt.node_attrs().select(K.NODE_ID, K.T, "z", "y", "x")
    ea = gt.edge_attrs().select(K.EDGE_SOURCE, K.EDGE_TARGET)
    if ea.height == 0:
        return np.empty(0), np.empty(0)
    j = (ea.join(na.rename({K.NODE_ID: K.EDGE_SOURCE, K.T: "t0", "z": "z0", "y": "y0", "x": "x0"}),
                 on=K.EDGE_SOURCE, how="left")
           .join(na.rename({K.NODE_ID: K.EDGE_TARGET, K.T: "t1", "z": "z1", "y": "y1", "x": "x1"}),
                 on=K.EDGE_TARGET, how="left"))
    dz = (j["z1"] - j["z0"]).to_numpy() * scale[0]
    dy = (j["y1"] - j["y0"]).to_numpy() * scale[1]
    dx = (j["x1"] - j["x0"]).to_numpy() * scale[2]
    dt = (j["t1"] - j["t0"]).to_numpy()
    return np.sqrt(dz**2 + dy**2 + dx**2), dt


all_disp = []
for name in train_names:
    _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    disp, dt = edge_displacements(gt, scale)
    if len(disp) == 0:
        continue
    all_disp.append(disp)
    bad_dt = int((dt != 1).sum())
    print(f"{name:<28} n={len(disp):>7}  median={np.median(disp):6.2f}um  "
          f"p90={np.percentile(disp,90):6.2f}  p99={np.percentile(disp,99):6.2f}  "
          f"max={disp.max():7.2f}" + (f"   !! {bad_dt} edges with dt!=1" if bad_dt else ""))

disp = np.concatenate(all_disp) if all_disp else np.empty(0)
if len(disp):
    print(f"\nPOOLED inter-frame displacement (um), n={len(disp):,}")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<5} = {np.percentile(disp, q):7.2f}")
    print(f"  max    = {disp.max():7.2f}")
    print(f"\nFraction of true links moving further than the {MATCH_UM} um match radius: "
          f"{(disp > MATCH_UM).mean():.3%}")
""")

md("## 4. Confusability — cell spacing vs the 7 µm match radius")

code(r"""
from scipy.spatial import cKDTree

nn_all, nn_ratio_all = [], []
for name in train_names:
    _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    na = gt.node_attrs().select(K.T, "z", "y", "x")
    per_ds = []
    for t, grp in na.group_by(K.T):
        pts = np.stack([grp["z"].to_numpy() * scale[0],
                        grp["y"].to_numpy() * scale[1],
                        grp["x"].to_numpy() * scale[2]], axis=1)
        if len(pts) < 2:
            continue
        dd, _ = cKDTree(pts).query(pts, k=2)
        per_ds.append(dd[:, 1])
    if per_ds:
        v = np.concatenate(per_ds)
        nn_all.append(v)
        print(f"{name:<28} NN spacing (annotated only): median={np.median(v):6.2f}um  "
              f"p10={np.percentile(v,10):6.2f}  frac<{MATCH_UM}um={np.mean(v<MATCH_UM):.2%}")

nn = np.concatenate(nn_all) if nn_all else np.empty(0)
if len(nn):
    print(f"\nPOOLED NN spacing between ANNOTATED cells: median={np.median(nn):.2f}um, "
          f"{np.mean(nn < MATCH_UM):.2%} closer than the {MATCH_UM}um match radius")
    print("NOTE: annotated cells are a sparse subset, so TRUE cell spacing is much tighter "
          "than this - divide by roughly the cube root of the annotated fraction for a feel.")
""")

md("## 5. Is the nearest neighbour the right link?\n\nThis is the `p` in the `link if p > J/(1+J)` rule from the metric findings.")

code(r"""
nn_correct, nn_total = 0, 0
rank_hist = Counter()

for name in train_names:
    _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    na = gt.node_attrs().select(K.NODE_ID, K.T, "z", "y", "x")
    ea = gt.edge_attrs().select(K.EDGE_SOURCE, K.EDGE_TARGET)
    if ea.height == 0:
        continue
    by_t = {int(t[0]): g for t, g in na.group_by(K.T)}
    true_tgt = dict(zip(ea[K.EDGE_SOURCE].to_list(), ea[K.EDGE_TARGET].to_list()))
    id_to_t = dict(zip(na[K.NODE_ID].to_list(), na[K.T].to_list()))

    ds_ok = ds_tot = 0
    for t, grp in by_t.items():
        nxt = by_t.get(t + 1)
        if nxt is None:
            continue
        src_ids = grp[K.NODE_ID].to_numpy()
        tgt_ids = nxt[K.NODE_ID].to_numpy()
        P = np.stack([grp["z"].to_numpy() * scale[0], grp["y"].to_numpy() * scale[1],
                      grp["x"].to_numpy() * scale[2]], axis=1)
        Q = np.stack([nxt["z"].to_numpy() * scale[0], nxt["y"].to_numpy() * scale[1],
                      nxt["x"].to_numpy() * scale[2]], axis=1)
        tree = cKDTree(Q)
        kk = min(5, len(Q))
        _, idx = tree.query(P, k=kk)
        idx = np.asarray(idx).reshape(len(P), -1)
        for row, sid in enumerate(src_ids):
            tid = true_tgt.get(int(sid))
            if tid is None or id_to_t.get(tid) != t + 1:
                continue
            ds_tot += 1
            cand = tgt_ids[idx[row]]
            hit = np.where(cand == tid)[0]
            rank_hist[int(hit[0]) + 1 if len(hit) else ">5"] += 1
            if len(hit) and hit[0] == 0:
                ds_ok += 1
    if ds_tot:
        print(f"{name:<28} nearest-neighbour is the true link: {ds_ok/ds_tot:.2%}  (n={ds_tot:,})")
    nn_correct += ds_ok
    nn_total += ds_tot

if nn_total:
    p_nn = nn_correct / nn_total
    print(f"\nPOOLED: the nearest annotated cell in t+1 is the true successor {p_nn:.2%} "
          f"of the time (n={nn_total:,})")
    print("rank of the true target among nearest neighbours:", dict(sorted(
        rank_hist.items(), key=lambda kv: (isinstance(kv[0], str), kv[0]))))
    print("\nCAVEAT: computed among ANNOTATED cells only. A real detector also proposes the "
          "unannotated majority, so the true competitor set is far denser and p will be lower.")
""")

md("""## 5b. Is the sparse annotation actually unbiased?

The ground truth comes from a **second, sparse fluorescence channel** we are not given —
Ultrack's dual-channel trick (the baseline's local data path is literally
`./data/dense_channel`). The labeling is described as *random*, but random at the
*genetic* level is not the same as random with respect to what we have to predict:

1. **Clonal clustering** — a mosaic label is inherited by both daughters, so annotated
   cells should arrive in clumps, and divisions should be **over-represented** relative
   to a uniform sample of cells.
2. **Depth bias** — a cell only becomes ground truth if it was visible in the *sparse*
   channel too, which suffers the same depth attenuation. Deep cells may be missing from
   GT, which would make our validation score optimistic exactly where the imaging is worst.

Both are testable here, and nothing outside the training set can answer them.
""")

code(r"""
# --- 1. depth bias: where in Z do annotations live vs. where is there imageable volume? ---
print("Annotated-node density vs Z (are deep cells under-represented?)\n")
for name in train_names:
    shape, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    z = gt.node_attrs()["z"].to_numpy().astype(float)
    Zmax = shape[1]
    # decile occupancy of the annotated set across the imaged Z range
    hist, _ = np.histogram(z, bins=10, range=(0, Zmax))
    frac = hist / max(1, hist.sum())
    bars = " ".join(f"{f:.2f}" for f in frac)
    print(f"{name:<26} Z=0..{Zmax:<4} deciles: {bars}")
    print(f"{'':<26} median z = {np.median(z):.1f} / {Zmax}  "
          f"(uniform would be {Zmax/2:.1f})")
print("\nA flat profile => no depth bias. Falling deciles => deep cells under-annotated, "
      "and our CV will overstate how well we do at depth.")
""")

code(r"""
# --- 2. clonal clustering: are annotated cells clumped vs a uniform null? ---
# Compare observed NN distance among annotated cells against random points drawn in the
# same bounding box at the same density. Clumping => observed noticeably smaller.
rng = np.random.default_rng(0)
print("Observed vs uniform-null nearest-neighbour spacing among annotated cells\n")
for name in train_names:
    _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    na = gt.node_attrs().select(K.T, "z", "y", "x")
    obs, null = [], []
    for t, grp in na.group_by(K.T):
        P = np.stack([grp["z"].to_numpy() * scale[0],
                      grp["y"].to_numpy() * scale[1],
                      grp["x"].to_numpy() * scale[2]], axis=1)
        if len(P) < 10:
            continue
        obs.append(cKDTree(P).query(P, k=2)[0][:, 1])
        lo, hi = P.min(0), P.max(0)
        Q = rng.uniform(lo, hi, size=P.shape)
        null.append(cKDTree(Q).query(Q, k=2)[0][:, 1])
    if not obs:
        continue
    o, n = np.concatenate(obs), np.concatenate(null)
    ratio = np.median(o) / max(1e-9, np.median(n))
    verdict = "CLUMPED" if ratio < 0.85 else ("dispersed" if ratio > 1.15 else "~uniform")
    print(f"{name:<26} observed={np.median(o):6.2f}um  null={np.median(n):6.2f}um  "
          f"ratio={ratio:.2f}  -> {verdict}")
print("\nClumping is expected if labels are clonal. It matters because our NN-linking "
      "statistics were computed among annotated cells only - if they cluster, the real "
      "detector's neighbourhood is denser AND differently structured than that estimate.")
""")

code(r"""
# --- 3. division enrichment: do annotated cells divide more than a uniform sample would? ---
# Daughters inherit a mosaic label, so both children of an annotated dividing cell are
# themselves annotated - divisions should be over-represented.
print("Division rate within the annotated set\n")
tot_div_nodes = tot_nodes_with_out = 0
for name in train_names:
    gt = load_geff(TRAIN / f"{name}.geff")
    ids = gt.node_ids()
    out_deg = np.asarray(gt.out_degree(ids))
    n_div = int((out_deg == 2).sum())
    n_out = int((out_deg > 0).sum())
    tot_div_nodes += n_div
    tot_nodes_with_out += n_out
    print(f"{name:<26} {n_div:>6} dividing / {n_out:>7} nodes with a successor "
          f"= {100*n_div/max(1,n_out):.3f}%")
print(f"\nPOOLED: {100*tot_div_nodes/max(1,tot_nodes_with_out):.3f}% of annotated nodes divide.")
print("Sanity anchor: a cell cycle of N frames implies roughly 1/N of nodes divide per frame. "
      "Invert the observed rate to estimate the cell-cycle length in frames, then compare "
      "against the frame interval implied below.")
""")

md("""## 5c. Frame interval and the Z/XY error budget

Two things the outside literature could not tell us, both decisive:

- **Frame interval.** Everything about linking scales with it. Zebrafish cells move at
  roughly 0.8–1 µm/min during somitogenesis (up to ~3.3 µm/min at peak epiboly), so the
  median per-edge displacement below inverts directly to an interval.
- **The Z error budget.** The 7 µm match cutoff is applied as an *isotropic physical*
  distance, but Z voxels are 4× coarser than XY (1.625 vs 0.40625 µm — exactly 4:1) and a
  nucleus spans only ~4–6 Z-slices. A 2-slice Z error is 3.25 µm, roughly **half the entire
  match budget**; the same 3.25 µm in XY is 8 pixels. Z centroid accuracy is worth far more
  than XY accuracy, and it is the axis most likely to carry systematic bias.
""")

code(r"""
# Decompose displacement into Z and XY components, in microns.
all_dz, all_dxy = [], []
for name in train_names:
    _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    na = gt.node_attrs().select(K.NODE_ID, K.T, "z", "y", "x")
    ea = gt.edge_attrs().select(K.EDGE_SOURCE, K.EDGE_TARGET)
    if ea.height == 0:
        continue
    j = (ea.join(na.rename({K.NODE_ID: K.EDGE_SOURCE, K.T: "t0", "z": "z0", "y": "y0", "x": "x0"}),
                 on=K.EDGE_SOURCE, how="left")
           .join(na.rename({K.NODE_ID: K.EDGE_TARGET, K.T: "t1", "z": "z1", "y": "y1", "x": "x1"}),
                 on=K.EDGE_TARGET, how="left"))
    dz = np.abs((j["z1"] - j["z0"]).to_numpy()) * scale[0]
    dy = (j["y1"] - j["y0"]).to_numpy() * scale[1]
    dx = (j["x1"] - j["x0"]).to_numpy() * scale[2]
    all_dz.append(dz)
    all_dxy.append(np.sqrt(dy**2 + dx**2))

if all_dz:
    dz = np.concatenate(all_dz); dxy = np.concatenate(all_dxy)
    print(f"per-edge |dZ|  median={np.median(dz):.3f}um  p90={np.percentile(dz,90):.3f}  "
          f"p99={np.percentile(dz,99):.3f}")
    print(f"per-edge |dXY| median={np.median(dxy):.3f}um  p90={np.percentile(dxy,90):.3f}  "
          f"p99={np.percentile(dxy,99):.3f}")
    print(f"\n|dZ| in VOXELS: median={np.median(dz)/1.625:.2f}  "
          f"p99={np.percentile(dz,99)/1.625:.2f}  (Z voxel = 1.625um)")
    print("If median |dZ| is well under one voxel, Z motion is being quantised by the grid "
          "and Z centroids are the precision bottleneck, exactly as the anisotropy predicts.")

    d3 = np.sqrt(dz**2 + dxy**2)
    med = np.median(d3)
    print(f"\nmedian 3D displacement = {med:.3f} um/frame")
    for speed, label in ((0.83, "somitogenesis PSM ~0.83 um/min"),
                         (1.00, "epiboly azimuthal ~1.0 um/min"),
                         (3.30, "peak epiboly ~3.3 um/min")):
        print(f"  if cells move at {label:<34} -> frame interval ~= {60*med/speed:6.1f} s")
    print("\nCross-check against the division rate above: cell-cycle length in frames x "
          "interval should land in a plausible range (zebrafish cycles are ~20-40 min at "
          "these stages). If the two disagree wildly, one of the assumptions is wrong.")
""")

code(r"""
# --- 4. do tracks enter/leave through the volume border? ---
# Dataset names like 2024_03_22_dorado_0002_0198_0184_0605 suggest these are CROPS of a
# larger acquisition, so cells cross the boundary. Ultrack has an explicit image_border_size
# for exactly this. If most track starts/ends sit at a face, appearance/disappearance is a
# boundary artifact rather than a biological event - and should not be modelled as one.
print("Track endpoints near the volume boundary\n")
MARGIN_UM = 10.0
for name in train_names:
    shape, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    gt = load_geff(TRAIN / f"{name}.geff")
    ids = gt.node_ids()
    indeg = np.asarray(gt.in_degree(ids)); outdeg = np.asarray(gt.out_degree(ids))
    na = gt.node_attrs().select(K.NODE_ID, K.T, "z", "y", "x")
    order = {int(n): i for i, n in enumerate(na[K.NODE_ID].to_list())}
    pos = np.stack([na["z"].to_numpy(), na["y"].to_numpy(), na["x"].to_numpy()], axis=1)
    tt = na[K.T].to_numpy()
    dims_um = np.array(shape[1:]) * np.array(scale)
    idx = np.array([order[int(n)] for n in ids])
    p_um = pos[idx] * np.array(scale)
    near = ((p_um < MARGIN_UM) | (p_um > dims_um - MARGIN_UM)).any(axis=1)
    interior_t = (tt[idx] > tt.min()) & (tt[idx] < tt.max())
    starts = (indeg == 0) & interior_t
    ends = (outdeg == 0) & interior_t
    n_s, n_e = int(starts.sum()), int(ends.sum())
    print(f"{name:<26} mid-movie track starts={n_s:>5} ({100*near[starts].mean() if n_s else 0:.0f}% at border)  "
          f"ends={n_e:>5} ({100*near[ends].mean() if n_e else 0:.0f}% at border)")
print(f"\nHigh border fractions => appearances/disappearances are mostly the crop edge. "
      f"Low fractions => they are annotation dropout, which is a different problem.")
""")

md("## 6. Divisions — is the 0.1 term worth anything?")

code(r"""
tot_div = int(inv["divisions"].sum())
tot_edges = int(inv["gt_edges"].sum())
print(f"divisions across train: {tot_div:,}")
print(f"GT edges across train:  {tot_edges:,}")
print(f"divisions per 1000 GT edges: {1000*tot_div/max(1,tot_edges):.2f}")
print(f"\nThe division term is worth at most 0.1 of the final score. Every division we chase "
      f"risks edge FPs worth 1.0-weighted points (see notes/02-metric-findings.md section 5).")
""")

md("""## 7. The linking-only ceiling

Feed the **GT nodes back in as perfect detections** and link them by nearest neighbour.
Scored with the official scorer, so it is directly comparable to a leaderboard number.

- If this lands near 1.0, tracking is easy and **detection is the whole contest**.
- If it lands low, linking is genuinely hard and deserves the modelling effort.

Optimistic in one direction (no unannotated distractors) and pessimistic in another
(nearest-neighbour is the dumbest possible linker) — read it as a decomposition, not a target.
""")

code(r"""
from scipy.optimize import linear_sum_assignment
from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise


def build_graph(coords):
    '''coords: (N,4) array of t,z,y,x -> InMemoryGraph, returns (graph, node_ids).'''
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                            for t, z, y, x in coords])
    return g, ids


def link_nearest(coords, scale, max_um=None, mode="hungarian"):
    '''Link consecutive frames. mode='hungarian' = optimal 1-1; 'greedy' = mutual NN.'''
    order = np.argsort(coords[:, 0], kind="stable")
    idx_by_t = {}
    for i in order:
        idx_by_t.setdefault(int(coords[i, 0]), []).append(i)
    phys = coords[:, 1:] * np.asarray(scale)[None, :]
    edges = []
    for t in sorted(idx_by_t):
        a, b = idx_by_t.get(t), idx_by_t.get(t + 1)
        if not a or not b:
            continue
        A, B = phys[a], phys[b]
        D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
        if mode == "hungarian":
            ri, ci = linear_sum_assignment(D)
            pairs = zip(ri, ci)
        else:
            pairs = [(i, int(np.argmin(D[i]))) for i in range(len(a))]
        for i, j in pairs:
            if max_um is None or D[i, j] <= max_um:
                edges.append((a[i], b[j]))
    return edges


results = {}
for mode in ("hungarian", "greedy"):
    rows = []
    for name in train_names:
        _, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
        gt = load_geff(TRAIN / f"{name}.geff")
        na = gt.node_attrs().select(K.T, "z", "y", "x")
        coords = np.stack([na[K.T].to_numpy(), na["z"].to_numpy(),
                           na["y"].to_numpy(), na["x"].to_numpy()], axis=1).astype(float)
        pred, ids = build_graph(coords)
        e = link_nearest(coords, scale, max_um=None, mode=mode)
        if e:
            pred.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in e])
        er = evaluate(pred, load_geff(TRAIN / f"{name}.geff"), scale=scale, max_distance=MATCH_UM)
        rec = node_recall(pred, load_geff(TRAIN / f"{name}.geff"))
        rows.append(per_sample_metrics(er, estimated_nodes(TRAIN / f"{name}.geff"), rec))
        print(f"[{mode:>9}] {name:<26} TP/FP/FN={er.edge_tp}/{er.edge_fp}/{er.edge_fn} "
              f"J={er.edge_tp/max(1,er.edge_tp+er.edge_fp+er.edge_fn):.4f}")
    s = summarise(rows)
    results[mode] = s
    print(f"\n=== {mode.upper()} on perfect detections ===")
    print(f"  edge_jaccard     = {s['edge_jaccard']:.4f}")
    print(f"  adj_edge_jaccard = {s['adj_edge_jaccard']:.4f}")
    print(f"  division_jaccard = {s['division_jaccard']:.4f}")
    print(f"  SCORE            = {s['score']:.4f}\n")
""")

md("## 8. Summary")

code(r"""
summary = {
    "n_train": len(train_names), "n_test": len(test_names),
    "train_names": train_names, "test_names": test_names,
    "inventory": inv.to_dicts(),
    "annotated_fraction_median": float(af.median()) if len(af) else None,
    "displacement_um": {f"p{q}": float(np.percentile(disp, q))
                        for q in (50, 75, 90, 95, 99)} if len(disp) else {},
    "nn_spacing_um_median": float(np.median(nn)) if len(nn) else None,
    "nn_is_true_link_rate": (nn_correct / nn_total) if nn_total else None,
    "divisions_total": tot_div, "gt_edges_total": tot_edges,
    "linking_ceiling": {m: {k: (None if v != v else v) for k, v in s.items()
                            if isinstance(v, (int, float))} for m, s in results.items()},
}
Path("/kaggle/working/recon_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("inventory", "train_names", "test_names")}, indent=2, default=str))
print("\nWrote /kaggle/working/recon_summary.json - commit it back to the repo so the "
      "numbers are on record before we start tuning anything.")
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")

# validate
loaded = json.loads(OUT.read_text())
assert loaded["nbformat"] == 4 and len(loaded["cells"]) == len(CELLS)
import ast
for i, c in enumerate(loaded["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        stripped = "\n".join("pass  # shell" if ln.strip().startswith("!") else ln
                             for ln in src.splitlines())
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} syntax error: {e}\n---\n{src}")
print("all code cells parse as valid Python")
