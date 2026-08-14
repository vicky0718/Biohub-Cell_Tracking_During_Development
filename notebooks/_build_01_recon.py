"""Build notebooks/01_recon.ipynb.

Design note: tracksdata requires numpy>2, which collides with the numpy<2 pin in
Kaggle's pip constraints file. So the notebook does NOT depend on tracksdata for its
analysis — ground truth is read through `geff` directly (light deps), and only the
linking-ceiling section, which needs the official scorer, requires tracksdata. Each
section is independently guarded so one failure never costs the whole run.
"""
import ast
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "01_recon.ipynb"
CELLS = []


def md(src):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": src.strip("\n").splitlines(keepends=True)})


def code(src):
    CELLS.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": src.strip("\n").splitlines(keepends=True)})


md(r"""
# Biohub Cell Tracking — Recon

Reads **only** ground-truth graphs and zarr metadata — not one image pixel — so it runs
on CPU in minutes. Set **Accelerator: None** and don't spend GPU quota here.

What it answers:

1. Inventory — datasets, frames, volume shapes, official fold splits.
2. Annotation density and the `estimated_number_of_nodes` budget the scorer measures us against.
3. Motion — per-edge displacement in µm, which sets the linking radius.
4. Confusability — cell spacing vs the 7 µm match cutoff.
5. Whether the nearest neighbour is even the right link.
5b. **Is the sparse annotation biased?** Depth, clonal clumping, division enrichment.
5c. **Frame interval** and the Z-vs-XY error budget.
6. Division counts — is the `0.1 · division_jaccard` term worth any effort.
7. **The linking-only ceiling** — perfect detections + optimal linking, scored with the
   official scorer. Splits the score into "detection problem" vs "linking problem".

Everything lands in `recon_summary.json`.

> **Dependencies.** `tracksdata` requires numpy>2 while Kaggle's image pins numpy<2, so
> the analysis reads `.geff` through the lighter `geff` package instead. Only section 7
> needs `tracksdata`; if it won't install, everything else still runs.
""")

code(r"""
# --- dependencies -------------------------------------------------------------
# Kaggle ships a pip CONSTRAINTS file that pins numpy<2; tracksdata wants numpy>2,
# which is the ResolutionImpossible you get from a plain `pip install tracksdata`.
# Clearing PIP_CONSTRAINT for the install is what lets it resolve.
import os, subprocess, sys
from pathlib import Path

print("pip constraint file:", os.environ.get("PIP_CONSTRAINT", "(none)"))

def pip_install(args, clear_constraint=True):
    env = dict(os.environ)
    if clear_constraint:
        env.pop("PIP_CONSTRAINT", None)
        env.pop("PIP_CONSTRAINTS", None)
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                       env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  pip {' '.join(args)} -> FAILED")
        print("   ", (r.stderr or r.stdout).strip().splitlines()[-1][:300])
    return r.returncode == 0

# Light and always needed: reading zarr metadata and .geff graphs.
print("\ninstalling geff + zarr ...")
pip_install(["geff", "zarr"])

# Only section 7 needs tracksdata (the official scorer imports it).
print("installing tracksdata (optional — section 7 only) ...")
pip_install(["tracksdata"])
""")

code(r"""
# --- what actually imported ---------------------------------------------------
import importlib
from pathlib import Path

def probe(mod):
    try:
        m = importlib.import_module(mod)
        return True, getattr(m, "__version__", "?")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

status = {m: probe(m) for m in ("numpy", "scipy", "zarr", "geff", "polars", "tracksdata")}
for m, (ok, info) in status.items():
    print(f"  {'OK  ' if ok else 'MISS'} {m:<12} {info}")

REQUIRED = ("numpy", "scipy", "zarr", "geff")
missing = [m for m in REQUIRED if not status[m][0]]
if missing:
    raise SystemExit(
        f"Missing required packages: {missing}\n"
        "If the install cell above succeeded but the import still fails, the kernel is "
        "holding an older module: use Run -> Restart & Run All. The packages are already "
        "installed, so the second pass works."
    )

HAVE_TRACKSDATA = status["tracksdata"][0]
if not HAVE_TRACKSDATA:
    print("\n! tracksdata unavailable -> section 7 (the linking ceiling) will be skipped.")
    print("  Everything else runs; the ceiling is the one number that needs the official scorer.")
""")

code(r"""
# --- official scorer (section 7 only) -----------------------------------------
CELLMOT = Path("/kaggle/working/kaggle-cell-tracking-competition")
if HAVE_TRACKSDATA and not (CELLMOT / "src" / "tracking_cellmot").is_dir():
    subprocess.run(["git", "clone", "--depth", "1", "--quiet",
                    "https://github.com/royerlab/kaggle-cell-tracking-competition.git",
                    str(CELLMOT)], check=False)
HAVE_SCORER = HAVE_TRACKSDATA and (CELLMOT / "src" / "tracking_cellmot").is_dir()
if HAVE_SCORER:
    sys.path.insert(0, str(CELLMOT / "src"))
print("official scorer available:", HAVE_SCORER)
""")

code(r"""
import json, warnings
from collections import Counter
from pathlib import Path          # re-imported so this cell stands alone

import numpy as np
import zarr
import geff
from geff import GeffMetadata
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

COMP = Path("/kaggle/input/competitions/biohub-cell-tracking-during-development")
if not COMP.exists():
    alt = Path("/kaggle/input/biohub-cell-tracking-during-development")
    COMP = alt if alt.exists() else COMP
TRAIN, TEST = COMP / "train", COMP / "test"
print("competition mount:", COMP, "| exists:", COMP.exists())
if COMP.exists():
    print("top level:", sorted(p.name for p in COMP.iterdir())[:20])
print("train:", TRAIN.exists(), "| test:", TEST.exists())

DEFAULT_SCALE = (1.625, 0.40625, 0.40625)   # z, y, x microns/px
MATCH_UM = 7.0                              # the scorer's node-match cutoff


def zarr_info(zpath):
    '''(T, Z, Y, X) shape and (z, y, x) scale from zarr metadata only — no pixels read.'''
    g = zarr.open_group(str(zpath), mode="r")
    attrs = dict(g.attrs)
    shape = tuple(g["0"].shape)
    scale = DEFAULT_SCALE
    if "multiscales" in attrs:
        tr = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr.get("type") == "scale":
            scale = tuple(tr["scale"][-3:])
    return shape, scale, attrs


def _find_key(obj, key):
    '''Recursively hunt for `key` anywhere in nested GEFF metadata extras.'''
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def estimated_nodes(geff_path):
    try:
        meta = GeffMetadata.read(str(geff_path))
        v = _find_key(meta.extra or {}, "estimated_number_of_nodes")
        return float(v) if v is not None else float("nan")
    except Exception:
        return float("nan")


class GT:
    '''Ground truth as plain numpy arrays — backend independent.

    t, z, y, x : (N,) per node, in VOXEL units, indexed 0..N-1
    src, dst   : (E,) node INDICES (not the file's ids)
    '''

    def __init__(self, name, path):
        self.name = name
        G, _ = geff.read(str(path), backend="networkx", structure_validation=False)
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        d = G.nodes
        self.t = np.array([d[n]["t"] for n in nodes], float)
        self.z = np.array([d[n].get("z", 0.0) for n in nodes], float)
        self.y = np.array([d[n]["y"] for n in nodes], float)
        self.x = np.array([d[n]["x"] for n in nodes], float)
        edges = list(G.edges())
        self.src = np.array([idx[u] for u, _ in edges], int) if edges else np.zeros(0, int)
        self.dst = np.array([idx[v] for _, v in edges], int) if edges else np.zeros(0, int)
        self.n_nodes, self.n_edges = len(nodes), len(edges)

    @property
    def out_deg(self):
        return np.bincount(self.src, minlength=self.n_nodes)

    @property
    def in_deg(self):
        return np.bincount(self.dst, minlength=self.n_nodes)

    def coords(self, scale=None):
        '''(N,3) z,y,x — in microns if `scale` given, else voxels.'''
        c = np.stack([self.z, self.y, self.x], axis=1)
        return c * np.asarray(scale)[None, :] if scale is not None else c

    def by_frame(self):
        '''{t: node index array}'''
        out = {}
        for i, tt in enumerate(self.t):
            out.setdefault(int(tt), []).append(i)
        return {k: np.array(v) for k, v in out.items()}


def section(fn):
    '''Run a section, report a failure, and keep going — one bad cell must not cost the run.'''
    try:
        return fn()
    except Exception:
        import traceback
        print("!! SECTION FAILED — continuing so the rest still reports\n")
        traceback.print_exc()
        return None
""")

md("## 1. Inventory")

code(r"""
# A train dataset needs BOTH the image and the ground truth. Deriving the list from
# *.zarr alone would blow up later on any zarr that ships without a paired .geff.
_train_zarr = {p.stem for p in TRAIN.glob("*.zarr")} if TRAIN.exists() else set()
_train_geff = {p.stem for p in TRAIN.glob("*.geff")} if TRAIN.exists() else set()
train_names = sorted(_train_zarr & _train_geff)
test_names  = sorted(p.stem for p in TEST.glob("*.zarr")) if TEST.exists() else []

if _train_zarr - _train_geff:
    print(f"!! {len(_train_zarr - _train_geff)} train zarr(s) have no .geff and are excluded: "
          f"{sorted(_train_zarr - _train_geff)[:5]}")
if _train_geff - _train_zarr:
    print(f"!! {len(_train_geff - _train_zarr)} .geff(s) have no image: "
          f"{sorted(_train_geff - _train_zarr)[:5]}")
print(f"{len(train_names)} usable train datasets, {len(test_names)} test datasets\n")
print("train:", train_names)
print("test: ", test_names)

splits = None
for cand in (TRAIN / "dataset_splits.json", COMP / "dataset_splits.json"):
    if cand.exists():
        splits = json.loads(cand.read_text())
        print(f"\nofficial splits at {cand}: {len(splits)} folds")
        for i, f in enumerate(splits):
            print(f"  fold {i}: {len(f.get('train', []))} train / {len(f.get('test', []))} test")
        break
if splits is None:
    print("\nNo dataset_splits.json — the harness falls back to deterministic folds.")
""")

code(r"""
# Load every ground-truth graph once and keep it; everything below reuses these.
GTS, INV = {}, []
for name in train_names:
    shape, scale, _ = zarr_info(TRAIN / f"{name}.zarr")
    g = GT(name, TRAIN / f"{name}.geff")
    GTS[name] = (g, shape, scale)
    n_est = estimated_nodes(TRAIN / f"{name}.geff")
    n_div = int((g.out_deg == 2).sum())
    INV.append(dict(name=name, T=shape[0], Z=shape[1], Y=shape[2], X=shape[3],
                    scale_z=scale[0], scale_y=scale[1], scale_x=scale[2],
                    gt_nodes=g.n_nodes, gt_edges=g.n_edges, divisions=n_div,
                    t_min=int(g.t.min()), t_max=int(g.t.max()), est_total_nodes=n_est))
    print(f"{name:<30} T={shape[0]:>4} ZYX={shape[1:]}  nodes={g.n_nodes:>8,} "
          f"edges={g.n_edges:>8,} div={n_div:>6,} est_total="
          + (f"{n_est:,.0f}" if n_est == n_est else "n/a"))
print(f"\nTOTAL: {sum(r['gt_nodes'] for r in INV):,} annotated nodes, "
      f"{sum(r['gt_edges'] for r in INV):,} edges, {sum(r['divisions'] for r in INV):,} divisions")
""")

md("## 2. Annotation density and the node budget")

code(r"""
def _s2():
    print(f"{'dataset':<30} {'T':>5} {'gt_nodes':>10} {'est_total':>12} {'annot_frac':>11} "
          f"{'gt/frame':>9} {'est/frame':>10}")
    fracs = []
    for r in INV:
        af = r["gt_nodes"] / r["est_total_nodes"] if r["est_total_nodes"] == r["est_total_nodes"] else float("nan")
        if af == af:
            fracs.append(af)
        print(f"{r['name']:<30} {r['T']:>5} {r['gt_nodes']:>10,} "
              + (f"{r['est_total_nodes']:>12,.0f}" if r['est_total_nodes'] == r['est_total_nodes'] else f"{'n/a':>12}")
              + (f" {af:>10.4f}" if af == af else f" {'n/a':>10}")
              + f" {r['gt_nodes']/max(1,r['T']):>9,.0f}"
              + (f" {r['est_total_nodes']/max(1,r['T']):>10,.0f}" if r['est_total_nodes'] == r['est_total_nodes'] else f" {'n/a':>10}"))
    if fracs:
        fr = np.array(fracs)
        print(f"\nAnnotated fraction: min={fr.min():.4f} median={np.median(fr):.4f} max={fr.max():.4f}")
        print(f"=> about 1 cell in {1/np.median(fr):.0f} is annotated (median dataset)")
        print("\nThis is the number behind 'false positives are nearly free': the scorer only "
              "ever sees edges touching that annotated minority.")
    else:
        print("\n!! No estimated_number_of_nodes in any geff. Then the adjusted metric CANNOT "
              "be reproduced locally, and the node-budget term is invisible to our CV — "
              "which would make local scores optimistic vs the leaderboard. Worth raising "
              "on the competition forum if so.")
    return fracs

FRACS = section(_s2)
""")

md("## 3. Motion — how far does a cell move between frames?")

code(r"""
def _s3():
    all_disp, all_dz, all_dxy = [], [], []
    print(f"{'dataset':<30} {'n_edges':>9} {'median':>8} {'p90':>8} {'p99':>8} {'max':>9}   (um)")
    for name in train_names:
        g, shape, scale = GTS[name]
        if g.n_edges == 0:
            continue
        dz = (g.z[g.dst] - g.z[g.src]) * scale[0]
        dy = (g.y[g.dst] - g.y[g.src]) * scale[1]
        dx = (g.x[g.dst] - g.x[g.src]) * scale[2]
        dt = g.t[g.dst] - g.t[g.src]
        d = np.sqrt(dz**2 + dy**2 + dx**2)
        all_disp.append(d); all_dz.append(np.abs(dz)); all_dxy.append(np.sqrt(dy**2 + dx**2))
        bad = int((dt != 1).sum())
        print(f"{name:<30} {g.n_edges:>9,} {np.median(d):>8.2f} {np.percentile(d,90):>8.2f} "
              f"{np.percentile(d,99):>8.2f} {d.max():>9.2f}"
              + (f"   !! {bad} edges with dt!=1" if bad else ""))
    if not all_disp:
        print("no edges found")
        return None
    disp = np.concatenate(all_disp)
    print(f"\nPOOLED per-edge displacement (um), n={len(disp):,}")
    for q in (50, 75, 90, 95, 99, 99.9):
        print(f"  p{q:<5} = {np.percentile(disp, q):8.2f}")
    print(f"  max    = {disp.max():8.2f}")
    print(f"\nTrue links moving further than the {MATCH_UM} um match radius: "
          f"{(disp > MATCH_UM).mean():.3%}")
    print("\nSet the LINKING radius from this distribution — not from the 7 um metric cutoff. "
          "p99 plus a margin is the sane choice; the lab's own zebrafish config uses 5 um.")
    return disp, np.concatenate(all_dz), np.concatenate(all_dxy)

MOTION = section(_s3)
DISP = MOTION[0] if MOTION else np.zeros(0)
""")

md("## 4. Confusability — cell spacing vs the 7 µm match radius")

code(r"""
def _s4():
    nn_all = []
    for name in train_names:
        g, shape, scale = GTS[name]
        per = []
        for t, idx in g.by_frame().items():
            if len(idx) < 2:
                continue
            P = g.coords(scale)[idx]
            per.append(cKDTree(P).query(P, k=2)[0][:, 1])
        if per:
            v = np.concatenate(per)
            nn_all.append(v)
            print(f"{name:<30} NN spacing median={np.median(v):7.2f}um  "
                  f"p10={np.percentile(v,10):6.2f}  frac<{MATCH_UM}um={np.mean(v<MATCH_UM):.2%}")
    if not nn_all:
        return None
    nn = np.concatenate(nn_all)
    print(f"\nPOOLED spacing between ANNOTATED cells: median={np.median(nn):.2f}um, "
          f"{np.mean(nn < MATCH_UM):.2%} closer than the {MATCH_UM}um match radius")
    print("CAVEAT: annotated cells are a sparse subset, so TRUE cell spacing is much tighter. "
          "If 1 cell in K is annotated, real spacing is roughly this / K**(1/3).")
    return nn

NN = section(_s4)
""")

md("## 5. Is the nearest neighbour the right link?\n\nThis is the `p` in the `link if p > J/(1+J)` rule from the metric findings.")

code(r"""
def _s5():
    ok = tot = 0
    rank_hist = Counter()
    for name in train_names:
        g, shape, scale = GTS[name]
        if g.n_edges == 0:
            continue
        frames = g.by_frame()
        true_of = dict(zip(g.src.tolist(), g.dst.tolist()))
        C = g.coords(scale)
        d_ok = d_tot = 0
        for t, idx in frames.items():
            nxt = frames.get(t + 1)
            if nxt is None or len(nxt) == 0:
                continue
            tree = cKDTree(C[nxt])
            kk = min(5, len(nxt))
            _, near = tree.query(C[idx], k=kk)
            near = np.asarray(near).reshape(len(idx), -1)
            for row, i in enumerate(idx):
                tgt = true_of.get(int(i))
                if tgt is None or g.t[tgt] != t + 1:
                    continue
                d_tot += 1
                cand = nxt[near[row]]
                hit = np.where(cand == tgt)[0]
                rank_hist[int(hit[0]) + 1 if len(hit) else ">5"] += 1
                if len(hit) and hit[0] == 0:
                    d_ok += 1
        if d_tot:
            print(f"{name:<30} NN is the true link: {d_ok/d_tot:.2%}  (n={d_tot:,})")
        ok += d_ok; tot += d_tot
    if not tot:
        return None
    print(f"\nPOOLED: the nearest annotated cell in t+1 is the true successor {ok/tot:.2%} "
          f"of the time (n={tot:,})")
    print("rank of the true target:", dict(sorted(rank_hist.items(),
                                                  key=lambda kv: (isinstance(kv[0], str), kv[0]))))
    print("\nCAVEAT: among ANNOTATED cells only. A real detector also proposes the unannotated "
          "majority, so the true competitor set is far denser and this rate is an upper bound.")
    return ok / tot

NN_RATE = section(_s5)
""")

md("""## 5b. Is the sparse annotation actually unbiased?

Ground truth comes from a **second, sparse fluorescence channel** we are not given —
Ultrack's dual-channel trick (the baseline's local data path is literally
`./data/dense_channel`). The labeling is called *random*, but random at the genetic level
is not random with respect to what we predict:

1. **Clonal clustering** — a mosaic label is inherited by both daughters, so annotations
   should clump and divisions should be over-represented.
2. **Depth bias** — a cell only becomes ground truth if it was visible in the *sparse*
   channel too, which suffers the same depth attenuation. If deep cells are missing, our
   validation is optimistic exactly where the imaging is worst.

Nothing outside the training set can settle either.
""")

code(r"""
def _depth():
    print("Annotated-node density across Z (falling deciles => deep cells under-annotated)\n")
    for name in train_names:
        g, shape, scale = GTS[name]
        Zmax = shape[1]
        hist, _ = np.histogram(g.z, bins=10, range=(0, Zmax))
        frac = hist / max(1, hist.sum())
        print(f"{name:<30} Z=0..{Zmax:<4} " + " ".join(f"{f:.2f}" for f in frac))
        print(f"{'':<30} median z={np.median(g.z):6.1f} / {Zmax}  (uniform would be {Zmax/2:.1f})")
    print("\nFlat profile => no depth bias. Falling => our CV overstates deep performance.")

section(_depth)
""")

code(r"""
def _clump():
    rng = np.random.default_rng(0)
    print("Observed vs uniform-null NN spacing (clumping is expected if labels are clonal)\n")
    for name in train_names:
        g, shape, scale = GTS[name]
        obs, null = [], []
        C = g.coords(scale)
        for t, idx in g.by_frame().items():
            if len(idx) < 10:
                continue
            P = C[idx]
            obs.append(cKDTree(P).query(P, k=2)[0][:, 1])
            Q = rng.uniform(P.min(0), P.max(0), size=P.shape)
            null.append(cKDTree(Q).query(Q, k=2)[0][:, 1])
        if not obs:
            continue
        o, n = np.concatenate(obs), np.concatenate(null)
        ratio = np.median(o) / max(1e-9, np.median(n))
        verdict = "CLUMPED" if ratio < 0.85 else ("dispersed" if ratio > 1.15 else "~uniform")
        print(f"{name:<30} observed={np.median(o):7.2f}um  null={np.median(n):7.2f}um  "
              f"ratio={ratio:.2f}  -> {verdict}")
    print("\nIf clumped, the NN statistics in section 5 are optimistic in a second way: the "
          "annotated neighbourhood is denser than a uniform subsample would be.")

section(_clump)
""")

code(r"""
def _div_enrich():
    print("Division rate within the annotated set\n")
    td_, tn_ = 0, 0
    for name in train_names:
        g, shape, scale = GTS[name]
        od = g.out_deg
        n_div = int((od == 2).sum()); n_out = int((od > 0).sum())
        td_ += n_div; tn_ += n_out
        print(f"{name:<30} {n_div:>7,} dividing / {n_out:>9,} with a successor "
              f"= {100*n_div/max(1,n_out):.3f}%")
    rate = td_ / max(1, tn_)
    print(f"\nPOOLED: {100*rate:.3f}% of annotated nodes divide.")
    if rate > 0:
        print(f"Implied cell-cycle length ~= {1/rate:,.0f} frames "
              "(1/rate, if every cell divides once per cycle).")
        print("Cross-check this against the frame interval inferred in 5c: cycle_frames x "
              "interval should land near the 20-40 min zebrafish cell cycle. If it does not, "
              "either the annotation over-represents divisions (the clonal-label prediction) "
              "or the interval estimate is wrong.")
    return rate

DIV_RATE = section(_div_enrich)
""")

md("""## 5c. Frame interval and the Z/XY error budget

The frame interval is the most consequential unknown — everything about linking scales
with it. And the 7 µm cutoff is an *isotropic physical* distance while Z voxels are 4×
coarser than XY, so a 2-slice Z error is 3.25 µm (46% of the budget) against 0.81 µm
for 2 XY pixels (12%). Z accuracy is worth roughly 4× XY accuracy.
""")

code(r"""
def _s5c():
    if MOTION is None:
        print("section 3 did not run — no displacement data")
        return None
    disp, dz, dxy = MOTION
    z_scales = {round(float(GTS[n][2][0]), 6) for n in train_names}
    print(f"per-edge |dZ|  median={np.median(dz):7.3f}um  p90={np.percentile(dz,90):7.3f}  "
          f"p99={np.percentile(dz,99):7.3f}")
    print(f"per-edge |dXY| median={np.median(dxy):7.3f}um  p90={np.percentile(dxy,90):7.3f}  "
          f"p99={np.percentile(dxy,99):7.3f}")
    if len(z_scales) == 1:
        zs = next(iter(z_scales))
        print(f"\n|dZ| in VOXELS: median={np.median(dz)/zs:.2f}  p99={np.percentile(dz,99)/zs:.2f}"
              f"   (Z voxel = {zs} um)")
        print("Median well under one voxel => Z motion is quantised by the grid, and Z centroid "
              "precision is the bottleneck exactly as the 4:1 anisotropy predicts.")
    else:
        print(f"\n!! Z scale differs across datasets: {sorted(z_scales)} — convert per dataset.")

    med = float(np.median(disp))
    print(f"\nmedian 3D displacement = {med:.3f} um/frame")
    for speed, label in ((0.83, "somitogenesis PSM ~0.83 um/min"),
                         (1.00, "epiboly azimuthal ~1.0 um/min"),
                         (3.30, "peak epiboly ~3.3 um/min")):
        print(f"  at {label:<34} -> frame interval ~= {60*med/speed:7.1f} s")
    if DIV_RATE:
        print(f"\nCross-check: cell cycle ~= {1/DIV_RATE:,.0f} frames. At the intervals above "
              "that is " + ", ".join(f"{(1/DIV_RATE)*60*med/s/60:.0f} min" for s in (0.83, 1.0, 3.3))
              + " — the one landing near 20-40 min is the plausible interval.")
    return med

MED_DISP = section(_s5c)
""")

code(r"""
def _border():
    # Dataset names like 2024_03_22_dorado_0002_0198_0184_0605 suggest these are CROPS,
    # so cells cross the boundary and appearance/disappearance there is an artifact.
    MARGIN_UM = 10.0
    print(f"Mid-movie track endpoints within {MARGIN_UM}um of a volume face\n")
    for name in train_names:
        g, shape, scale = GTS[name]
        dims_um = np.array(shape[1:]) * np.array(scale)
        P = g.coords(scale)
        near = ((P < MARGIN_UM) | (P > dims_um - MARGIN_UM)).any(axis=1)
        interior = (g.t > g.t.min()) & (g.t < g.t.max())
        starts = (g.in_deg == 0) & interior
        ends = (g.out_deg == 0) & interior
        ns, ne = int(starts.sum()), int(ends.sum())
        print(f"{name:<30} starts={ns:>7,} ({100*near[starts].mean() if ns else 0:>5.1f}% at border)  "
              f"ends={ne:>7,} ({100*near[ends].mean() if ne else 0:>5.1f}% at border)")
    print("\nHigh border fraction => appearances/disappearances are the crop edge, and should "
          "not be modelled as biology. Low => annotation dropout, a different problem.")

section(_border)
""")

md("## 6. Divisions — is the 0.1 term worth anything?")

code(r"""
tot_div = sum(r["divisions"] for r in INV)
tot_edges = sum(r["gt_edges"] for r in INV)
print(f"divisions: {tot_div:,}   GT edges: {tot_edges:,}")
print(f"divisions per 1000 GT edges: {1000*tot_div/max(1,tot_edges):.2f}")
print("\nThe division term is worth at most 0.1 of the score, and a mistimed division costs "
      "more edge Jaccard than it earns (notes/02-metric-findings.md §5). Chase it last.")
""")

md("""## 7. The linking-only ceiling

Feed the **GT nodes back in as perfect detections** and link them. Scored with the official
scorer, so it is directly comparable to a leaderboard number.

- Near 1.0 → tracking is easy and **detection is the whole contest**.
- Low → linking is genuinely hard and deserves the modelling effort.

Optimistic in one way (no unannotated distractors) and pessimistic in another (these are
dumb linkers). Read it as a decomposition, not a target.

Needs `tracksdata`; skipped automatically if it would not install.
""")

code(r"""
if not HAVE_SCORER:
    print("SKIPPED — tracksdata / official scorer unavailable.")
    print("Everything above still stands; only this ceiling number is missing.")
    CEILING = {}
else:
    import polars as pl
    import tracksdata as td
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist
    from tracking_cellmot.metrics import evaluate, per_sample_metrics, node_recall, summarise
    try:
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import min_weight_full_bipartite_matching as _sparse_lsa
    except ImportError:
        _sparse_lsa = None

    DENSE_CAP = 4_000_000     # n_src * n_tgt above which we go sparse
    LINK_RADIUS_UM = 25.0     # generous; section 3 says what is actually needed

    def build_graph(coords):
        g = td.graph.InMemoryGraph()
        for k in ("z", "y", "x"):
            g.add_node_attr_key(k, pl.Float64, -999999.0)
        ids = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                                for t, z, y, x in coords])
        return g, ids

    def _link_frame(A, B, radius_um, mode):
        nA, nB = len(A), len(B)
        if nA == 0 or nB == 0:
            return []
        if mode == "greedy":
            d, j = cKDTree(B).query(A, k=1, distance_upper_bound=radius_um)
            return [(i, int(j[i])) for i in range(nA) if np.isfinite(d[i])]
        if nA * nB <= DENSE_CAP:
            D = cdist(A, B)
            ri, ci = linear_sum_assignment(D)
            return [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= radius_um]
        sp = cKDTree(A).sparse_distance_matrix(cKDTree(B), radius_um, output_type="coo_matrix")
        if sp.nnz == 0:
            return []
        sp.data = sp.data + 1e-9
        if _sparse_lsa is not None:
            try:
                ri, ci = _sparse_lsa(csr_matrix(sp))
                return [(int(i), int(j)) for i, j in zip(ri, ci)]
            except Exception:
                pass
        d, j = cKDTree(B).query(A, k=1, distance_upper_bound=radius_um)
        return [(i, int(j[i])) for i in range(nA) if np.isfinite(d[i])]

    def link_all(coords, scale, mode, radius_um=LINK_RADIUS_UM):
        by_t = {}
        for i in np.argsort(coords[:, 0], kind="stable"):
            by_t.setdefault(int(coords[i, 0]), []).append(int(i))
        phys = coords[:, 1:] * np.asarray(scale)[None, :]
        edges = []
        for t in sorted(by_t):
            a, b = by_t.get(t), by_t.get(t + 1)
            if not a or not b:
                continue
            for i, j in _link_frame(phys[a], phys[b], radius_um, mode):
                edges.append((a[i], b[j]))
        return edges

    def load_td_geff(path):
        r = td.graph.IndexedRXGraph.from_geff(str(path))
        return r[0] if isinstance(r, tuple) else r

    import time
    for name in train_names:
        g, _, _ = GTS[name]
        pf = g.n_nodes / max(1, len(np.unique(g.t)))
        print(f"{name:<30} {g.n_nodes:>9,} nodes, ~{pf:,.0f}/frame"
              + ("   (dense)" if pf ** 2 <= DENSE_CAP else "   (SPARSE)"))

    CEILING = {}
    for mode in ("hungarian", "greedy"):
        rows = []
        for name in train_names:
            g, shape, scale = GTS[name]
            coords = np.column_stack([g.t, g.z, g.y, g.x])
            t0 = time.time()
            pred, ids = build_graph(coords)
            e = link_all(coords, scale, mode)
            if e:
                pred.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in e])
            gt_td = load_td_geff(TRAIN / f"{name}.geff")
            er = evaluate(pred, gt_td, scale=scale, max_distance=MATCH_UM)
            rec = node_recall(pred, gt_td)
            rows.append(per_sample_metrics(er, estimated_nodes(TRAIN / f"{name}.geff"), rec))
            print(f"[{mode:>9}] {name:<28} TP/FP/FN={er.edge_tp:,}/{er.edge_fp:,}/{er.edge_fn:,} "
                  f"J={er.edge_tp/max(1,er.edge_tp+er.edge_fp+er.edge_fn):.4f} "
                  f"({len(e):,} links, {time.time()-t0:.1f}s)", flush=True)
        s = summarise(rows)
        CEILING[mode] = s
        print(f"\n=== {mode.upper()} on perfect detections ===")
        for k in ("edge_jaccard", "adj_edge_jaccard", "division_jaccard", "score"):
            print(f"  {k:<18} = {s[k]:.4f}")
        print()
""")

md("## 8. Summary")

code(r"""
def _clean(d):
    return {k: (None if isinstance(v, float) and v != v else v)
            for k, v in d.items() if isinstance(v, (int, float, str))}

summary = {
    "n_train": len(train_names), "n_test": len(test_names),
    "train_names": train_names, "test_names": test_names,
    "inventory": [_clean(r) for r in INV],
    "annotated_fraction_median": float(np.median(FRACS)) if FRACS else None,
    "displacement_um": {f"p{q}": float(np.percentile(DISP, q))
                        for q in (50, 75, 90, 95, 99)} if len(DISP) else {},
    "median_displacement_um": MED_DISP,
    "nn_spacing_um_median": float(np.median(NN)) if NN is not None and len(NN) else None,
    "nn_is_true_link_rate": NN_RATE,
    "division_rate": DIV_RATE,
    "divisions_total": tot_div, "gt_edges_total": tot_edges,
    "linking_ceiling": {m: _clean(s) for m, s in CEILING.items()},
    "have_tracksdata": HAVE_TRACKSDATA,
}
Path("/kaggle/working/recon_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps({k: v for k, v in summary.items()
                  if k not in ("inventory", "train_names", "test_names")}, indent=2, default=str))
print("\nWrote /kaggle/working/recon_summary.json")
""")

nb = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}},
      "nbformat": 4, "nbformat_minor": 5}
OUT.write_text(json.dumps(nb, indent=1))
print(f"wrote {OUT} ({len(CELLS)} cells)")

for i, c in enumerate(json.loads(OUT.read_text())["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        stripped = "\n".join("pass  # shell" if l.strip().startswith("!") else l
                             for l in src.splitlines())
        try:
            ast.parse(stripped)
        except SyntaxError as e:
            raise SystemExit(f"cell {i} syntax error: {e}\n---\n{src}")
print("all code cells parse as valid Python")
