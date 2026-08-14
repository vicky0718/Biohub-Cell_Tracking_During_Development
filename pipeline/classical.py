"""A complete detect -> link -> submit pipeline with no training required.

Purpose is twofold: get a real number on the leaderboard quickly, and give us a
detector whose threshold we control directly, so the first experiment from
`notes/02-metric-findings.md` — that the official baseline's `--det-threshold 0.99`
is tuned on the axis that barely matters — becomes a sweep instead of an argument.

Every parameter that the recon measurements should set is a field on `Config`, with
the source of its value named in the comment. Nothing here is tuned yet; the defaults
are placeholders chosen from the domain notes, not from data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import tracksdata as td
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching as _sparse_lsa
except ImportError:  # pragma: no cover
    _sparse_lsa = None

DENSE_CAP = 4_000_000  # n_src * n_tgt above which the assignment goes sparse


@dataclass
class Config:
    """Everything tunable. Values marked TODO are placeholders until recon lands."""

    # --- detection ---
    # Intensity threshold on the quantile-normalised image. LOW is the hypothesis:
    # unmatched detections cost nothing but the node budget (metric findings §1-2),
    # while a missed detection is 2 permanent FN (§3). This is experiment #1.
    det_threshold: float = 0.30          # TODO sweep
    # Minimum separation between detections, in microns. Anchored on nucleus
    # diameter ~6-17 µm (domain intel §4) and the baseline's own pool kernel.
    min_separation_um: float = 5.0       # TODO confirm from recon NN spacing
    # Optional cap on detections per frame, keeping the strongest. Guards the node
    # budget: adj_J is multiplied by (1 - 0.1*(N_pred - N_total)/N_total).
    max_per_frame: int | None = None     # TODO set from est_cells_per_frame

    # --- linking ---
    # Physical search radius. Set from MOTION, not from the 7 µm metric cutoff —
    # matching the metric buys ambiguity, not recall (domain intel §4).
    link_radius_um: float = 8.0          # TODO set from recon p99 displacement
    # 'hungarian' = optimal one-to-one; 'greedy' = nearest neighbour, collisions allowed.
    link_mode: str = "hungarian"
    # Allow a second child per node (a division). The division term is only worth
    # 0.1 and a mistimed division costs more edge Jaccard than it gains (§5),
    # so this stays off until the rest is solid.
    allow_divisions: bool = False

    # --- image handling ---
    # Strided spatial downsample (Z, Y, X). (1, 4, 4) makes the grid isotropic at
    # 1.625 µm, since the anisotropy is exactly 4:1 (domain intel §2).
    downsample: tuple[int, int, int] = (1, 4, 4)
    max_frames: int | None = None        # cap T for quick smoke runs


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------

def _footprint(min_sep_um: float, voxel_um: tuple[float, float, float]) -> tuple[int, int, int]:
    """Odd, per-axis local-max window covering `min_sep_um` in physical space.

    Anisotropy means this is NOT a cube: at (1.625, 0.40625, 0.40625) µm a 5 µm
    window is 3 voxels in Z and 13 in XY.
    """
    out = []
    for s in voxel_um:
        k = max(1, int(round(min_sep_um / s)))
        out.append(k + 1 if k % 2 == 0 else k)
    return tuple(out)


def detect_frame(
    vol: np.ndarray,
    voxel_um: tuple[float, float, float],
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-maxima detection on one (Z, Y, X) normalised volume.

    Returns ``(coords, scores)`` with coords as (N, 3) float in *this volume's* voxel
    index space, and scores the intensity at each peak.
    """
    fp = _footprint(cfg.min_separation_um, voxel_um)
    pooled = maximum_filter(vol, size=fp, mode="nearest")
    peaks = (vol == pooled) & (vol > cfg.det_threshold)
    idx = np.argwhere(peaks)
    if idx.size == 0:
        return np.zeros((0, 3), float), np.zeros(0, float)

    scores = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    if cfg.max_per_frame is not None and len(idx) > cfg.max_per_frame:
        keep = np.argpartition(-scores, cfg.max_per_frame)[: cfg.max_per_frame]
        idx, scores = idx[keep], scores[keep]
    return idx.astype(float), scores


def refine_centroids(
    vol: np.ndarray,
    coords: np.ndarray,
    voxel_um: tuple[float, float, float],
    radius_um: float = 2.5,
) -> np.ndarray:
    """Intensity-weighted centre of mass in a local window around each peak.

    Worth doing specifically in Z. A raw local max snaps to a voxel centre, and a
    2-slice Z error is 3.25 µm — 46% of the 7 µm match budget — versus 12% for the
    same error in XY (domain intel §2). Sub-voxel Z is the cheapest accuracy on offer.
    """
    if len(coords) == 0:
        return coords
    r = [max(1, int(round(radius_um / s))) for s in voxel_um]
    shape = vol.shape
    out = np.empty_like(coords)
    for n, c in enumerate(coords.astype(int)):
        lo = [max(0, c[d] - r[d]) for d in range(3)]
        hi = [min(shape[d], c[d] + r[d] + 1) for d in range(3)]
        patch = vol[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        tot = patch.sum()
        if tot <= 0:
            out[n] = c
            continue
        grids = np.meshgrid(*[np.arange(lo[d], hi[d]) for d in range(3)], indexing="ij")
        out[n] = [float((g * patch).sum() / tot) for g in grids]
    return out


# --------------------------------------------------------------------------
# linking
# --------------------------------------------------------------------------

def link_frame(A: np.ndarray, B: np.ndarray, cfg: Config) -> list[tuple[int, int]]:
    """Index pairs between consecutive frames. A and B are (N, 3) in MICRONS."""
    nA, nB = len(A), len(B)
    if nA == 0 or nB == 0:
        return []

    if cfg.link_mode == "greedy":
        d, j = cKDTree(B).query(A, k=1, distance_upper_bound=cfg.link_radius_um)
        return [(i, int(j[i])) for i in range(nA) if np.isfinite(d[i])]

    if nA * nB <= DENSE_CAP:
        D = cdist(A, B)
        ri, ci = linear_sum_assignment(D)
        return [(int(i), int(j)) for i, j in zip(ri, ci) if D[i, j] <= cfg.link_radius_um]

    sp = cKDTree(A).sparse_distance_matrix(cKDTree(B), cfg.link_radius_um,
                                           output_type="coo_matrix")
    if sp.nnz == 0:
        return []
    sp.data = sp.data + 1e-9
    if _sparse_lsa is not None:
        try:
            ri, ci = _sparse_lsa(csr_matrix(sp))
            return [(int(i), int(j)) for i, j in zip(ri, ci)]
        except Exception:
            pass
    d, j = cKDTree(B).query(A, k=1, distance_upper_bound=cfg.link_radius_um)
    return [(i, int(j[i])) for i in range(nA) if np.isfinite(d[i])]


def build_graph(coords_tzyx: np.ndarray, edges: list[tuple[int, int]]) -> td.graph.InMemoryGraph:
    """(N,4) t,z,y,x array + index pairs -> a tracksdata graph in voxel space."""
    g = td.graph.InMemoryGraph()
    for k in ("z", "y", "x"):
        g.add_node_attr_key(k, pl.Float64, -999999.0)
    ids = g.bulk_add_nodes([{"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                            for t, z, y, x in coords_tzyx])
    if edges:
        g.bulk_add_edges([{"source_id": ids[i], "target_id": ids[j]} for i, j in edges])
    return g


def link_all(coords_tzyx: np.ndarray, scale: tuple[float, float, float],
             cfg: Config) -> list[tuple[int, int]]:
    """Link every consecutive frame pair of a (N,4) t,z,y,x array."""
    by_t: dict[int, list[int]] = {}
    for i in np.argsort(coords_tzyx[:, 0], kind="stable"):
        by_t.setdefault(int(coords_tzyx[i, 0]), []).append(int(i))
    phys = coords_tzyx[:, 1:] * np.asarray(scale)[None, :]

    edges: list[tuple[int, int]] = []
    for t in sorted(by_t):
        a, b = by_t.get(t), by_t.get(t + 1)
        if not a or not b:
            continue
        for i, j in link_frame(phys[a], phys[b], cfg):
            edges.append((a[i], b[j]))
    return edges


# --------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------

def predict_dataset(
    ds_path: Path | str,
    cfg: Config,
    verbose: bool = True,
) -> td.graph.InMemoryGraph:
    """Detect + link one dataset, returning a graph in ORIGINAL voxel coordinates.

    Frames are streamed one at a time, so peak memory is one volume rather than the
    whole movie. Normalisation uses the quantiles stored in the zarr attrs, so no
    extra pass over the data is needed.
    """
    import zarr

    ds_path = Path(ds_path)
    zpath = ds_path if ds_path.suffix == ".zarr" else ds_path.with_suffix(".zarr")
    grp = zarr.open_group(str(zpath), mode="r")
    arr = grp["0"]
    attrs = dict(grp.attrs)

    scale = (1.625, 0.40625, 0.40625)
    if "multiscales" in attrs:
        tr = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if tr.get("type") == "scale":
            scale = tuple(tr["scale"][-3:])

    q = attrs.get("image_statistics", {}).get("quantiles", {})
    q_lo = float(q.get("0.001", 0.0))
    q_hi = float(q.get("0.999", 0.0))
    if q_hi <= q_lo:  # attrs missing or degenerate — fall back to the first frame
        f0 = np.asarray(arr[0]).astype(np.float32)
        q_lo, q_hi = (float(v) for v in np.quantile(f0.ravel()[::50], [0.001, 0.999]))

    dz, dy, dx = cfg.downsample
    voxel_um = (scale[0] * dz, scale[1] * dy, scale[2] * dx)

    T = arr.shape[0] if cfg.max_frames is None else min(arr.shape[0], cfg.max_frames)
    all_coords = []
    for t in range(T):
        vol = np.asarray(arr[t, ::dz, ::dy, ::dx]).astype(np.float32)
        vol = np.clip((vol - q_lo) / (q_hi - q_lo + 1e-6), 0.0, None)
        c, _ = detect_frame(vol, voxel_um, cfg)
        c = refine_centroids(vol, c, voxel_um)
        if len(c):
            all_coords.append(np.column_stack([np.full(len(c), t, float), c]))
        if verbose and (t % 25 == 0 or t == T - 1):
            print(f"    t={t:>4}/{T}  {len(c):>6} detections", flush=True)

    if not all_coords:
        return build_graph(np.zeros((0, 4)), [])

    coords = np.vstack(all_coords)
    # Back to ORIGINAL voxel coordinates — the ground truth lives in that space.
    coords[:, 1:] *= np.array([dz, dy, dx], float)

    edges = link_all(coords, scale, cfg)
    if verbose:
        print(f"    -> {len(coords):,} nodes, {len(edges):,} links", flush=True)
    return build_graph(coords, edges)


def make_predictor(cfg: Config, verbose: bool = False):
    """Adapt `predict_dataset` to the harness's ``fn(name, data_dir) -> graph``."""
    def _fn(name: str, data_dir: Path) -> td.graph.InMemoryGraph:
        return predict_dataset(Path(data_dir) / name, cfg, verbose=verbose)
    return _fn


__all__ = ["Config", "detect_frame", "refine_centroids", "link_frame", "link_all",
           "build_graph", "predict_dataset", "make_predictor"]
