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
    """Everything tunable. Defaults now come from recon (`notes/04-recon-results.md`)
    except `det_threshold`, which is the axis we are here to sweep."""

    # --- detection ---
    # Intensity threshold on the quantile-normalised image. LOW is the hypothesis:
    # a missed detection is 2 permanent FN (metric findings §3), while a spurious one
    # costs only through the node budget — and `budget_fill` below bounds that cost
    # directly. This is experiment #1.
    det_threshold: float = 0.30          # the sweep axis
    # Minimum separation between detections, in microns. Recon §4: median annotated
    # NN spacing 24.99 µm at a ~1/28 annotated fraction implies true spacing
    # 25/28^(1/3) ~ 8 µm; the 7 µm match radius is the other anchor.
    min_separation_um: float = 6.0       # recon §4 (true spacing ~8 µm)
    # Hard cap on detections per frame, keeping the strongest. Overrides the budget
    # cap below when set; normally leave it None and let `budget_fill` do the work.
    max_per_frame: int | None = None
    # Fraction of the dataset's OWN node budget to spend, read per dataset from the
    # GEFF `estimated_number_of_nodes`:  cap = budget_fill * est_total / T.
    # Recon §9: the budget varies 20.8x across datasets (3,783 -> 78,644) and 11x
    # between the two test datasets carrying 95% of the weight. A fixed global
    # threshold emits a roughly fixed count, which zeroes out the sparse crops —
    # the multiplier max(0, 1 - 0.1*(N_pred - N_total)/N_total) reaches 0 at 11x over.
    # 1.0 sits exactly at budget; slightly under earns the uncapped bonus of §7.
    budget_fill: float | None = 1.0      # recon §9 — a required guard, not a nicety

    # --- linking ---
    # Physical search radius. Set from MOTION, not from the 7 µm metric cutoff —
    # matching the metric buys ambiguity, not recall (domain intel §4).
    # Recon §3: displacement p99 = 8.38 µm, only 2.1% of true links exceed 7 µm.
    link_radius_um: float = 9.0          # recon §3 (p99 8.38 µm)
    # 'hungarian' = optimal one-to-one; 'greedy' = nearest neighbour, collisions allowed.
    # Recon §7: on perfect detections hungarian 0.9915 vs greedy 0.9847 — take the
    # +0.0068, it is cheap, but expect nothing more from the linker than that.
    link_mode: str = "hungarian"
    # Allow a second child per node (a division). Recon §10 settles this: a hedged
    # second child must be right >49.6% of the time to pay, and the second-nearest
    # neighbour is the true successor 0.19% of the time. Keep this off.
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
    cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Local-maxima detection on one (Z, Y, X) normalised volume.

    Returns ``(coords, scores)`` with coords as (N, 3) float in *this volume's* voxel
    index space, and scores the intensity at each peak.

    ``cap`` is the per-frame detection limit for *this dataset*, normally derived from
    its own node budget by `predict_dataset`; it overrides `cfg.max_per_frame`.
    """
    fp = _footprint(cfg.min_separation_um, voxel_um)
    pooled = maximum_filter(vol, size=fp, mode="nearest")
    peaks = (vol == pooled) & (vol > cfg.det_threshold)
    idx = np.argwhere(peaks)
    if idx.size == 0:
        return np.zeros((0, 3), float), np.zeros(0, float)

    scores = vol[idx[:, 0], idx[:, 1], idx[:, 2]]
    cap = cfg.max_per_frame if cap is None else cap
    if cap is not None and len(idx) > cap:
        keep = np.argpartition(-scores, cap)[:cap]
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

def _find_key(obj, key: str):
    """Depth-first search for `key` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def estimated_total_nodes(ds_path: Path | str, zarr_attrs: dict | None = None) -> float | None:
    """The scorer's ``N_total`` for one dataset, or None if it cannot be found.

    This is the denominator of the node-budget multiplier
    ``max(0, J * (1 - 0.1*(N_pred - N_total)/N_total))``, and recon §9 shows it varies
    20.8x across datasets — so it has to be read per dataset, not assumed.

    Looked for in two places, because it is *unconfirmed* whether the test folder ships
    ``.geff`` metadata at all: the sibling ``.geff``'s GEFF metadata extras (where it
    lives for train), then the image's own zarr attrs. Returns None if absent, and the
    caller then runs uncapped — see the fallback note in `predict_dataset`.
    """
    ds_path = Path(ds_path)
    geff_path = ds_path if ds_path.suffix == ".geff" else ds_path.with_suffix(".geff")
    if geff_path.exists():
        try:
            from geff import GeffMetadata
            meta = GeffMetadata.read(str(geff_path))
            v = _find_key(meta.extra or {}, "estimated_number_of_nodes")
            if v is not None:
                return float(v)
        except Exception:  # metadata is a nicety; never let it break a prediction
            pass
    if zarr_attrs:
        v = _find_key(zarr_attrs, "estimated_number_of_nodes")
        if v is not None:
            return float(v)
    return None


def predict_dataset(
    ds_path: Path | str,
    cfg: Config,
    verbose: bool = True,
    est_total_nodes: float | None = None,
) -> td.graph.InMemoryGraph:
    """Detect + link one dataset, returning a graph in ORIGINAL voxel coordinates.

    Frames are streamed one at a time, so peak memory is one volume rather than the
    whole movie. Normalisation uses the quantiles stored in the zarr attrs, so no
    extra pass over the data is needed.

    ``est_total_nodes`` overrides the per-dataset node budget; when None it is looked
    up via `estimated_total_nodes`.
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

    T_full = arr.shape[0]
    T = T_full if cfg.max_frames is None else min(T_full, cfg.max_frames)

    # Per-dataset detection cap from that dataset's own node budget (recon §9). The
    # budget is for the whole movie, so divide by the FULL length, not the capped one.
    cap = cfg.max_per_frame
    if cfg.budget_fill is not None:
        if est_total_nodes is None:
            est_total_nodes = estimated_total_nodes(ds_path, attrs)
        if est_total_nodes is not None and est_total_nodes > 0:
            cap = max(1, int(cfg.budget_fill * est_total_nodes / max(1, T_full)))
            if cfg.max_per_frame is not None:
                cap = min(cap, cfg.max_per_frame)
        elif verbose:
            print("    !! no estimated_number_of_nodes for this dataset — running "
                  "UNCAPPED; the budget multiplier is unguarded here", flush=True)

    all_coords = []
    for t in range(T):
        vol = np.asarray(arr[t, ::dz, ::dy, ::dx]).astype(np.float32)
        vol = np.clip((vol - q_lo) / (q_hi - q_lo + 1e-6), 0.0, None)
        c, _ = detect_frame(vol, voxel_um, cfg, cap=cap)
        c = refine_centroids(vol, c, voxel_um)
        if len(c):
            all_coords.append(np.column_stack([np.full(len(c), t, float), c]))
        if verbose and (t % 25 == 0 or t == T - 1):
            print(f"    t={t:>4}/{T}  {len(c):>6} detections"
                  + (f"  (cap {cap})" if cap is not None else ""), flush=True)

    if not all_coords:
        return build_graph(np.zeros((0, 4)), [])

    coords = np.vstack(all_coords)
    # Back to ORIGINAL voxel coordinates — the ground truth lives in that space.
    coords[:, 1:] *= np.array([dz, dy, dx], float)

    edges = link_all(coords, scale, cfg)
    if verbose:
        print(f"    -> {len(coords):,} nodes, {len(edges):,} links", flush=True)
    return build_graph(coords, edges)


def make_predictor(cfg: Config, verbose: bool = False,
                   budgets: dict[str, float] | None = None):
    """Adapt `predict_dataset` to the harness's ``fn(name, data_dir) -> graph``.

    ``budgets`` maps dataset name -> ``estimated_number_of_nodes``, for the case where
    the budget has to come from somewhere other than the dataset's own files.
    """
    def _fn(name: str, data_dir: Path) -> td.graph.InMemoryGraph:
        return predict_dataset(Path(data_dir) / name, cfg, verbose=verbose,
                               est_total_nodes=(budgets or {}).get(name))
    return _fn


__all__ = ["Config", "detect_frame", "refine_centroids", "link_frame", "link_all",
           "build_graph", "predict_dataset", "make_predictor", "estimated_total_nodes"]
