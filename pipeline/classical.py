"""A complete detect -> link -> submit pipeline with no training required.

Experiment #1 is settled (`notes/05-first-sweep.md`): the official baseline's
`--det-threshold 0.99` scores **0.049** on 199 datasets, and dropping it to 0.15 scores
**0.533** — +0.48, positive in all five folds. The threshold is now saturated; below 0.15
extra detections stop finding new ground-truth cells.

Every tunable is a field on `Config` with the source of its value named in the comment.
Values now come from measurement, not from the domain notes, and two of them are marked
with what falsified the earlier guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from harness.tracks import Tracks

try:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import min_weight_full_bipartite_matching as _sparse_lsa
except ImportError:  # pragma: no cover
    _sparse_lsa = None

DENSE_CAP = 4_000_000  # n_src * n_tgt above which the assignment goes sparse


@dataclass
class Config:
    """Everything tunable. Defaults come from measurement — `notes/05-first-sweep.md`
    for det_threshold and budget_fill, `notes/04-recon-results.md` for the rest."""

    # --- detection ---
    # Intensity threshold on the quantile-normalised image. LOW is the hypothesis:
    # a missed detection is 2 permanent FN (metric findings §3), while a spurious one
    # costs only through the node budget — and `budget_fill` below bounds that cost
    # directly. This is experiment #1.
    det_threshold: float = 0.15          # MEASURED best (notes/05, 199 datasets)
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
    # MEASURED AND REJECTED (notes/05 §2): the cap cost -0.0226, regressing all five
    # folds. §9's mechanism is real but its premise was not — this detector runs *under*
    # budget (pooled ratio -0.111 at threshold 0.15), so the cap could only ever bind on
    # real detections. Keep the machinery: a future detector that emits far more nodes
    # would need it. Check the ratio before assuming the bonus still applies.
    budget_fill: float | None = None     # measured worse than no cap

    # --- linking ---
    # Physical search radius. Set from MOTION, not from the 7 µm metric cutoff —
    # matching the metric buys ambiguity, not recall (domain intel §4).
    # Recon §3: displacement p99 = 8.38 µm, only 2.1% of true links exceed 7 µm — but
    # that p99 was measured on the sparse ANNOTATED set, where neighbours sit 25 µm apart.
    # The real detection field is ~210 nodes/frame at ~8 µm spacing with 15% of true
    # successors missing, and notes/05 §3 measures ~42k wrong links against ~38k missing
    # ones. Sweeping this DOWN is the next experiment.
    link_radius_um: float = 9.0          # recon §3 — untested against real detections
    # 'hungarian' = optimal one-to-one; 'greedy' = nearest neighbour, collisions allowed.
    # Recon §7: on perfect detections hungarian 0.9915 vs greedy 0.9847 — take the
    # +0.0068, it is cheap, but expect nothing more from the linker than that.
    link_mode: str = "hungarian"
    # Allow a second child per node (a division). Recon §10 settles this: a hedged
    # second child must be right >49.6% of the time to pay, and the second-nearest
    # neighbour is the true successor 0.19% of the time. Keep this off.
    allow_divisions: bool = False

    # --- graph repair (notes/06-competitor-intel.md §4) ---
    # Bridge a missing detection by INSERTING a node at t+1, turning an unusable t->t+2
    # bridge into two scoring edges. 0 = off, 1 = close one-frame gaps. Aimed straight at
    # the 15.5% of GT nodes the threshold can no longer reach (notes/05 §1), and at the
    # linking gap, since a track end next to a track start is where wrong links come from.
    gap_close: int = 0                   # UNMEASURED — an arm, not a default
    # Ceiling on inserted nodes as a fraction of detections. Both public learned pipelines
    # cap this hard (~0.045) and one is explicitly testing for over-repair.
    gap_close_max_frac: float = 0.045
    # Multiplier on the 2-frame search radius. A cell crossing a gap has two frames to
    # move, so the budget is 2 x link_radius_um, times this slack.
    gap_close_slack: float = 1.0
    # Drop detections that ended up with no edge. They cannot score but do count against
    # the node budget. Not provably free — see prune_isolated().
    prune_isolated_nodes: bool = False   # UNMEASURED — an arm, not a default

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


def build_graph(coords_tzyx: np.ndarray, edges: list[tuple[int, int]]) -> Tracks:
    """(N,4) t,z,y,x array + index pairs -> `Tracks` in voxel space.

    Plain arrays rather than a tracksdata graph: tracksdata cannot be installed on
    Kaggle, and Kaggle is where this runs. `Tracks.to_tracksdata()` converts when the
    official code is wanted. See `harness/tracks.py`.
    """
    coords_tzyx = np.asarray(coords_tzyx, float).reshape(-1, 4)
    return Tracks(coords_tzyx[:, 0], coords_tzyx[:, 1:],
                  np.asarray(edges, int).reshape(-1, 2) if edges else np.zeros((0, 2), int))


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


def close_gaps(
    coords_tzyx: np.ndarray,
    edges: list[tuple[int, int]],
    scale: tuple[float, float, float],
    cfg: Config,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Bridge one missing frame by INSERTING a node, not by spanning the gap.

    The scorer drops any edge that does not span exactly `t -> t+1`, so a direct
    `t -> t+2` link scores identically to no edge at all (metric findings §3). Inserting
    a synthetic detection at `t+1` instead converts that unusable bridge into **two
    scoring edges**, which is why both learned notebooks in `notes/06-competitor-intel.md`
    do this and cap it tightly.

    A track end at `t` is matched to a track start at `t+2` within
    `2 * link_radius_um * gap_close_slack`, by optimal assignment, and the new node goes
    at the midpoint. `gap_close_max_frac` bounds how many nodes this may add, because
    over-repair is a real failure mode — one of those notebooks is explicitly testing
    whether its remaining loss comes from exactly that.

    Returns the extended `(coords, edges)`.
    """
    if cfg.gap_close <= 0 or len(coords_tzyx) == 0:
        return coords_tzyx, edges

    n = len(coords_tzyx)
    out_deg = np.bincount([s for s, _ in edges], minlength=n) if edges else np.zeros(n, int)
    in_deg = np.bincount([d for _, d in edges], minlength=n) if edges else np.zeros(n, int)

    by_t: dict[int, list[int]] = {}
    for i in np.argsort(coords_tzyx[:, 0], kind="stable"):
        by_t.setdefault(int(coords_tzyx[i, 0]), []).append(int(i))
    phys = coords_tzyx[:, 1:] * np.asarray(scale)[None, :]

    # ceil, not floor: int() would silently disable repair on any small dataset
    budget = int(np.ceil(cfg.gap_close_max_frac * n))
    radius = 2.0 * cfg.link_radius_um * cfg.gap_close_slack
    new_coords: list[np.ndarray] = []
    new_edges: list[tuple[int, int]] = []

    for t in sorted(by_t):
        if len(new_coords) >= budget:
            break
        ends = [i for i in by_t.get(t, []) if out_deg[i] == 0]
        starts = [j for j in by_t.get(t + 2, []) if in_deg[j] == 0]
        if not ends or not starts:
            continue

        D = cdist(phys[ends], phys[starts])
        D[D > radius] = np.inf
        finite = np.isfinite(D)
        if not finite.any():
            continue
        big = D[finite].max() * 10 + 1.0
        ri, ci = linear_sum_assignment(np.where(finite, D, big))
        for i, j in zip(ri, ci):
            if not finite[i, j] or len(new_coords) >= budget:
                continue
            src, dst = ends[i], starts[j]
            mid = 0.5 * (coords_tzyx[src, 1:] + coords_tzyx[dst, 1:])
            new_idx = n + len(new_coords)
            new_coords.append(np.concatenate([[t + 1], mid]))
            new_edges += [(src, new_idx), (new_idx, dst)]
            out_deg[src] += 1
            in_deg[dst] += 1

    if not new_coords:
        return coords_tzyx, edges
    return np.vstack([coords_tzyx, np.array(new_coords)]), edges + new_edges


def prune_isolated(
    coords_tzyx: np.ndarray, edges: list[tuple[int, int]]
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Drop detections with no edge at all, and reindex.

    They cannot contribute an edge but they do count in `N_pred`, so they only ever cost
    budget multiplier. NOT provably free, though: node matching is a bipartite assignment,
    so a pruned node may currently be winning a match that would otherwise go to a linked
    node — which can change edge TP/FP. Measure it, do not assume it.
    """
    n = len(coords_tzyx)
    if n == 0:
        return coords_tzyx, edges
    keep = np.zeros(n, bool)
    for s, d in edges:
        keep[s] = keep[d] = True
    if keep.all():
        return coords_tzyx, edges
    remap = np.full(n, -1, int)
    remap[keep] = np.arange(int(keep.sum()))
    return coords_tzyx[keep], [(int(remap[s]), int(remap[d])) for s, d in edges]


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
) -> Tracks:
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

    n_before = len(coords)
    coords, edges = close_gaps(coords, edges, scale, cfg)
    if verbose and len(coords) != n_before:
        print(f"    gap closing inserted {len(coords) - n_before:,} nodes", flush=True)

    if cfg.prune_isolated_nodes:
        n_before = len(coords)
        coords, edges = prune_isolated(coords, edges)
        if verbose and len(coords) != n_before:
            print(f"    pruned {n_before - len(coords):,} isolated nodes", flush=True)

    if verbose:
        print(f"    -> {len(coords):,} nodes, {len(edges):,} links", flush=True)
    return build_graph(coords, edges)


def make_predictor(cfg: Config, verbose: bool = False,
                   budgets: dict[str, float] | None = None):
    """Adapt `predict_dataset` to the harness's ``fn(name, data_dir) -> graph``.

    ``budgets`` maps dataset name -> ``estimated_number_of_nodes``, for the case where
    the budget has to come from somewhere other than the dataset's own files.
    """
    def _fn(name: str, data_dir: Path) -> Tracks:
        return predict_dataset(Path(data_dir) / name, cfg, verbose=verbose,
                               est_total_nodes=(budgets or {}).get(name))
    return _fn


__all__ = ["Config", "detect_frame", "refine_centroids", "link_frame", "link_all",
           "build_graph", "predict_dataset", "make_predictor", "estimated_total_nodes"]
