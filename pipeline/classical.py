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

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
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
    # 'intensity' = threshold the quantile-normalised image (what we measured at 0.5552).
    # 'dog' = multi-scale Difference-of-Gaussians (notes/08). The public rule-based
    # baseline scored CV 0.682 with plain DoG and 0.824 multi-scale, against our 0.5552,
    # and its author measured multi-scale alone worth +0.040 LB. UNMEASURED here.
    detector: str = "intensity"          # 'intensity' | 'dog'
    # DoG sigma pairs in microns. A scale-space max over these catches nuclei of
    # different sizes. None -> a single (dog_small_um, dog_large_um) pair.
    dog_scales: list | None = None
    dog_small_um: float = 1.5
    dog_large_um: float = 4.0
    # Cut on the DoG response, plus a floor on the normalised intensity so that
    # band-pass noise in genuinely empty regions cannot peak.
    dog_rel_threshold: float = 0.02
    dog_abs_percentile: float = 50.0
    # Per-frame normalisation percentiles. Whole-movie quantiles (what `intensity` uses,
    # from the zarr attrs) drift with bleaching; these do not.
    dog_norm_lo: float = 1.0
    dog_norm_hi: float = 99.7

    # Intensity threshold on the quantile-normalised image. LOW is the hypothesis:
    # a missed detection is 2 permanent FN (metric findings §3), while a spurious one
    # costs only through the node budget — and `budget_fill` below bounds that cost
    # directly. This is experiment #1.
    det_threshold: float = 0.15          # MEASURED best (notes/05, 199 datasets)
    # Minimum separation between detections, in microns. Recon §4: median annotated
    # NN spacing 24.99 µm at a ~1/28 annotated fraction implies true spacing
    # 25/28^(1/3) ~ 8 µm; the 7 µm match radius is the other anchor.
    min_separation_um: float = 6.0       # recon §4 (true spacing ~8 µm)
    # Non-maximum window shape. 'box' is what 0.5552 was measured with and is kept so
    # that number stays reproducible — but it is DEGENERATE below ~6 um: _footprint
    # rounds to whole voxels, and on the 1.625 um isotropic grid that downsample=(1,4,4)
    # produces, 4.5 / 3.5 / 2.5 um ALL collapse to a (3,3,3) window. The 03 grid lost
    # four arms to that collision (notes/09 §1). Any separation sweep must use 'ball'.
    footprint: str = "box"               # 'box' | 'ball'
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
    # Choose `min_separation_um` PER DATASET so the detector naturally emits about
    # `adaptive_target * estimated_number_of_nodes / T` detections per frame, instead of
    # truncating a global setting down to that count.
    #
    # notes/12 §3: every node-count change so far splits the two embryos — `44b6` wants a
    # denser field (median budget 32,681, ~327 cells/frame) and `6bba` a sparser one
    # (9,691, ~97/frame), so `05` and `06` kept failing the gate on one embryo or the
    # other. There is no global setting that serves both. Adapting per dataset removes the
    # question: dense crops get tight separation, sparse crops get wide, with no embryo
    # assumption anywhere. And unlike `budget_fill` it hits the count by changing the
    # detector, not by discarding its output — `06` measured blind truncation at
    # -0.0264 edge Jaccard.
    adaptive_separation: bool = False    # UNMEASURED — an arm, not a default
    adaptive_target: float = 1.0         # fraction of the per-frame budget to aim at
    # Clamp, in microns. The upper bound must actually reach the sparse datasets: after
    # downsample=(1,4,4) the volume is a 104 um cube, so hitting recon's minimum budget of
    # 38 detections/frame needs ~31 um of separation, and the median (179/frame) ~18.5 um.
    # An earlier value of 12.0 only reached ~786/frame -- the densest dataset in the whole
    # corpus -- so it would have bound on essentially every run and made this a no-op.
    # NOTE: above ~9.5 um the ball footprint exceeds BALL_MAX_VOXELS and the separable box
    # is used instead. The box quantises to odd voxel counts (3.25 um steps up there), but
    # the calibration loop measures the ACHIEVED count and re-solves, so it simply
    # converges on whichever reachable separation lands closest to the target.
    adaptive_bounds: tuple[float, float] = (2.5, 32.0)
    # Threshold on a LEARNED probability map, used only when `predict_dataset` is given a
    # `prob_fn`. Kept at a floor rather than a real operating point: notes/18 §5 settles
    # density with the per-frame cap, and a threshold that binds first is exactly the
    # confound that made phase 0's "matched count" comparison meaningless.
    unet_threshold: float = 1e-6
    # Intensity-weighted sub-voxel refinement of each peak. COHERENT for the classical
    # detectors, whose coordinates are intensity maxima being refined against intensity.
    # INCOHERENT for a learned detector: peaks_from_prob already returns the sub-voxel
    # centroid of the probability plateau, and refining that against intensity applies a
    # second, uncalibrated shift which varies frame to frame with intensity noise --
    # a candidate source of the temporal jitter notes/21 §2 measured.
    refine: bool = True
    # Half-width of the temporal window handed to `prob_fn`, in frames. 0 reproduces the
    # single-frame behaviour every run up to notes/22 used; 1 stacks (t-1, t, t+1) as
    # input channels so the network can see persistence.
    #
    # This exists because of notes/21 §2: at identical node count AND identical node
    # recall the per-frame UNet's edge Jaccard is 0.0737 below DoG's, and it sits BELOW
    # the temporal-independence bound where DoG sits above it. An edge scores only if
    # both endpoints match, so a detector that wobbles between frames pays twice for
    # every wobble. Dropping `refine` closed 12 % of that deficit (notes/22); the other
    # 88 % is that nothing in the model's input or loss ever asks it to agree with the
    # frame before or after. This is the input half of that fix.
    #
    # Windows are CLAMPED at the movie boundaries, never zero-padded: t=0 sees (0, 0, 1).
    # Training must clamp identically -- a model shown a black frame at inference that it
    # never saw in training would be off-distribution exactly where nothing checks it.
    #
    # Ignored by the classical detectors, which have no temporal capacity: they are handed
    # the window's centre frame, so an arm can switch detector without also silently
    # switching what "frame t" means.
    temporal_radius: int = 0

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


# `maximum_filter` with an explicit boolean footprint costs O(N x |footprint|); with
# `size=` it is separable and costs O(N) per axis. A 12 um ball is 1,743 voxels on the
# isotropic 1.625 um grid that downsample=(1,4,4) produces, but 27,067 on a full-resolution
# anisotropic grid -- enough to wedge the process for minutes per frame. Above this many
# elements we fall back to the separable box.
BALL_MAX_VOXELS = 4_000


def _max_filter_sep(vol: np.ndarray, radius_um: float, voxel_um, want_ball: bool
                    ) -> tuple[np.ndarray, bool]:
    """Local maximum over a `radius_um` neighbourhood. Returns (filtered, used_ball).

    The box is a superset of the ball, so `vol == box_max` implies `vol == ball_max`:
    the fallback yields a SUBSET of the true ball maxima -- fewer detections, never
    spurious ones. It is a conservative degradation, and the caller is told which
    happened so it never silently changes what an experiment measured.
    """
    if want_ball:
        fp = _ball_footprint(radius_um, voxel_um)
        if int(fp.sum()) <= BALL_MAX_VOXELS:
            return maximum_filter(vol, footprint=fp, mode="nearest"), True
    return (maximum_filter(vol, size=_footprint(radius_um, voxel_um), mode="nearest"),
            False)


def _ball_footprint(radius_um: float, voxel_um) -> np.ndarray:
    """Ellipsoidal (physically spherical) neighbourhood, not a box.

    A box footprint reaches `r` along the axes but `r*sqrt(3)` into the corners, so it
    suppresses diagonal neighbours that are physically further away than the radius
    claims. On a 4:1 anisotropic grid that error is not symmetric between Z and XY.
    """
    r = [max(1, int(round(radius_um / s))) for s in voxel_um]
    grids = np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing="ij")
    d2 = sum((g * s) ** 2 for g, s in zip(grids, voxel_um))
    return d2 <= radius_um ** 2


def dog_response(vol: np.ndarray, voxel_um: tuple[float, float, float], cfg: Config
                 ) -> tuple[np.ndarray, np.ndarray]:
    """The per-frame normalised volume and its multi-scale DoG response.

    Split out of `detect_frame_dog` so the learned detector's loss mask
    (`pipeline/detector.make_loss_mask`) scores "how blob-like is this voxel" with
    exactly the filter the classical arm uses. Two implementations would drift, and the
    mask decides which voxels a network is allowed to call background.
    """
    vf = np.asarray(vol, np.float32)
    lo, hi = np.percentile(vf, [cfg.dog_norm_lo, cfg.dog_norm_hi])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((vf - lo) / (hi - lo), 0, None)

    scales = cfg.dog_scales or [(cfg.dog_small_um, cfg.dog_large_um)]
    eff = np.asarray(voxel_um, float)
    dog = None
    for s_um, l_um in scales:
        resp = (gaussian_filter(norm, sigma=s_um / eff)
                - gaussian_filter(norm, sigma=l_um / eff))
        dog = resp if dog is None else np.maximum(dog, resp)
    return norm, dog


def detect_frame_dog(
    vol: np.ndarray,
    voxel_um: tuple[float, float, float],
    cfg: Config,
    cap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-scale Difference-of-Gaussians blob detection.

    Ported from the public rule-based baseline described in
    `notes/08-leaderboard-and-dog.md`, whose author measured **multi-scale DoG alone
    worth +0.040 LB** (0.786 -> 0.826) and whose plain DoG version scored CV 0.682
    against our 0.5552 with raw-intensity thresholding.

    Three things differ from `detect_frame`, and each one has a reason:

    1. **Band-pass, not absolute intensity.** DoG responds to blob-scale structure and
       suppresses the smooth background, so a dim nucleus on a bright background still
       peaks. An absolute intensity cut cannot separate those — which is exactly the
       15.5% of GT nodes `notes/05` §1 found the threshold could no longer reach.
    2. **Per-frame normalisation.** `detect_frame` uses whole-movie quantiles from the
       zarr attrs; this renormalises each frame, so bleaching and intensity drift do not
       move the operating point through the movie.
    3. **Scale-space maximum** over several (small, large) sigma pairs, which catches
       nuclei of different sizes — recon §7 noted cells with Z spans over 24 µm
       alongside ordinary ones.
    """
    norm, dog = dog_response(vol, voxel_um, cfg)
    mx, _ = _max_filter_sep(dog, cfg.min_separation_um, voxel_um, want_ball=True)
    abs_thr = np.percentile(norm, cfg.dog_abs_percentile)
    peaks = (dog == mx) & (dog >= cfg.dog_rel_threshold) & (norm >= abs_thr)
    idx = np.argwhere(peaks)
    if idx.size == 0:
        return np.zeros((0, 3), float), np.zeros(0, float)

    scores = dog[peaks]
    order = np.argsort(scores)[::-1]
    idx, scores = idx[order], scores[order]
    cap = cfg.max_per_frame if cap is None else cap
    if cap is not None and len(idx) > cap:
        idx, scores = idx[:cap], scores[:cap]
    return idx.astype(float), scores


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
    pooled, _ = _max_filter_sep(vol, cfg.min_separation_um, voxel_um,
                                want_ball=cfg.footprint == "ball")
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

    Looked for in three places: the sibling ``.geff``'s GEFF metadata extras (where it
    lives for train), the same group's raw zarr attrs, then the image's own zarr attrs.
    Returns None if absent, and the caller then runs uncapped — see the fallback note in
    `predict_dataset`.

    The middle path exists because a ``.geff`` is just a zarr group, and the scored
    rerun has internet off: whatever is not already in the image cannot be installed.
    Reading the attrs directly means the submission needs `zarr` and nothing else, which
    is what lets `09` fit the budget regression on `train/` at runtime (`notes/14` §4).
    `notes/05` §0: `test/` ships images only, so this is never the test path — only the
    path by which the submission learns from train.
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
        try:
            import zarr
            v = _find_key(dict(zarr.open_group(str(geff_path), mode="r").attrs),
                          "estimated_number_of_nodes")
            if v is not None:
                return float(v)
        except Exception:
            pass
    if zarr_attrs:
        v = _find_key(zarr_attrs, "estimated_number_of_nodes")
        if v is not None:
            return float(v)
    return None


def open_movie(ds_path: Path | str, cfg: Config):
    """Open one dataset's image and return ``(arr, attrs, scale, voxel_um, q_lo, q_hi)``.

    ``scale`` is the stored voxel size in microns; ``voxel_um`` is that scale after
    `cfg.downsample`. Both are needed: detection works on the downsampled grid, while
    linking happens in ORIGINAL voxel coordinates.

    Everything that turns a stored zarr into the array space the detector works in —
    the coordinate scale, the downsample, and the normalisation quantiles — lives
    here, so `predict_dataset` and `budget_features` cannot drift apart.
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
    return arr, attrs, scale, voxel_um, q_lo, q_hi


def load_frame(arr, t: int, cfg: Config, q_lo: float, q_hi: float) -> np.ndarray:
    """One frame, downsampled and normalised into the detector's units."""
    dz, dy, dx = cfg.downsample
    vol = np.asarray(arr[t, ::dz, ::dy, ::dx]).astype(np.float32)
    return np.clip((vol - q_lo) / (q_hi - q_lo + 1e-6), 0.0, None)


def budget_features(
    ds_path: Path | str,
    cfg: Config,
    frac_frames: tuple[float, ...] = (0.2, 0.5, 0.8),
    ref_seps: tuple[float, ...] = (4.0, 8.0, 16.0),
) -> dict[str, float]:
    """Metadata-free proxies for a dataset's true cell count.

    `estimated_number_of_nodes` is GEFF metadata, and `notes/05` §0 found the test
    folder ships images only — so at submission time the node budget that both
    `budget_fill` and `adaptive_separation` key off cannot be read (`notes/13` §3).
    These features are what a regression can use to predict it from the image instead.

    Cheap by construction: a few frames, and detection only — no linking.

    Two kinds of peak count are returned, and the difference matters. `n_sep*` is the raw
    count; `nstrong_sep*` keeps only peaks sitting on real signal, by an ABSOLUTE
    intensity cut. `dog_abs_percentile` floors the detector at a *percentile* of the
    volume, so as a frame empties that floor sinks into the background and noise peaks
    fill the space — on synthetic movies of 12/8/6/3 nuclei the raw count runs
    20/36/50/104, i.e. backwards.

    That artefact needs volumes far emptier than this corpus: on the real data the raw
    count is +0.91 correlated with the true budget (`notes/14` §1), not negative. But the
    absolute cut is still the better feature — **+0.987, and equally strong on both
    embryos**, where the raw counts drop to +0.87 on the sparse one. Both are returned
    and the regression picks.
    """
    arr, _attrs, _scale, voxel_um, q_lo, q_hi = open_movie(ds_path, cfg)
    T = int(arr.shape[0])
    ts = sorted({min(T - 1, max(0, int(round(f * T)))) for f in frac_frames})

    counts = {s: [] for s in ref_seps}
    strong = {s: [] for s in ref_seps}
    mean_int, frac_fg = [], []
    for t in ts:
        vol = load_frame(arr, t, cfg, q_lo, q_hi)
        mean_int.append(float(vol.mean()))
        frac_fg.append(float((vol > cfg.det_threshold).mean()))
        hi_shape = np.array(vol.shape) - 1
        for s in ref_seps:
            c, _ = detect_frame_dog(vol, voxel_um, replace(cfg, min_separation_um=s))
            counts[s].append(len(c))
            # The same peaks, but only those sitting on real signal. An ABSOLUTE cut,
            # so it cannot slide into the background the way the percentile floor does.
            idx = np.clip(np.round(np.asarray(c)).astype(int), 0, hi_shape) if len(c) else None
            strong[s].append(0 if idx is None else int((vol[tuple(idx.T)] > 0.3).sum()))

    out = {f"n_sep{s:g}": float(np.mean(counts[s])) for s in ref_seps}
    out.update({f"nstrong_sep{s:g}": float(np.mean(strong[s])) for s in ref_seps})
    out["mean_int"] = float(np.mean(mean_int))
    out["frac_fg"] = float(np.mean(frac_fg))
    out["T"] = float(T)
    return out


def predict_dataset(
    ds_path: Path | str,
    cfg: Config,
    verbose: bool = True,
    est_total_nodes: float | None = None,
    prob_fn=None,
) -> Tracks:
    """Detect + link one dataset, returning a graph in ORIGINAL voxel coordinates.

    Frames are streamed one at a time, so peak memory is one volume rather than the
    whole movie. Normalisation uses the quantiles stored in the zarr attrs, so no
    extra pass over the data is needed.

    ``est_total_nodes`` overrides the per-dataset node budget; when None it is looked
    up via `estimated_total_nodes`.

    ``prob_fn`` swaps the detector for a learned one: a callable taking the normalised
    (Z, Y, X) volume and returning a probability map of the same shape. It is passed as a
    function rather than selected by a `Config.detector` string so that **torch never
    becomes an import of this module** -- it has to stay importable on the submission
    image. Everything downstream (adaptive separation, the budget cap, pruning, linking)
    is untouched, so a learned arm and a DoG arm differ in exactly one place.
    """
    ds_path = Path(ds_path)
    arr, attrs, scale, voxel_um, q_lo, q_hi = open_movie(ds_path, cfg)

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

    # Temporal window. `window(t)` is what the DETECTOR sees; `centre(x)` recovers the
    # frame those detections belong to, which is what refinement and the coordinate
    # bookkeeping downstream must use.
    rad = max(0, int(cfg.temporal_radius))

    _cache: dict[int, np.ndarray] = {}

    def _frame(t: int) -> np.ndarray:
        # CLAMP at the movie ends -- see Config.temporal_radius. Zero-padding would show
        # the network a black frame it never met in training.
        t = int(min(max(t, 0), T_full - 1))
        if t not in _cache:
            _cache[t] = load_frame(arr, t, cfg, q_lo, q_hi)
            # With radius r each frame is needed 2r+1 times; caching turns that back into
            # one zarr read per frame, and anything below t-r can never be asked for again
            # because t only ever increases.
            for k in [k for k in _cache if k < t - 2 * rad]:
                del _cache[k]
        return _cache[t]

    def window(t: int) -> np.ndarray:
        if rad == 0:
            return _frame(t)
        return np.stack([_frame(t + d) for d in range(-rad, rad + 1)])

    def centre(x: np.ndarray) -> np.ndarray:
        return x if x.ndim == 3 else x[x.shape[0] // 2]

    if prob_fn is None:
        _base = detect_frame_dog if cfg.detector == "dog" else detect_frame

        def det(x, voxel_um, cfg_, cap=None):
            # The classical detectors are single-frame. Handing them the window's centre
            # (rather than refusing) means temporal_radius changes what the LEARNED
            # detector sees and nothing else -- so a DoG arm and a UNet arm in the same
            # sweep still differ in exactly one place.
            return _base(centre(x), voxel_um, cfg_, cap=cap)
    else:
        # Imported here, not at module scope: pipeline.detector imports _max_filter_sep
        # from this module, so a top-level import would be circular.
        from pipeline.detector import peaks_from_prob

        def det(x, voxel_um, cfg_, cap=None):
            return peaks_from_prob(prob_fn(x), voxel_um, cfg_.min_separation_um,
                                   threshold=cfg_.unet_threshold, cap=cap)

    # Adaptive per-dataset separation. Calibrated on one mid-movie frame: count at the
    # configured separation, then solve for the separation that would hit the target.
    # A 3D non-maximum window of radius r excludes a volume ~r^3, so the surviving peak
    # count goes roughly as 1/r^3 and sep_new = sep_old * (n_now / n_target)^(1/3).
    # Two refinement passes, because the 1/r^3 law is only approximate once the field
    # stops being packing-limited.
    if cfg.adaptive_separation:
        if est_total_nodes is None:
            est_total_nodes = estimated_total_nodes(ds_path, attrs)
        if est_total_nodes and est_total_nodes > 0:
            target = max(1.0, cfg.adaptive_target * est_total_nodes / max(1, T_full))
            # The calibration frame goes through `window` too, so the separation is solved
            # against the same detector input the real loop will use.
            vol = window(T_full // 2)
            lo_b, hi_b = cfg.adaptive_bounds
            sep = cfg.min_separation_um
            for _ in range(3):
                n_now = len(det(vol, voxel_um, replace(cfg, min_separation_um=sep))[0])
                if n_now == 0:
                    break
                sep = float(np.clip(sep * (n_now / target) ** (1.0 / 3.0), lo_b, hi_b))
            cfg = replace(cfg, min_separation_um=sep)
            if verbose:
                print(f"    adaptive separation -> {sep:.2f} um "
                      f"(target {target:.0f} detections/frame)", flush=True)
        elif verbose:
            print("    !! no budget for this dataset — adaptive separation disabled",
                  flush=True)

    all_coords = []
    for t in range(T):
        x = window(t)
        vol = centre(x)          # refinement acts on frame t, never on its neighbours
        c, _ = det(x, voxel_um, cfg, cap=cap)
        if cfg.refine:
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
    coords[:, 1:] *= np.array(cfg.downsample, float)

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
                   budgets: dict[str, float] | None = None, prob_fn=None):
    """Adapt `predict_dataset` to the harness's ``fn(name, data_dir) -> graph``.

    ``budgets`` maps dataset name -> ``estimated_number_of_nodes``, for the case where
    the budget has to come from somewhere other than the dataset's own files.
    """
    def _fn(name: str, data_dir: Path) -> Tracks:
        return predict_dataset(Path(data_dir) / name, cfg, verbose=verbose,
                               est_total_nodes=(budgets or {}).get(name),
                               prob_fn=prob_fn)
    return _fn


__all__ = ["Config", "detect_frame", "detect_frame_dog", "refine_centroids", "link_frame", "link_all",
           "open_movie", "load_frame", "budget_features", "dog_response",
           "build_graph", "predict_dataset", "make_predictor", "estimated_total_nodes"]
