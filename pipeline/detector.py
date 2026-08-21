"""Learned-detector support: training targets, loss masks, and peak extraction.

Deliberately **numpy/scipy only** — no torch. The model lives in `pipeline/unet.py`, so
this module stays importable in the submission path and testable without a GPU stack.

Why a learned detector at all (`notes/16`): with ground-truth nodes fed back in as
detections, Hungarian linking scores 1.0825 (`notes/04` §7). Everything between that and
our 0.752 is cells not found, or junk found instead. Optimal linking beats naive
nearest-neighbour by +0.0068, so the whole linking problem is worth ~0.015 and the
detector is the contest.

The quantity to optimise is **recall at a matched node count**, not recall. `notes/09` §2
reached 97.6 % recall and *lost* 0.234 of score, spending 571 spurious detections per
extra ground-truth node found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import center_of_mass, gaussian_filter, label, maximum

from harness.purescore import match_nodes
from pipeline.classical import _max_filter_sep


@dataclass
class TargetConfig:
    """How annotations become a training signal.

    ``loss`` selects which of the three candidates in `notes/16` §4 is being measured.
    None of them is a default yet — that is what phase 0 decides.
    """

    # A voxel counts as positive if it is within this distance of an annotated centre.
    # Nuclei are 4-8 um across and the grid is 1.625 um isotropic after downsample=(1,4,4),
    # so 2.0 um is a small ball of ~7 voxels per cell: enough signal, tight enough that the
    # peak still localises inside the metric's 7 um match radius.
    pos_radius_um: float = 2.0

    loss: str = "masked"  # 'naive' | 'masked' | 'pu'

    # --- 'masked' only: what counts as CLEARLY empty, and so safe to call negative. ---
    # Everything else -- the ambiguous middle, where an unannotated real cell probably
    # lives -- is dropped from the loss entirely rather than labelled background.
    #
    # The trap to avoid: masking out everything the classical detector likes would leave
    # only "DoG-positive vs empty space", and the model would learn to reproduce DoG. So
    # the DoG condition is a PERCENTILE of the frame's own response, not the detector's
    # operating threshold -- it excludes the top of the response distribution, not
    # everything DoG would emit.
    empty_intensity: float = 0.10   # normalised intensity below this, AND
    empty_dog_pct: float = 60.0     # DoG response below this percentile of the frame, AND
    empty_margin_um: float = 7.0    # further than this from any annotation

    # --- 'pu' only ---
    # Positive-unlabelled: unlabelled voxels are a mixture of background and unannotated
    # cells in a ratio we KNOW, because the GEFF carries estimated_number_of_nodes. The
    # prior is passed in per dataset rather than stored here.
    pu_clip: float = 0.95  # cap on the prior, so a bad estimate cannot invert the loss


def make_target(centres_vox: np.ndarray, shape: tuple[int, int, int],
                voxel_um, pos_radius_um: float) -> np.ndarray:
    """Binary (Z, Y, X) uint8: 1 within ``pos_radius_um`` of an annotated centre.

    ``centres_vox`` is in the voxel space of ``shape`` — i.e. already divided by the
    downsample. Out-of-range centres are dropped, not clipped: clipping would pile
    spurious positives onto the volume face.
    """
    out = np.zeros(shape, np.uint8)
    c = np.asarray(centres_vox, float).reshape(-1, 3)
    if len(c) == 0:
        return out
    inside = np.all((c >= 0) & (c < np.array(shape)), axis=1)
    c = c[inside]
    if len(c) == 0:
        return out

    vox = np.asarray(voxel_um, float)
    r = [int(np.ceil(pos_radius_um / s)) for s in vox]
    off = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in r], indexing="ij"), -1)
    ball = off[(np.sum((off * vox) ** 2, -1) <= pos_radius_um ** 2)]  # (M, 3) offsets

    base = np.round(c).astype(np.int64)
    pts = (base[:, None, :] + ball[None, :, :]).reshape(-1, 3)
    ok = np.all((pts >= 0) & (pts < np.array(shape)), axis=1)
    pts = pts[ok]
    out[pts[:, 0], pts[:, 1], pts[:, 2]] = 1
    return out


def make_loss_mask(vol: np.ndarray, dog: np.ndarray, target: np.ndarray,
                   voxel_um, cfg: TargetConfig) -> np.ndarray:
    """uint8 (Z, Y, X): 1 where the loss applies, 0 where the label is untrustworthy.

    Positives always count. Negatives count only where the voxel is clearly empty by all
    three tests. The remainder — plausible but unannotated — is excluded, which is the
    whole point: `notes/04` §5b found annotations are a *uniform random* 1/28 sample, so
    an unannotated cell is not a negative example, it is an unlabelled one.
    """
    if cfg.loss != "masked":
        return np.ones_like(target, np.uint8)

    dog_thr = float(np.percentile(dog, cfg.empty_dog_pct))
    empty = (vol < cfg.empty_intensity) & (dog < dog_thr)

    # ... and far from any annotation, so the shoulder of an annotated cell is never
    # used as a negative just because its intensity dips.
    if target.any():
        near, _ = _max_filter_sep(target.astype(np.float32), cfg.empty_margin_um,
                                  voxel_um, want_ball=True)
        empty &= near < 0.5

    return (empty | (target > 0)).astype(np.uint8)


def peaks_from_prob(prob: np.ndarray, voxel_um, min_separation_um: float,
                    threshold: float = 0.05, cap: int | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Non-maximum suppression on a probability map, returning (coords, scores).

    Same ball footprint and same top-k-by-score truncation as `detect_frame_dog`'s final
    stage, so a UNet arm and a DoG arm differ ONLY in what produced the response map.

    Two things a probability map needs that a DoG response does not, both learned the
    hard way in `tests/test_detector.py`:

    * **`threshold` must be strictly positive.** ``prob == mx`` is true across any FLAT
      region, so a background plateau of zeros makes every one of its voxels a "local
      maximum" -- 13,400 peaks from a 32^3 volume in the test that caught this.
      `detect_frame_dog` never sees it because its `dog_rel_threshold` and
      `dog_abs_percentile` cuts exclude the background first.
    * **Tied plateaus must be collapsed.** A confident network saturates: sigmoid returns
      exactly 1.0 over a whole nucleus, and every voxel of it ties. Labelling the peak
      mask and taking one centroid per connected component fixes that, and gives
      sub-voxel localisation for free -- which matters against a 7 um match radius on a
      1.625 um grid.

    The cap still does the real work of hitting a node budget; `threshold` only has to
    clear the floor.
    """
    mx, _ = _max_filter_sep(prob, min_separation_um, voxel_um, want_ball=True)
    peaks = (prob == mx) & (prob > threshold)
    if not peaks.any():
        return np.zeros((0, 3), float), np.zeros(0, float)

    lbl, n = label(peaks)
    ids = np.arange(1, n + 1)
    idx = np.asarray(center_of_mass(prob, lbl, ids), float).reshape(-1, 3)
    scores = np.asarray(maximum(prob, lbl, ids), float).reshape(-1)

    order = np.argsort(scores)[::-1]
    idx, scores = idx[order], scores[order]
    if cap is not None and len(idx) > cap:
        idx, scores = idx[:cap], scores[:cap]
    return idx, scores


def recall_at_budget(pred_t, pred_zyx, gt_t, gt_zyx, scale, max_distance: float = 7.0
                     ) -> float:
    """Fraction of ground-truth nodes matched, using the SCORER's own node matching.

    Not a proxy. `purescore.match_nodes` is the verified reimplementation of the official
    per-frame bipartite assignment (`probes/verify_purescore.py`: 47 cases, 0 mismatches),
    so a recall measured here is the recall the leaderboard would see for these nodes.
    A greedy "is anything within 7 um" check would overcount, because two GT nodes cannot
    both claim the same prediction.
    """
    if len(gt_t) == 0:
        return float("nan")
    matched = match_nodes(pred_t, pred_zyx, gt_t, gt_zyx, scale=scale,
                          max_distance=max_distance)
    return float(len(np.unique(matched[matched >= 0])) / len(gt_t))


def gaussian_heatmap(centres_vox: np.ndarray, shape, voxel_um, sigma_um: float
                     ) -> np.ndarray:
    """Soft float32 target: sum of unit Gaussians at the annotated centres, peak-scaled.

    Offered as an alternative to `make_target`'s hard ball for regression-style losses.
    Peak-normalised to 1 so the loss scale does not move with sigma.
    """
    out = np.zeros(shape, np.float32)
    c = np.asarray(centres_vox, float).reshape(-1, 3)
    inside = np.all((c >= 0) & (c < np.array(shape)), axis=1) if len(c) else np.zeros(0, bool)
    c = np.round(c[inside]).astype(np.int64)
    if len(c) == 0:
        return out
    np.add.at(out, (c[:, 0], c[:, 1], c[:, 2]), 1.0)
    sig = [sigma_um / s for s in voxel_um]
    out = gaussian_filter(out, sigma=sig)
    m = out.max()
    return (out / m).astype(np.float32) if m > 0 else out


__all__ = ["TargetConfig", "make_target", "make_loss_mask", "peaks_from_prob",
           "recall_at_budget", "gaussian_heatmap"]
