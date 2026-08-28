"""Blend a second edge model's logits into the primary's, inside the pack's own loop.

`notes/33` §1: the 0.927 public notebook attaches three model datasets where we attach one.
`biohub-deepcenter-unet3d-center-prior-v1` turned out to be worth ~0.002 (`notes/34`). This
is the other one — `biohub-temporal-unet3d-seed314159-v1`, a second edge predictor of the
same architecture trained from a different seed — and it is the **only remaining lever that
adds model capacity** rather than re-reading the model we already have (`notes/40` §4).

The published *weights* are a public Kaggle dataset and are used as such. The public
notebook's own code is **not** usable — its `licenseName` is `None`, recorded in
`pipeline/repair.py`'s header — so the mechanism here is implemented from understanding,
the same way `pipeline/bidirectional.py` was, with its own calibration and its own tests.

## Where it hooks

The pack ships no ensemble hook, so `patch_source` edits `predict_video`'s source text at
two sites, both read from the pack's actual script rather than guessed:

    unet_out, det_logits = model.encode(imgs)        <- also encode with the secondary
    edge_logits_pair = model.predict_edges(...)      <- index, predict, blend

`unet_out` is assigned exactly once in that function and never reassigned (the detection
TTA that follows rewrites `det_logits` only), so a single encode alongside it is safe.

## How the two are combined

The secondary's logits are on their own scale, so they are first shifted and scaled onto
the primary's per-target mean and standard deviation — the same `calibrate` the
bidirectional blend uses. Mixing uncalibrated logits would let the secondary's arbitrary
temperature dominate, and the downstream candidate threshold and ILP were both tuned
against the primary's scale.

Two mixing modes:

* **`fixed`** — a constant weight everywhere. The simple baseline, and the control for the
  second mode being worth its complexity.
* **`low_margin`** — the weight scales with the *primary's own uncertainty* (its top-2
  softmax margin), and is zeroed where the two models disagree about the best parent. The
  reasoning: where the primary is confident there is nothing to gain, and where the two
  disagree outright, averaging two contradictory answers produces a blur that is worse than
  either. So the secondary is leaned on exactly where the primary is unsure and the
  secondary corroborates it.

`edge_weight=0` returns the primary unchanged **by identity**, so the control arm is
bit-identical to the unensembled pipeline rather than approximately equal.
"""
from __future__ import annotations

from pipeline.bidirectional import ANCHOR as BLEND_ANCHOR
from pipeline.bidirectional import _clip_min, _xp, calibrate, softmax

ENCODE_ANCHOR = "        unet_out, det_logits = model.encode(imgs)\n"

ENCODE_INSERT = """        _sec_out = None
        if _SECONDARY_MODEL is not None:
            _sec_out, _ = _SECONDARY_MODEL.encode(imgs)
"""

BLEND_INSERT = """
            if _SECONDARY_MODEL is not None and _SECONDARY_W > 0.0:
                _sec_src = _SECONDARY_MODEL._index_features(
                    _sec_out[:, f_idx], p_coords_src, p_mask_src,
                )
                _sec_tgt = _SECONDARY_MODEL._index_features(
                    _sec_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,
                )
                _sec_logits = _SECONDARY_MODEL.predict_edges(
                    _sec_src, _sec_tgt,
                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                    p_pos_src, p_pos_tgt,
                    p_mask_src, p_mask_tgt,
                )
                edge_logits_pair = _secondary_blend(
                    edge_logits_pair, _sec_logits, _SECONDARY_W,
                    mode=_SECONDARY_MODE, low_margin_max=_SECONDARY_MARGIN)
                del _sec_src, _sec_tgt, _sec_logits
"""


def secondary_blend(primary, secondary, weight: float, axis: int = 1,
                    mode: str = "low_margin", low_margin_max: float = 0.35):
    """Mix `secondary` into `primary` after calibrating it onto the primary's scale.

    Both are `(..., n_src, n_tgt)`. `weight` is the secondary's share; **0 returns
    `primary` unchanged by identity**, which is what makes a control arm exact.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    if mode not in ("fixed", "low_margin"):
        raise ValueError(f"mode must be 'fixed' or 'low_margin', got {mode!r}")
    if weight == 0.0:
        return primary
    if primary.shape != secondary.shape:
        raise ValueError(
            f"shape mismatch {tuple(primary.shape)} vs {tuple(secondary.shape)} — the "
            "secondary must be predicted in the primary's orientation, not transposed")

    xp, kw = _xp(primary)
    aligned = calibrate(secondary, primary, axis)
    if mode == "fixed":
        return primary * (1.0 - weight) + aligned * weight

    # low_margin: lean on the secondary only where the primary is UNSURE and the secondary
    # agrees about the best parent. Both conditions matter — see the module docstring.
    p_prob = softmax(primary, axis, xp, kw)
    s_prob = softmax(aligned, axis, xp, kw)
    p_sorted = _sort_desc(p_prob, axis, xp)
    margin = p_sorted[0] - p_sorted[1] if len(p_sorted) > 1 else p_sorted[0] * 0.0
    agree = _argmax(p_prob, axis, xp) == _argmax(s_prob, axis, xp)

    lm = max(float(low_margin_max), 1e-6)
    uncertainty = _clip_min(-(margin - lm) / lm, 0.0)
    uncertainty = uncertainty * 0 + _minimum(uncertainty, 1.0, xp)
    w = weight * uncertainty * _as_float(agree, xp)
    w = _expand(w, axis)
    return primary * (1.0 - w) + aligned * w


# --- small helpers, written against the numpy/torch intersection like bidirectional.py ---

def _sort_desc(a, axis, xp):
    if hasattr(a, "topk"):
        k = min(2, a.shape[axis])
        v = a.topk(k, dim=axis).values
        return [v.select(axis, i) for i in range(k)]
    s = xp.sort(a, axis=axis)
    s = xp.flip(s, axis=axis)
    k = min(2, a.shape[axis])
    return [xp.take(s, i, axis=axis) for i in range(k)]


def _argmax(a, axis, xp):
    return a.argmax(dim=axis) if hasattr(a, "argmax") and hasattr(a, "dim") else xp.argmax(a, axis=axis)


def _minimum(a, b, xp):
    return a.clamp(max=b) if hasattr(a, "clamp") else xp.minimum(a, b)


def _as_float(a, xp):
    return a.float() if hasattr(a, "float") else a.astype(float)


def _expand(w, axis):
    """Restore the reduced axis so the weight broadcasts against `(..., n_src, n_tgt)`."""
    if hasattr(w, "unsqueeze"):
        return w.unsqueeze(axis)
    import numpy as np
    return np.expand_dims(w, axis)


def patch_source(src: str, edge_weight: float, mode: str = "low_margin",
                 low_margin_max: float = 0.35) -> str:
    """Insert the secondary encode and the logit blend into `predict_video`'s source.

    Both anchors are asserted to match exactly once. A patch that matches zero times leaves
    the pipeline unmodified and turns every arm into its own control — the failure
    `pipeline/deepcenter.py::load` refuses for the checkpoint and `bidirectional.patch_source`
    refuses for its own anchor.
    """
    for name, anchor in (("encode", ENCODE_ANCHOR), ("blend", BLEND_ANCHOR)):
        n = src.count(anchor)
        if n != 1:
            raise ValueError(
                f"secondary {name} anchor matched {n} times, expected exactly 1. The pack's "
                "predict_video source has changed; re-read it and update the anchor rather "
                "than loosening this check — a zero-match patch runs the control under a "
                "new name.")
    out = src.replace(ENCODE_ANCHOR, ENCODE_ANCHOR + ENCODE_INSERT, 1)
    out = out.replace(BLEND_ANCHOR, BLEND_ANCHOR + BLEND_INSERT, 1)
    header = (
        "_SECONDARY_MODEL = None\n"
        f"_SECONDARY_W = {float(edge_weight)!r}\n"
        f"_SECONDARY_MODE = {mode!r}\n"
        f"_SECONDARY_MARGIN = {float(low_margin_max)!r}\n"
        "from pipeline.secondary import secondary_blend as _secondary_blend\n")
    return header + out
