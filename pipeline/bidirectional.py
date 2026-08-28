"""Require a link to be plausible read BOTH ways in time.

`notes/33` §2: the 0.927 public notebook runs the pack's linker forward *and* reverse and
combines the two with a weighted **harmonic** mean in probability space. Its own config
comments date the feature to their 0.915 reference run, so it predates their best score and
is not a late micro-tweak.

The harmonic mean is the whole point, and it is not interchangeable with an average. For a
candidate parent p of target c:

    combined = 1 / ((1 - w) / P_forward  +  w / P_reverse)

If **either** direction assigns the pair a near-zero probability, its reciprocal blows up
and the combined value collapses. An arithmetic mean would let a confident forward vote
carry a pair the reverse pass rejects; the harmonic mean requires *mutual* support. That is
exactly the failure `notes/26` named as `fn_mislink` — the model picking a plausible-looking
wrong parent — and a wrong parent is much less likely to look right when the question is
asked backwards.

`notes/37`: the edge term is now the entire remaining gap (raw `edge_J` 0.9047 against the
public notebooks' 0.923-0.927 total), and this is the only lever on it that needs **no new
weights**.

Two implementation notes that matter:

* The softmax is over the **source** axis. Each target has at most one parent, so the
  distribution being sharpened is "which cell did this one come from".
* The result is re-centred and re-scaled onto the *forward* logits' own mean and standard
  deviation. Everything downstream — the candidate threshold, the ILP's `edge_prob` — was
  tuned against that scale, so the blend must not move it. Skipping this silently
  re-tunes `DET_THRESHOLD` and the ILP weights at the same time.

  The mean is preserved exactly. The standard deviation is preserved **except where the
  `[0.5, 2.0]` scale-ratio clamp binds**, which `probes/exec_bidirectional.py` measures at
  1.2 % of targets at `weight=0.15` and 5.0 % at `weight=0.5`. The clamp is deliberate — an
  unclamped ratio explodes whenever one side is nearly flat, which happens on every frame
  pair with a single plausible parent — so that bind rate is the size of an unavoidable
  leak, not a bug. Worth knowing before attributing a small delta to bidirectionality.

Written against the intersection of the numpy and torch APIs so the identical code runs
inside the model loop on tensors and under `probes/exec_bidirectional.py` on arrays. The
alternative -- torch-only -- would be untestable in this container, which is how untested
numerics reach a submission.
"""
from __future__ import annotations

EPS = 1e-8
SCALE_CLAMP = (0.5, 2.0)


def _xp(a):
    """The array module for `a` — numpy or torch — plus the kwarg spelling it wants."""
    mod = type(a).__module__.split(".")[0]
    if mod == "torch":
        import torch
        return torch, "keepdim"
    import numpy as np
    return np, "keepdims"


def _mean(a, axis, xp, kw):
    return a.mean(**{"dim" if kw == "keepdim" else "axis": axis, kw: True})


def _std(a, axis, xp, kw):
    if kw == "keepdim":
        return a.float().std(dim=axis, keepdim=True, unbiased=False)
    return a.std(axis=axis, keepdims=True)


def _sum(a, axis, xp, kw):
    return a.sum(**{"dim" if kw == "keepdim" else "axis": axis, kw: True})


def _clip(a, lo, hi, xp):
    return a.clamp(lo, hi) if hasattr(a, "clamp") else a.clip(lo, hi)


def _clip_min(a, lo):
    return a.clamp_min(lo) if hasattr(a, "clamp_min") else a.clip(lo, None)


def softmax(a, axis, xp=None, kw=None):
    """Softmax written with ops numpy and torch spell the same way."""
    if xp is None:
        xp, kw = _xp(a)
    m = a.max(**({"dim": axis, "keepdim": True} if kw == "keepdim"
                 else {"axis": axis, "keepdims": True}))
    if kw == "keepdim":
        m = m.values if hasattr(m, "values") else m
    e = (a - m).exp() if hasattr(a, "exp") else xp.exp(a - m)
    return e / _sum(e, axis, xp, kw)


def calibrate(source, target, axis):
    """Shift and scale `source` onto `target`'s mean and standard deviation along `axis`.

    The scale ratio is clamped to [0.5, 2.0]: an unclamped ratio explodes when one side is
    nearly flat, which happens whenever a frame pair has a single plausible parent.
    """
    xp, kw = _xp(source)
    t_c, t_s = _mean(target, axis, xp, kw), _clip_min(_std(target, axis, xp, kw), 1e-4)
    s_c, s_s = _mean(source, axis, xp, kw), _clip_min(_std(source, axis, xp, kw), 1e-4)
    ratio = _clip(t_s / s_s, *SCALE_CLAMP, xp)
    return (source - s_c) * ratio + t_c


def harmonic_blend(forward_logits, reverse_logits, weight: float, axis: int = 1):
    """Combine forward and reverse edge logits by a weighted harmonic mean of probabilities.

    `forward_logits` and `reverse_logits` are both `(..., n_src, n_tgt)` — the reverse pass
    must already be transposed back into the forward pass's orientation, because
    `predict_edges(tgt, src, ...)` returns `(1, n_tgt, n_src)` and blending that without a
    transpose silently pairs every cell with the wrong partner.

    `weight` is the reverse pass's share. **0 returns the forward logits unchanged**, which
    is what makes the control arm exact rather than approximate.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight}")
    xp, kw = _xp(forward_logits)
    if forward_logits.shape != reverse_logits.shape:
        raise ValueError(
            f"shape mismatch {tuple(forward_logits.shape)} vs {tuple(reverse_logits.shape)} "
            "— the reverse pass is (1, n_tgt, n_src) and must be transposed first")
    if weight == 0.0:
        return forward_logits

    reverse_aligned = calibrate(reverse_logits, forward_logits, axis)
    fwd = _clip_min(softmax(forward_logits, axis, xp, kw), EPS)
    rev = _clip_min(softmax(reverse_aligned, axis, xp, kw), EPS)

    # The harmonic mean: either direction voting near-zero collapses the pair.
    combined = 1.0 / ((1.0 - weight) / fwd + weight / rev)
    combined = combined / _clip_min(_sum(combined, axis, xp, kw), EPS)

    logits = _clip_min(combined, EPS)
    logits = logits.log() if hasattr(logits, "log") else xp.log(logits)
    # Back onto the forward pass's own scale — everything downstream was tuned there.
    out = calibrate(logits, forward_logits, axis)
    return out.to(forward_logits.dtype) if hasattr(out, "to") else out


# --------------------------------------------------------------------------- source patch

ANCHOR = """            edge_logits_pair = model.predict_edges(
                unet_feat_src, unet_feat_tgt,
                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,
                p_pos_src, p_pos_tgt,
                p_mask_src, p_mask_tgt,
            )  # (1, n_src, n_tgt)
"""

INSERT = """
            if _BIDIRECTIONAL_WEIGHT > 0.0:
                _rev = model.predict_edges(
                    unet_feat_tgt, unet_feat_src,
                    p_coords_tgt * ds_arr_t, p_coords_src * ds_arr_t,
                    p_pos_tgt, p_pos_src,
                    p_mask_tgt, p_mask_src,
                ).transpose(1, 2)  # (1, n_tgt, n_src) -> (1, n_src, n_tgt)
                edge_logits_pair = _harmonic_blend(
                    edge_logits_pair, _rev, _BIDIRECTIONAL_WEIGHT)
                del _rev
"""


def patch_source(src: str, weight: float) -> str:
    """Insert the reverse pass into the pack's `predict_video` source.

    The pack ships no hook for this, so the public notebook edits the source text and
    re-execs it; this does the same, with the anchor count asserted. A patch that matches
    zero times leaves the pipeline silently unmodified and every arm becomes its control —
    the same failure mode `pipeline/deepcenter.py::load` refuses for the checkpoint.
    """
    n = src.count(ANCHOR)
    if n != 1:
        raise ValueError(
            f"bidirectional anchor matched {n} times, expected exactly 1. The pack's "
            "predict_video source has changed; re-read it and update ANCHOR rather than "
            "loosening this check — a zero-match patch runs the control under a new name.")
    out = src.replace(ANCHOR, ANCHOR + INSERT, 1)
    header = (f"_BIDIRECTIONAL_WEIGHT = {float(weight)!r}\n"
              "from pipeline.bidirectional import harmonic_blend as _harmonic_blend\n")
    return header + out
