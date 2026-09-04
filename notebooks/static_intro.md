# Don't smooth what isn't moving

```
0.937 fork (rank ~320)    0.940 = rank 100    0.947 gold
```

`notes/59` measured every ground-truth link in the training set and found:

```
128,883 single-frame GT links, all dt=1
10,772 of them (8.36%) have EXACTLY 0.0 um displacement
```

**One ground-truth link in twelve connects two nodes at identical coordinates.** `notes/58`
recorded hengck23's two explanations and both are confirmed at that scale: the volumes are
crops of one master acquisition so *"tracks freeze after the same frame indices"*, and some
annotations are *"the result of interpolation — label frame t=1 and t=3 and interpolate for
t=2."*

## Why that breaks the biggest repair we have

`linefit_smooth` is the **major** half of the repair chain — `notes/26`/`27` attribute
**+0.0086 of +0.0113** to it, against gap-closing's +0.0013. It pulls each node toward a
local straight-line fit of its own track.

Where the truth is **static**, our detections still jitter. A line fitted through that
jitter has a **spurious slope**, and smoothing then drags the node *along* it — away from
the fixed position it should sit at. The fit is confidently wrong precisely where the
answer is simplest.

`pipeline/repair.py::linefit_smooth` now takes `static_um`: when the fitted speed falls
below it, the slope is zeroed and the node is pulled toward the **window mean**, which is
the right estimator for a static point. Default `0.0` is an exact no-op.

On a synthetic static track with jitter a line reads as a trend (`tests/test_repair.py`):

```
mean |error| vs the true fixed position
  raw detections        0.3000
  linefit, static off   0.2981     <- the fit buys almost nothing
  linefit, static on    0.2238     <- 25% closer
```

and a genuinely moving track (v = 2.0/frame) is bit-identical either way.

## Why this one is ours

Every other lever tried lately came from reading someone else's notebook, and `notes/60`
recorded the rule that explains why they keep failing: **a parameter already tuned on the
metric is near its optimum on the metric.** `close_gaps`' radius, the fork's division gates
— both refused to move.

`static_um` is not a re-tune of an existing constant. It is a **new term**, derived from a
property of the annotation that we measured ourselves and that appears in no public
notebook. That does not make it right; it makes it untested rather than retested.

## The grid

The shipped chain (`gaps(2) -> smooth -> prune(6)`), one ILP solve, `static_um` swept:

```
s0.0    off -- the shipped chain, and the anchor
s0.3    below the median single-frame step (1.82 um): only near-frozen chains
s0.6
s1.0
s1.8    ~the median step: aggressive, most slow chains treated as static
```

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `s0.0` equals `claude_divsweep`'s `inc/g2sp6` — 0.9188, `div_J` 0.1154,
   1,443 forks. It is an exact no-op by construction, so anything else is a bug.
2. **The fallback fires on a real fraction of nodes.** Node positions differ from `s0.0` on
   more than 1% of nodes at `s0.6`. If almost nothing changes, the fitted speeds are all
   above the threshold and the frozen links are not reaching this stage.
3. **Some `static_um` beats `s0.0` by more than 0.0015** (`notes/44`'s floor). The crux.
4. **The effect is non-monotonic** — it rises then falls. Monotone up means the grid stopped
   too early; monotone down means treating chains as static is simply worse and the
   frozen-GT reasoning does not survive contact with the metric.
5. **The best arm holds in sign on BOTH embryos.**

*`notes/59`'s zero-displacement measurement is solid and unaffected either way. What is on
trial is whether acting on it pays.*
