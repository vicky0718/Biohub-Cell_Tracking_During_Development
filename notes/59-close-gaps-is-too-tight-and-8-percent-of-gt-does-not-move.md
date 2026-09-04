# `close_gaps` is too tight, and 8.4% of ground-truth links do not move at all

`claude_linkgeom`, CPU, all 199 training datasets, **128,883 single-frame GT links**. Every
GT edge spans exactly one frame (`dt` histogram is `{1: 128883}` — no multi-frame edges
exist at all). 5 of 5 predictions passed.

```
single-frame displacement (um)   median   p90    p95    p99      max
euclidean                          1.82   4.14   5.34   8.38   60.76
|dz|                               0.00   3.25   3.25   6.50   60.12
|dy|                               0.81   2.03   2.84   5.28   25.59
|dx|                               0.81   2.44   3.25   5.69   17.88
```

---

## 1. `close_gaps max_um = 5.75` rejects roughly a quarter of real two-frame spans

```
gate                                        rejects     pct
close_gaps max_um=5.75  (vs 2-frame span)     30,128   23.4%
cap_edge_length max_um=14.0 (1-frame)            124    0.10%
```

`close_gaps` bridges a **two-frame** hole — a track ending at `t` and another starting at
`t+2` — so the span it must accommodate is two frames of motion, not one. Against that,
5.75 µm is too tight by a wide margin, and the recommended value is **≥ 10.7 µm** (p95).

`cap_edge_length` at 14.0 µm is fine: it drops 0.10% of real links. That one was set
correctly.

**The caveat, and it matters.** I approximated a two-frame span as **2× a single-frame
step**, which assumes straight-line motion. Real motion wanders, so a true `t → t+2`
displacement is *at most* twice a single step and usually less. **23.4% is therefore an
upper bound on the rejection rate, not a measurement of it.** The honest next step is not to
trust this number but to sweep `max_um` directly on the cached graphs, where the score is
read rather than inferred — cheap, CPU-only, and it needs no geometric assumption.

I am also not reporting the `linefit_smooth max_shift_um=3.2` row the grader printed
(20.4%). That cap limits how far smoothing **moves a node**, which is a different quantity
from link length; comparing them is not meaningful and the number should be ignored.

## 2. 8.4% of ground-truth links describe no motion whatsoever

```
10,772 of 128,883 links have EXACTLY 0.0 um displacement   (8.36%)
```

One GT link in twelve connects two nodes at identical coordinates. `notes/58` recorded
hengck23's two warnings — that GT tracks *"freeze after the same frame indices"* because the
volumes are crops of one master acquisition, and that some annotations are *"the result of
interpolation, e.g. annotation labels frame t=1 and t=3 and interpolates for t=2."* **Both
are confirmed here at scale.**

This project has assumed clean ground truth in every local measurement it has made. It bears
on:

- **Motion-based repairs.** `linefit_smooth` fits a line through positions that, for 8% of
  links, are frozen by annotation artefact rather than by biology. `notes/26`/`27` measured
  smoothing as the *major* half of the repair chain (+0.0086 of +0.0113), and it is being
  fit partly to non-motion.
- **Every displacement statistic above.** The median of 1.82 µm is pulled down by a
  frozen 8%; the median over moving links is higher.

The tail confirms his other flag: **max displacement 60.76 µm**, with `|dz|` reaching 60.12
against a p99 of 6.50. He posted `min dz -37, max +35  ###???` on his own data. There are
implausible jumps in the annotation, and they are rare enough to be outliers rather than
signal.

## 3. Unlike divisions, one global gate serves both embryos

```
44b6   n= 19,826   median 1.72   p99 7.20
6bba   n=109,057   median 1.82   p99 8.49      spread 0.09um
```

`notes/57` found the division gates needing to be per-embryo (sister medians 8.98 vs 11.47,
spread 2.49 µm). Linking is the opposite: the two embryos are indistinguishable, so a single
constant is the right shape here and only its *value* is wrong.

## 4. What follows

1. **Sweep `close_gaps max_um` on the cached graphs** — 5.75 against roughly 8, 10.7 and 14 —
   and read the score rather than inferring it from geometry. This is the honest version of
   §1 and it is a CPU re-solve, the same shape as `claude_divsweep`. Note `close_gaps` also
   carries `max_added_frac=0.038` and `max_added_abs=1650`, so a wider radius may simply hit
   those caps instead; the sweep will show which binds.
2. **`cap_edge_length` needs no change.**
3. **Treat frozen links as a known GT property**, not noise to model. Worth checking whether
   excluding them changes `linefit_smooth`'s measured contribution.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
n = 128,883 links, all dt=1, across 199 datasets
```
