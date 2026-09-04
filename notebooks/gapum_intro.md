# Is `close_gaps`'s 5.75 µm radius costing us score?

```
0.901 submitted (rank ~1388/3038)    0.938 = rank 100    0.947 gold
```

`notes/57` found `pipeline/divisions.py`'s geometry gates rejecting **88%** of real
divisions — constants adopted from a public notebook and never checked. `notes/59` applied
the same check to linking and found:

```
128,883 single-frame GT links, all dt=1
euclidean displacement   median 1.82   p90 4.14   p95 5.34   p99 8.38   max 60.76

close_gaps max_um=5.75  (vs 2-frame span)   rejects up to 23.4%
cap_edge_length max_um=14.0                 rejects 0.10%   <- correctly set
```

**But that 23.4% is an upper bound, not a measurement.** `close_gaps` bridges a two-frame
hole, and I approximated a two-frame span as **2× a single-frame step** — which assumes
straight-line motion. Real motion wanders, so the true `t → t+2` displacement is at most
twice a single step and usually less. The geometric argument says *"look here"*; it does not
say the radius costs score.

This run reads the score directly instead of inferring it. No geometry, no assumption: sweep
the radius on the cached instances and see what the metric says.

## What could make it a non-issue

`close_gaps` carries three limits, not one:

```
max_um          = 5.75     the radius under test
max_added_frac  = 0.038    ceiling on inserted nodes, as a fraction
max_added_abs   = 1650     ceiling on inserted nodes, absolute
```

If the two budget caps bind first, the radius was never the constraint and `notes/59`'s
argument is moot regardless of whether the 23.4% is right. **Prediction 2 tests exactly
that**, and it is the reason this run reports inserted-node counts per arm rather than only
scores.

There is also a reason to expect the trade to be tight rather than free. `notes/26` and
`notes/34` both measured gap-closing as the **minor** half of the repair chain — worth
**+0.0013** against `linefit_smooth`'s +0.0086 — so even a well-set radius is playing for
thousandths. And `notes/52` measured the node budget at `ratio = -0.129`: we sit 12.9% under,
so inserted nodes are affordable, but the multiplier still costs 0.1 per unit of ratio.

## The grid

The shipped chain (`gaps(2) -> smooth -> prune(6)`), one ILP solve, radius swept:

```
g5.75   the shipped value, and the anchor
g8.0    between the shipped value and p95
g10.7   notes/59's p95 of 2x single-frame spans
g14.0   cap_edge_length's value, an upper bracket
g20.0   past anything defensible, to find where it turns
```

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `g5.75` equals `claude_divsweep`'s `inc/g2sp6` — total 0.9188, `div_J`
   0.1154, 1,443 forks. Otherwise nothing below is comparable.
2. **A wider radius actually inserts more nodes** (>50 on average). If not, `max_added_frac`
   or `max_added_abs` binds first and the radius was never the constraint.
3. **Some radius beats 5.75 by more than 0.0015** (`notes/44`'s floor). The crux. Failing it
   says `notes/59`'s geometric argument does not survive contact with the metric — which is
   a result, and the reason to run this rather than just widening the constant.
4. **The best arm holds in sign on both embryos.** `notes/59` found linking geometry
   *identical* across embryos (medians 1.72 vs 1.82), unlike divisions, so this one should
   pass if anything real is happening.

*The honest outcome here is as likely to be "the constant was fine" as "widen it". Both are
worth one CPU run, and only one of them is available by reading the geometry.*
