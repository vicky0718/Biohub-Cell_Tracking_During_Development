# Spend the budget: `det_threshold` below 0.965, where nobody has looked

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
adj = max(0, edge_J * (1 - 0.1 * ratio)),   ratio = (N_pred - N_est) / N_est
```

`notes/52` measured the one column nobody had read: **we sit at `ratio = -0.129`**, already
12.9% *under* the node budget, with the multiplier paying 1.013. Every attempt to collect
that bonus assumed there was over-prediction to remove. There was none, and all three
selection rules failed for that single reason:

```
notes/46     pool_kernel_um, an NMS radius            node_recall 0.983 -> 0.537
notes/48/49  det_threshold pushed UP                  0.901 -> 0.863 on the leaderboard
notes/52     track ranking under a per-dataset cap    monotonically worse
```

Put that beside `notes/51`, and the chain is mispriced in a specific, measurable way:

```
ratio        -0.129     the multiplier bonus we hold          worth ~1.3%
fn_detect       583     4.21% of GT edges, endpoint never matched -- the LARGEST bucket
fn_mislink      226     1.63%
```

**We deleted so many nodes chasing a bonus that we now miss 4.21% of ground-truth edges
through endpoints that were never detected.** That is an over-correction, and it is worth
undoing if — and only if — loosening detection brings enough of those endpoints back.

## The break-even, computed before the run

At the shipped chain `edge_J = 12,962 / 14,343`:

```
give up the multiplier entirely (ratio -0.129 -> 0)      -0.0118 on adj_edge
each recovered fn_detect edge is worth                   ~1 / 14,343
break-even                                               ~169 of 583 edges  (29%)
```

So the question is sharp: **does a ~13% rise in node count return more than 29% of the
undetected-endpoint edges?**

## Why this region is unexplored

Every threshold grid this project has run:

```
notes/40   0.99   0.985   0.975   0.96875
notes/41   0.98   0.975   0.97    0.965
notes/44   0.98   0.975   0.97
notes/48   0.975  0.999   0.9999   0.99999   0.999999
```

**The lowest value ever tried is 0.96875.** `notes/44` called the surface flat across
[0.965, 0.99] — node count moving only 5.6% — and closed the axis. `notes/49` then found a
cliff *above* it. Both statements are about a box whose lower wall was never touched, and
`notes/49`'s lesson was precisely that a sweep growing inward from a frozen value never
learns what is outside it.

Every prior move on this axis **hoarded** budget. This one spends it. That is the difference,
and it is the first time the direction of travel has been argued from a measurement of where
we actually sit rather than from the assumption that less is better.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`) — the check that 0.901 → 0.863 lacked.

1. **The anchor reproduces.** `p0.975_m6_g2` must land within 0.003 of `notes/48`'s
   **0.9410** on these same 36 datasets, or nothing below is comparable to the record.
2. **Node count rises more than 20%** from 0.975 to the lowest threshold. `notes/44`
   measured 5.6% over [0.965, 0.99]; if the sigmoid is just as saturated *below* 0.965 then
   the budget cannot be spent either, and the axis is closed in both directions by one run.
3. **`fn_detect` falls by more than 15%.** The mechanism, stated directly. Extra nodes that
   do not recover missed endpoints are pure cost, and if the count does not move then the
   583 are not threshold-recoverable at all — they are detector-capacity misses, which is
   `notes/51`'s reading and would point at `claude_zhpilot` instead.
4. **Some arm beats the anchor by more than 0.0015** (`notes/44`'s floor). The crux. This is
   where `notes/52`'s break-even arithmetic is either confirmed or refuted.
5. **The best arm holds in sign on BOTH embryos.** The test set is a third pair
   (`notes/07` §3); a pooled win across crops of two is not evidence about a third.

*The `ratio`, `mult`, `nodes` and `fn_detect` columns are printed for every arm regardless,
so a clean failure still yields the shape of the trade — which no run has ever plotted
across the region where the budget is spent rather than saved.*
