# Where did half the division term go? Post-processing, with `div_J` as the read-out

```
0.901 submitted    0.926 bronze    0.944 gold
score = adj_edge_jaccard + 0.1 * div_J   (max 1.1)
```

Two numbers, from this project's own record, that cannot both describe the chain we ship:

```
notes/36   div_J 0.1154   ratio0.4_2.0 + close_gaps(max_gap=1) + linefit_smooth   n=24
notes/42   div_J 0.0645   the "best config" chain that became the submission       n=12
```

**Same ILP weights.** The difference is everything downstream of the solve: `max_gap` went
1 -> 2, and `prune_short_tracks(min_frames=6)` was added. `notes/42` §3 caught the drop and
named the cause correctly — *"the config audit optimised the total and gave half of it
back, which nobody noticed because the total went up"* — and then the direction was
dropped for divisions-by-classifier, which `notes/43` closed on 151 training events.

At stake: `div_J` 0.0645 -> 0.1154 is **+0.0051** on the score. `notes/43`'s ceiling for
driving chargeable false forks to zero is **+0.016**. The gap to bronze is 0.025.

## The confound this run exists to kill

**The two numbers are on different dataset samples**, and `notes/44` measured that exact
sample as biased: *"the 12 were an easy subset by +0.0116."* Comparing 0.1154 (n=24)
against 0.0645 (n=12) is `notes/47`'s error shape for the fifth time — a ratio read across
two populations. **The drop may not exist.**

So every arm here runs on **the same cached instances**, and prediction 2 is written so that
"the drop was a sampling artifact" is a clean, cheap close rather than a disappointment.

## Why post-processing and not the weights

The weight axis is **closed, three times**. `claude_ilp_sweep3` ran 18 settings and nothing
beat `ratio0.4_2.0` (closest −0.0009). `notes/36` §: `div_J` keeps climbing to 0.1500 at
`r5_1.6` while the score falls to 0.8615, and cheapening `division_weight` makes *more*
forks and a *worse* `div_J`. Forks are already precision-selected by the termination
penalty; the solver is not the problem.

`close_gaps` and `prune_short_tracks` are. Both were tuned by reading the **total**, and both
have a documented mechanism for destroying a correctly-found fork:

* `close_gaps` **inserts** edges. `pipeline/divisions.py`'s FP rules: a sister with
  in-degree > 0 makes the fork `malformed`, *"an automatic false positive — the worst
  possible trade."* At `max_gap=2` it bridges further and has more chances to attach a
  second parent.
* `prune_short_tracks` **deletes** components. It carries
  `keep_division_components=True`, which should protect forks — *should*. That flag has
  never been measured with `div_J` as the read-out, only asserted.
* `linefit_smooth` **moves nodes**, and the division topology test is geometric.

The cache is `claude_relink_sweep`'s, built at `det_threshold=0.985`. We ship 0.975;
`notes/44` measured the whole 0.965–0.99 interval moving the score by 0.0001, so the two are
inside one plateau. (`notes/49`: 0.999 is *not*, which is why it is not in this grid.)

## Design: one solve, many free post-chains

`notes/40`'s split, applied one layer down. The ILP solve is the expensive step, so it runs
**twice** — pack defaults and the incumbent — and eight post-processing chains are graded on
each solved graph for free.

```
raw        nothing                            g2s        gaps(2) + smooth
g1         gaps(max_gap=1)                    g2p6       gaps(2) + prune(6)      <- no smooth
g1s        gaps(1) + smooth       <- notes/36 g2sp6      gaps(2) + smooth + prune(6)  <- SHIPPED
g1sp6      gaps(1) + smooth + prune(6)        g2sp6_nk   same, keep_division_components=OFF
```

Each stage is isolated by a pair that differs in exactly one thing, so a `div_J` drop is
attributable rather than merely visible.

## Pre-registered predictions

Graded **per embryo**, both means printed, after `notes/49`.

1. **Reproduction.** `ctl` gives `div_J` ≈ 0.000 (the pack's dead division term) and
   `inc/g1s` lands within 0.010 of `notes/36`'s **0.1154**. If either fails, the cache or
   the solver has moved and nothing below is comparable to the record.
2. **The drop is real and post-processing causes it.** `inc/g2sp6` (what we ship) scores
   `div_J` at least **0.020** below `inc/g1s` *on identical datasets*. **If it does not, the
   0.1154 → 0.0645 gap was a sampling artifact, this direction closes in one cheap run, and
   that is a result** — it removes the last unexamined item on `notes/44`'s shortlist.
3. **One stage dominates.** The largest single-stage drop is more than half the total drop,
   naming a culprit instead of diffusing blame across three.
4. **The recovery is not free.** Arms that restore `div_J` lose `adj_edge_jaccard`, per
   `notes/36`'s trade. The interesting outcome is an arm that gains `0.1·div_J` **more than**
   it loses on edges — that is the only thing here worth a submission slot.
5. **It holds on both embryos.** The best arm beats `g2sp6` in sign on `44b6` *and* `6bba`.
   `notes/49`: a pooled win across crops of two embryos is not evidence about a third.

*If 2 fails, the division term is finished and the remaining gap is entirely the edge term,
exactly as `notes/36` §concluded. If 2 passes and 4 fails, we know the cost and can price
it. Only 2-and-4 together are submittable.*
