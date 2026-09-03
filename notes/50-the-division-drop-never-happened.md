# The division drop never happened, and notes/42 §3 is withdrawn

`claude_divsweep`, 24 cached instances, 2 solves × 8 post-processing chains, 1,450 s, CPU.
**1 of 5 predictions passed, and the one that passed closes the direction.**

```
arm               total  adj_edge    div_J  0.1divJ   forks     edges
ctl/raw          0.8806    0.8806   0.0000   0.0000      54   453,274
ctl/g1sp6        0.9076    0.9076   0.0000   0.0000      54   420,598
ctl/g2sp6        0.9054    0.9054   0.0000   0.0000      54   436,852
ctl/g2sp6_nk     0.9054    0.9054   0.0000   0.0000      53   436,846
inc/raw          0.9093    0.8977   0.1154   0.0115   1,443   410,906
inc/g1s          0.9179    0.9063   0.1154   0.0115   1,443   415,496   <- notes/36
inc/g2s          0.9179    0.9063   0.1154   0.0115   1,443   425,246
inc/g1sp6        0.9180    0.9064   0.1154   0.0115   1,443   398,049
inc/g2p6         0.9111    0.8996   0.1154   0.0115   1,443   411,789
inc/g2sp6        0.9188    0.9072   0.1154   0.0115   1,443   411,789   <- SHIPPED
inc/g2sp6_nk     0.9188    0.9072   0.1154   0.0115   1,443   411,789
```

---

## 1. Reproduction is exact, so the rest is comparable to the record

`inc/g1s` is `notes/36`'s arm rebuilt from the same cache: **0.9179 total, `div_J` 0.1154,
1,443 forks** — all three to the digit. `ctl` gives `div_J` 0.0000, the pack's dead division
term. The cache and solver have not moved.

## 2. Post-processing does not touch `div_J`. At all.

`div_J` is **0.1154 on every one of the eight chains**, and the fork count is **1,443 on
every one**. Gap width 1 vs 2, pruning on vs off, smoothing on vs off — none of them removes
a fork.

The obvious worry is that the chains were not running. They were: `adj_edge` moves 0.8977 →
0.9072 and the edge count moves 410,906 → 425,246 across the same arms. And
`keep_division_components` is not inert either — the **control** arm shows it working,
`ctl/g2sp6_nk` dropping 54 → 53 forks and 6 edges when the flag is turned off. Under the
incumbent weights no fork sits in a component short enough for `prune_short_tracks` to
reach.

That is coherent with `notes/36` §: a high termination penalty produces *"precision-selected
forks, not volume"* — forks embedded in long, well-established tracks. Nothing downstream is
in a position to destroy them.

## 3. `notes/42` §3 is withdrawn

> *"`notes/35` measured **0.1154** at the same ILP weights. The config audit then optimised
> the total and gave half of it back, which nobody noticed because the total went up."*

**Nothing was given back.** The two figures being compared — 0.1154 at n=24 and 0.0645 at
n=12 — were never measured on the same datasets, and `notes/44` had already measured that
n=12 sample as easy by +0.0116. On one sample the config makes no difference to `div_J`
whatsoever.

This is the same error shape as `notes/34`, `35`, `38` and `47`, and it is the **fifth**
instance: *a quantity compared across two populations and read as a change over time*. The
previous four were caught by a follow-up run. So was this one — but only because the run was
designed with the confound as its crux rather than as an afterthought, and that is the only
reason it cost 1,450 s of CPU instead of a submission slot.

`notes/42` §3 was the origin of the entire divisions direction. `notes/43` had already cut
its estimate from +0.0935 to +0.016 on the 151-event count; this removes the remaining
premise.

## 4. The shipped chain is confirmed best, and the margin is noise

```
inc/g2sp6   0.9188      what we ship
inc/g1s     0.9179      notes/36's chain
            +0.0009     under notes/44's 0.0015 floor
```

Prediction 4 asked for an arm beating the shipped chain by more than 0.0015. The best arm
**is** the shipped chain. Nothing to submit, and `0.901` stands unchanged for the second
time in three runs.

Worth recording that `max_gap` 1 vs 2 splits by ILP weight: at the control weights `g1sp6`
(0.9076) beats `g2sp6` (0.9054), at the incumbent's it reverses (0.9180 vs 0.9188). Both
gaps are inside the floor, so this is a note about why single-config sweeps mislead, not a
recommendation.

## 5. What is left

`notes/44`'s shortlist had two items. The cheap one is now closed:

- ~~**Divisions by fork suppression**~~ — closed here. `div_J` 0.1154 is what these weights
  produce and no downstream stage changes it. `notes/36` was right the first time: *"the
  division term is no longer where we are losing."*
- **A detector trained on dense labels** — `claude_zhpilot` is built and unrun. High
  variance, high ceiling, and the only remaining item that adds capacity rather than
  re-reading what we already have.

**The remaining gap is the edge term, entirely.** `adj_edge` 0.9072 on train against a
leaderboard of 0.901, bronze at 0.926. Three runs have now confirmed there is no
configuration left that moves it.

```
0.752 floor    0.901 best    0.926 bronze    0.944 gold
config: closed (notes/44, 49)    divisions: closed (notes/43, 50)
remaining: the edge term, and it needs capacity rather than tuning
```
