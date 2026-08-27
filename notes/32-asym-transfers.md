# 0.880 → 0.883. The lab's own asymmetry transfers — and the two transfer ratios disagree.

`claude_submit_ilp` v1, leaderboard **0.883**. One weight changed from the 0.880 chain:
`disappearance_weight` 0.1 → 0.5. Confirmed minimal by diffing the two generated workers —
one functional line, plus its comment and the summary filename.

```
0.752  classical champion
0.843  pack, ILP bypassed
0.867  pack, ILP running
0.880  + gap-close(5.75 µm) + linefit-smooth(0.76, win 2)
0.883  + disappearance 0.1 -> 0.5                          <- here
0.913–0.916  the cluster, same weights                     <- 0.030–0.033 away
```

**+0.003 on the leaderboard against +0.0037 measured on training data** (`notes/31`:
`asym0.1_0.5+repair` 0.8958 vs `control+repair` 0.8921).

---

## 1. The substantive win: a constant from a different acquisition moved the test set

This is the first time in this project that a **non-repair mechanism** has produced a
positive leaderboard delta. Everything else that ever moved the score was either the pack's
own weights (0.752 → 0.867) or geometric graph repair (0.867 → 0.880).

The change is not a fitted number. `notes/03` §3 recorded the same lab's own zebrafish
Ultrack config using `appear_weight = -0.002`, `disappear_weight = -0.01` — a deliberate
**5× asymmetry**, "discouraging track termination more than initiation." The submitted arm
keeps appearance at the pack's 0.1 and raises disappearance to 0.5: the lab's ratio, at the
pack's scale. That transferred to a different acquisition, a different metric, and a hidden
test set.

`notes/31` §2 had already ruled out the obvious alternative explanation on train — the grid
paired every asymmetric arm with a **symmetric arm at matched magnitude**, and asymmetry
beat magnitude by +0.0017. The leaderboard does not re-test that decomposition (only the
winning arm was submitted), so the attribution still rests on the train-side control. Worth
stating rather than letting the LB result launder it into something it did not measure.

## 2. The transfer ratios do not agree, and I am not going to average them

Two measurements now exist:

```
repair chain     train +0.0115   ->   LB +0.013    ratio 1.13x   (notes/28)
ILP asymmetry    train +0.0037   ->   LB +0.003    ratio 0.81x   (here)
```

`notes/28` §2 explicitly refused to treat 1.13× as a coefficient — "a direction, not a
coefficient, and I will not use it to project the next result." That refusal was correct
and this is why.

**But the point estimates overstate the disagreement, because the leaderboard is reported
to three decimals.** Propagating that quantization honestly:

```
                 reported     true LB delta       implied ratio
repair chain     +0.013       (0.0120, 0.0140)    (1.04, 1.22)
ILP asymmetry    +0.003       (0.0020, 0.0040)    (0.54, 1.08)
                                       overlap:   (1.04, 1.08)
```

The ranges overlap on a sliver. A single common transfer ratio of ~1.05 is *not* excluded —
it requires the repair chain to have landed at the bottom of its range and this arm at the
top of its, simultaneously, which is possible but unlikely. So: **the two measurements
probably differ, and the evidence that they differ is weak.** Not "transfer collapsed."

**Cheap fix, worth doing before either number is used again:** if the Kaggle leaderboard
exposes more than three decimals for this competition, reading the actual values off it
collapses both intervals to nothing and settles this outright. That costs no submission
slot and no compute.

What follows either way: a train-measured gain is worth roughly its face value, sometimes a
little more, sometimes a little less. It is not a multiplier to be applied. Both directional
conclusions — repair helps, asymmetry helps — held on test, and that is the part with a
track record.

## 3. What this does and does not license

**Licensed.** `notes/31` §3 flagged that `asym0.1_0.5` is the *largest* asymmetry in the
grid, so the optimum may lie past the boundary. The leaderboard confirms the direction is
real on unseen data, which is exactly the evidence that makes extending the grid worth the
solver time rather than a fishing trip.

**Not licensed.** The mechanism has a documented cost. `notes/31` §3's anatomy at this arm:

```
                control+repair -> asym0.1_0.5+repair
fn_mislink            411     ->      382    (-29)
fn_gap                177     ->      159    (-18)
fn_detect             218     ->      254    (+36)   <- the tax
```

Raising the disappearance penalty makes the solver reluctant to end a track, so it links
through ambiguity it previously abandoned. Some of those links are right; some strand nodes
that no longer match. The net is positive **here**, and that ratio is what inverts if the
knob is pushed too far. The grid extension is a search for where it inverts, not an
assumption that more is better.

## 4. Next — two follow-ups on the cache already on Kaggle, no GPU, no submission slot

`claude_relink_sweep`'s `cand_*.npz` (24 datasets, candidate edges with probabilities) is
still attachable as a `kernelDataSource`. Both of these re-solve cached instances:

1. **`division_weight` in the unsaturated gap (0.5, 1.0).** `notes/31` §1 found the knob
   samples only two points — default (54 forks, `div_J` 0.0000) and saturation (7,149 forks,
   `div_J` 0.0562) — with 0.5/0.2/0.0/−0.5 giving byte-identical results. The two endpoints
   price out to a **wash within 0.0002**. Whether an intermediate fork count buys the
   division term without the full edge cost is unsampled, and `division_jaccard` is 0.0562
   of an available 1.000, so the headroom is real even if the price currently is not.
2. **Extend the asymmetry grid past 0.1/0.5** — §3 above.

Both belong in one notebook, `claude_ilp_sweep2`, since they share the same cache load and
the same control. Control must reproduce **0.8806**, as in every run since `notes/26`.

Banked floor **0.752**. Best scored submission **0.883**. Cluster **0.913–0.916**.
