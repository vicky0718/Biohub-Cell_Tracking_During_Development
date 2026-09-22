# The node-count axis is closed on data, not on argument

2026-09-22, same day as `notes/89`. Three measurements, one CPU kernel and two GPU arms that
were already in flight when the axis was closed by someone else's board. They agree.

## 1. There is no node budget on the test side, and none can be inferred

`claude-probe` (CPU, no GPU quota) mounted the competition and looked:

```
TEST SIDE
  entries by suffix: {'.zarr': 4}
  non-zarr entries in test/: []
  files anywhere in the mount mentioning estimated_number_of_nodes, test side: 0
```

`T_est` is read by the scorer from the ground-truth `.geff`, and the test movies ship no
ground truth of any kind. So the per-movie calibration that `notes/89` §3 wanted — trim the
movies where we over-predict, leave the rest — cannot be done from the data.

Nor can `T_est` be *predicted*, and the reason is sharper than "the regression is hard":

```
every one of the 199 training movies:   shape [100, 64, 256, 256]
T_est across them:                      5,161 ... 78,644      (15x)
```

**The geometry is a constant and the budget varies 15-fold.** Frames, volume and voxel grid
are identical in every movie in the competition, so there is nothing observable on a test
movie for a regression to use. This is not a missing feature; it is an absent one.

## 2. Label density, measured on all 199 movies, both embryo types

```
type   movies   T_est (median)   GT nodes (median)   density: min / median / max
44b6       71          32,681                 214     0.13%  /  0.77%  /  5.02%
6bba      128           9,691                 826     0.93%  /  9.71%  / 20.21%
all       199                                         0.13%  /  3.56%  / 20.21%
                                            pooled GT 133,318 of 4,725,117 = 2.82%
```

This is `zhincez`'s claim — *"one has 0.2 to 1.3 percent of its nodes labelled, the other has
13 to 15"* — reproduced from the files, and it is the mechanism behind their four lost
submissions. On a 44b6 movie, 99.2% of what we emit has no labelled counterpart, so deleting
it is invisible to the validator and fully visible to the node-count multiplier.

It also explains why the two embryo types cannot share a deletion policy, which is what their
per-type experiment was reaching for and why it still lost.

## 3. And on our own pipeline the deletion trade is negative *before* the board sees it

`det975` and `mtl12` were launched before `notes/89` closed the axis and were left to finish.
Their value is that they price the trade on our stack, with both halves separated:

```
arm       change                    Δadj      from J    from mult   mult share   Δ rows
det975    DET_THRESHOLD 0.975     -0.0029    -0.0040     +0.0011        22%      -1,628
mtl12     OUTPUT_MIN_TRACK_LEN 12 -0.0115    -0.0176     +0.0062        26%      -9,479
lb30      learned bonus 3.0       +0.0014    +0.0015     +0.0000         0%         +51
```

The multiplier bonus is real, positive and exactly the size the formula predicts. It is also
**3.6x and 2.8x smaller than the Jaccard it costs.**

The important part is where that was measured: on the *training* movies, whose labels are
2.82% dense and whose checkpoint saw all 199 of them (`notes/81`). That is the friendliest
terrain deletion will ever get, and it loses there. The hidden test set charges more, not
less — which is precisely what `zhincez` and `rogerrogerroger3r` measured from their boards.

**Closed three ways: no budget to calibrate against, no signal to predict one from, and a
trade that loses on the most favourable data available.** Cost: one CPU kernel and two GPU
hours that were already spent. No submission slots.

## 4. What this leaves

`lb30` is unchanged by all of it and is the one arm still standing: +0.0014 with **0%** from
the multiplier and node count +51 in 122,794. By `notes/89` §3's rule — and by
`rogerrogerroger3r`'s independent statement of the same rule, *"the proxy is admissible only
for comparing two candidates whose node count is essentially unchanged"* — that is the single
shape the offline validator is entitled to rank, and it is the one lb30 has.

Running: `tight60` (pushed 21:57), `lb50` (22:00), then `geofus`, `lb30t60`, `flow2`, `x138`.

## 5. An infrastructure fact worth more than the arms

`/kernels/status` answered `429 TooManyRequests` for over half an hour while `/kernels/list`,
`/kernels/output` and the submissions list all answered normally in the same second. **Kaggle
rate-limits its endpoints on separate budgets**, and status is both the one this project
exhausts and the least informative of them — it returns a word where output returns the log.

`tools/collect.py` now reads completion off the artefact instead of the state machine: a
finished kernel's log ends in Kaggle's NbConvert trailer and is empty while it runs, which
makes "log exists and ends in the trailer" a terminal signal that cannot be rate-limited out
of existence. That is how `det975`, `mtl12` and `claude-probe` were collected at 22:24 while
`/kernels/status` was still refusing every call.
