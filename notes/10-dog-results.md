# Experiment #3 — DoG wins, and the node budget flips sign

`04_dog_detector.ipynb`, run on Kaggle 2026-08-16, ~2 h on the same fixed 60-dataset
subset as `03`. Raw output in `dog_results.json`.

| arm | SCORE | edge_J | recall | nodes | budget ratio |
|---|---|---|---|---|---|
| **dog_matched** (2 scales, sep 6.0) | **0.6760** | 0.7053 | 0.857 | 1,160,180 | +0.239 |
| dog_sep4.5 (2 scales) | 0.6671 | **0.7195** | 0.904 | 1,444,570 | +0.528 |
| dog_sep7.5 (2 scales) | 0.6511 | 0.6623 | 0.775 | 876,301 | −0.018 |
| dog_singlescale | 0.6407 | 0.6736 | 0.871 | 1,219,983 | +0.314 |
| intensity_incumbent | 0.5790 | 0.5861 | 0.895 | 1,259,152 | +0.065 |
| dog_multiscale3 | 0.4377 | 0.7027 | 0.882 | 2,364,653 | **+3.040** |

**Prediction 1 CONFIRMED. `gate()` says PROMOTE: +0.0970 pooled, positive in every fold**
(+0.079, +0.090, +0.099, +0.145, +0.086). At 0.92× the incumbent's node count, so this is
a property of the detector, not of density. Largest gain since the threshold sweep.

Note `dog_matched` reaches a *lower* node recall than the incumbent (0.857 vs 0.895) and
still scores 0.097 higher. That is `notes/09` §2 restated: recall is not the objective.
Fewer, better detections beat more, worse ones.

---

## 1. ❌ My notebook tested prediction 2 wrongly, and the answer flips

The automated check compared `dog_multiscale3` (3 scales) against `dog_singlescale` and
printed FALSIFIED. That comparison is **confounded** — I calibrated `dog_rel_threshold`
once for the 2-scale configuration and reused it for the 3-scale arm, which then emitted
**2,364,653 nodes against 1,219,983**. It was never density-matched, which is the exact
error `notes/09` warned about and which this notebook was built to avoid.

Its edge Jaccard is **0.7027** — statistically indistinguishable from the winner's 0.7053.
The entire score collapse is the budget multiplier: `0.7027 × (1 − 0.1×3.040) = 0.489`.

The properly matched comparison is **`dog_matched` (2 scales, 1.16M nodes) vs
`dog_singlescale` (1 scale, 1.22M nodes)** — 0.95×, near-matched:

| | score | edge_J |
|---|---|---|
| 2 scales | 0.6760 | 0.7053 |
| 1 scale | 0.6407 | 0.6736 |
| **Δ** | **+0.0353** | **+0.0317** |

**Multi-scale is CONFIRMED at matched density**, at roughly the +0.040 the rule-based
author reported. The notebook's own verdict line is wrong and the `dog_multiscale3` arm
needs re-running with its own calibration before we know whether 3 scales beats 2.

## 2. ❌ The fold label in `04` was a lie

The log prints `folds (leave-one-embryo-out): 0:11, 1:15, 2:9, 3:8, 4:17` — **five folds**,
which is the hash split, not the two-fold embryo split. The dataset mount also moved to
`.../biohub-cell-tracking/RoyerLab-Cell_Tracking_competition`, so the uploaded snapshot
carries a `harness.py` from before `fold_by` existed.

The notebook printed "leave-one-embryo-out" as **hardcoded label text** rather than
verifying what the harness actually did. That is a bad habit in an experiment log — the
one place a claim must be derived, not asserted. `05` now asserts the fold structure and
fails loudly if it does not match.

This does not invalidate the result: +0.0970 positive across five folds is if anything a
broader test than two. But it means the **cross-embryo** question is still unmeasured, and
the next run must upload the current code.

## 3. ⭐⭐ The node budget has flipped sign — and there is ~0.05 sitting in it

`notes/05` §2 measured `budget_fill` as harmful and I set it to `None`, with the caveat
that *"any change that pushes node counts up should re-check the ratio"*. This is that
change. Every DoG arm except `sep7.5` now runs **over** budget:

| arm | ratio | multiplier | score if brought to budget |
|---|---|---|---|
| dog_sep4.5 | +0.528 | 0.947 | **0.7195** |
| dog_matched | +0.239 | 0.976 | 0.7053 |
| dog_multiscale3 | +3.040 | 0.696 | 0.7027 |
| dog_singlescale | +0.314 | 0.969 | 0.6736 |
| dog_sep7.5 | −0.018 | 1.002 | 0.6623 |

**`dog_sep4.5` has the best underlying detection+linking quality of anything we have run —
edge Jaccard 0.7195 — and is being taxed 5.3 % for over-detecting.** Bringing it to budget
is worth **+0.052**, which would land around **0.72** and past the rule-based author's
0.682 plain-DoG figure.

So `budget_fill` is back on the table, for the reason it was taken off: the sign of the
ratio changed. That is the caveat working as intended rather than a reversal of the
finding.

## 4. The density knob is separation, not threshold

The calibration cell tried to hit 242 detections/frame and never got closer than 216, even
at the extreme:

| `dog_rel_threshold` | detections/frame |
|---|---|
| 0.005 | 216 |
| 0.08 | 183 |

A 16× change in threshold moved density by 15 %. That is DoG behaving correctly — a
band-pass response with a hard local-maximum constraint is dominated by the *geometry* of
the suppression window, not by the cut value. The real density knob is
`min_separation_um`, through the continuous ball footprint:

| `min_separation_um` | nodes |
|---|---|
| 4.5 | 1,444,570 |
| 6.0 | 1,160,180 |
| 7.5 | 876,301 |

Roughly linear in the reciprocal of the window volume, and cleanly controllable — unlike
the intensity path's four-valued box footprint (`notes/09` §1).

## 5. Runtime

DoG costs about 2.3× the intensity detector (1,336 s vs 580 s per arm over 60 datasets).
Extrapolated to ~200 hidden test datasets: **~74 minutes**, against a 12 h budget. Not a
constraint.

---

## What to do next

1. **Calibrate density to the budget.** Two ways, and `05` runs both: re-enable
   `budget_fill` (a per-dataset cap, now that the ratio is positive) and sweep
   `min_separation_um` between 4.5 and 6.0. Worth ~+0.05 on the best arm.
2. **Re-run 3-scale with its own calibration.** Its edge quality already ties the winner at
   twice the density; matched, it may beat it.
3. **Upload the current code** so folds are actually leave-one-embryo-out. The
   cross-embryo generalisation of DoG is still unmeasured.
4. Standing: still no leaderboard score, and no submission notebook. At CV ≈ LB (`notes/08`
   §1) the projected 0.72 would sit around rank 1,800 of 2,402 — better, still bottom-quartile,
   and still short of the 0.89 median. **The detector swap does not close the gap to the
   pack; it makes the classical pipeline a fair baseline for the pretraining work to beat.**
