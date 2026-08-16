# First real scores — the threshold sweep, and a prediction of mine that died

`02_classical_baseline.ipynb`, run on Kaggle 2026-08-16. All **199 train datasets, full
100 frames**, ~4.6 hours. Raw output in `sweep_results.json`.

Dataset mount for future runs:
`/kaggle/input/datasets/vigneshnehru/biohub-cell-tracking/biohub-cell_tracking_during_development`

---

## 0. The two §0 questions, answered

**`test/` ships images only — 4 `.zarr`, zero `.geff`.** So `estimated_number_of_nodes`
is *not* readable at test time. We know the four budgets anyway, because those four names
also appear in train and recon read their GEFF metadata (§9). But a general solution
cannot rely on that.

**The leak is real.** `sample_submission.csv` names exactly
`44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1` — the four datasets whose
ground truth ships in `train/`. Anyone can score ~1.0 by echoing the training annotations.

**Action: tell the organisers. Do not exploit it.** And treat any leaderboard position
built on it as meaningless — ours included, until we know whether a private split exists.

## 1. The sweep — the prediction was right, and by a lot

Pre-registered in `notes/02-metric-findings.md`: the official baseline's
`--det-threshold 0.99` is tuned on the axis that barely matters, and the score should
*improve* as the threshold falls.

| threshold | score | edge_J | node_recall | predicted nodes |
|---|---|---|---|---|
| **0.99** (baseline default) | **0.0490** | 0.0450 | 0.064 | 173,670 |
| 0.70 | 0.2583 | 0.2438 | 0.372 | 1,432,235 |
| 0.50 | 0.4413 | 0.4275 | 0.652 | 2,955,581 |
| 0.30 | 0.5295 | 0.5215 | 0.815 | 3,929,697 |
| **0.15** | **0.5327** | 0.5282 | 0.845 | 4,200,408 |
| 0.05 | 0.5230 | 0.5211 | 0.846 | 4,346,107 |

**PROMOTE**, pooled `+0.4836`, positive in all five folds (+0.434 … +0.520). The largest
single result we have: **our detector's threshold belongs near 0.15, not near 1.0.**

> **Correction (2026-08-16, `notes/06-competitor-intel.md` §2).** This section originally
> read the result as "the official baseline's default costs 0.48 of score". That was a
> category error and is withdrawn. Our `det_threshold` cuts *quantile-normalised image
> intensity*; the baseline's `--det-threshold` cuts the *TemporalUNet3D's predicted centre
> probability*. Competitors run the latter at 0.96875–0.985 and score 0.9+ on the
> leaderboard. The measured fact — where **our** threshold belongs — is unaffected.

**The threshold is now saturated**, and that matters more than the winner:

| step | nodes gained | recall gained |
|---|---|---|
| 0.30 → 0.15 | +270,711 | +0.030 |
| 0.15 → 0.05 | +145,699 | **+0.0014** |

Below 0.15, extra detections stop finding new ground-truth cells. **The remaining 15.5 %
of GT nodes are not threshold-limited** — they are suppressed by the non-maximum window
(`min_separation_um = 6.0`) or invisible in this channel. Threshold is a spent knob; the
next detection experiment is the separation window.

## 2. ❌ FALSIFIED: the node-budget cap made things worse

`notes/04-recon-results.md` §9 argued the per-dataset detection cap was "a required guard,
not a nicety". Measured, paired, same datasets:

| | score |
|---|---|
| budget cap **ON** (`budget_fill=1.0`) | 0.5327 |
| budget cap **OFF** (`budget_fill=None`) | **0.5552** |

**REJECT, −0.0226 pooled, regressing in all five folds.** Not noise, not a fold artefact —
uniformly worse.

**Why I was wrong.** §9's analysis of the *metric* was correct: the multiplier really does
reach zero at 11× over budget. The *empirical premise* was wrong. This detector does not
over-produce — it **under**-produces:

| threshold | predicted nodes | pooled budget ratio |
|---|---|---|
| 0.30 | 3,929,697 | **−0.168** |
| 0.15 | 4,200,408 | **−0.111** |
| 0.05 | 4,346,107 | **−0.080** |

against a pooled budget of 4,725,117. Every arm sits *under* the budget, collecting the
uncapped under-prediction bonus rather than paying a penalty. The cap could therefore only
ever bind on real detections, and it did: it cost recall to protect against a penalty that
was never going to be charged.

The per-dataset table confirms it dataset by dataset — every `total_node_ratio` in the top
15 by weight is negative (−0.13 … −0.00).

**What survives:** the *mechanism* is real and still worth guarding against if a future
detector produces far more nodes. **What dies:** the claim that a cap is needed *now*.
`budget_fill` now defaults to `None`, and any change that pushes node counts up should
re-check the ratio before assuming the bonus still applies.

The trap description in §9 was also over-stated in one respect: it assumed a detector
calibrated to a dense crop would emit that density everywhere. A *threshold*-based detector
does not — it emits fewer detections where there is less signal, which is exactly the
adaptive behaviour §9 said we would have to engineer.

## 3. ⭐⭐ The real headroom is in LINKING, not detection

This is the finding that redirects the project, and it contradicts recon §7's headline.

At the best arm: `node_recall = 0.8446`, `edge_jaccard = 0.5282`.

If linking were as good as recon's measured ceiling, edge Jaccard would be about
`recall² × 0.9915 = 0.7073`.

**Actual: 0.5282. Shortfall: 0.179.**

And 0.179 is a *lower* bound on the loss, because `recall²` assumes the two endpoints of an
edge are matched independently — they are not. A cell bright enough to detect in frame `t`
is usually bright enough in `t+1`, so true edge recall is *higher* than `recall²`, making
the linking gap larger still.

Implied counts at the best arm (from `J` and `recall`, GT = 128,883 edges):

| | approx |
|---|---|
| edge TP | ~91,200 |
| edge FN | ~37,700 |
| edge FP | ~42,300 |
| edge precision | **~0.68** |

**Roughly 42,000 wrong links.** Recon §7 said "at most ~0.015 of edge Jaccard exists in the
entire linking problem". That measurement was taken with **perfect detections** — 133k GT
nodes, ~6.6 per frame, median nearest-neighbour spacing 25 µm. Nearest-neighbour is
trivially correct when the next-nearest cell is 25 µm away.

Real detection changes the problem completely: **~210 detections per frame** in the same
volume, true spacing ~8 µm, and 15 % of true successors missing entirely. When a cell's
real successor was not detected, a 9 µm search radius does not decline to link — it links
to a neighbour, converting a false negative into a false negative *and* a false positive.
Recon §5's "nearest neighbour is right 99.80 % of the time" is a fact about the *annotated
subset in isolation*, not about linking inside a dense detection field.

**Recon §7's conclusion should be read as: given perfect detections, linking is solved. We
do not have perfect detections, and at 84 % recall the linker is losing 12× more than the
0.015 recon said was on the table.**

---

## What to do next, in order

1. **`budget_fill=None`.** Measured, +0.0226, free. Done — it is the new default.
2. **Sweep `link_radius_um` downward** (9.0 → 7, 5, 4, 3). We are FP-heavy (~42k FP vs
   ~38k FN) and the radius is set from a p99 measured on the sparse annotated set, not on
   the dense detection field. Tightening trades FP for FN, and the metric's own rule says
   link when `p > J/(1+J)` — at `J = 0.53` that bar is **0.35**, so a candidate must be
   right about a third of the time to be worth linking. At 9 µm in a field with 8 µm
   spacing, many are not.
3. **Sweep `min_separation_um`** (6.0 → 4.5, 3.5, 2.5) to attack the 15.5 % of GT nodes the
   threshold can no longer reach. Interacts with (2) — a denser detection field makes
   linking harder — so run them as a small grid, not two independent sweeps.
4. **Score the four test datasets locally.** We hold their ground truth. That predicts the
   leaderboard before spending a submission, and comparing the two settles whether the
   visible test set is the scored set.
5. **Report the leak to the organisers.**

Not worth doing yet: finer `downsample` (XY localisation error at (1,4,4) is ~0.8 µm
against a 7 µm match radius — not the bottleneck), divisions (recon §6), any learned model
(the classical linker has 0.18 of unclaimed Jaccard sitting in it).
