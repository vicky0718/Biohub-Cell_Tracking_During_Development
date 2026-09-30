# Eight leaderboard experiments on the public 0.947 stack — all negative, here are the numbers

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743222
- **Topic id**: 743222
- **Author**: imissher (CONTRIBUTOR)
- **Posted**: 2026-09-25T08:59:22.308897500Z
- **Votes**: 1
- **Comments**: 0

---

## Opening post

We tried for two weeks to beat the public dual-seed notebook (UNet + node transformer + ILP) and never did. Every change we tested scored at or below the unmodified notebook. Posting the numbers so nobody else spends the same two weeks.

## Local scores did not predict the leaderboard

The public secondary checkpoint (`seed314159`) was trained on all 199 train videos — its own `split_manifest.json` says so. Any hold-out you take from train is in-sample for the detector behind 80% of the detection logits.

Same predictions, official metric, paired:

| Change | Local (19 videos) | Public LB |
|---|---|---|
| Motion relink off | +0.017 | −0.002 |
| ILP candidate threshold 0.48 → 0.30 | +0.009 | 0.000 |

Four local decisions, four leaderboard misses, one with a flipped sign. @Eric found the same thing independently, and @hikaggler measured a local-to-LB correlation near zero in this score band.

## Every single-variable change we submitted

| Configuration | Public LB |
|---|---|
| v30, unmodified | **0.948** |
| v29, unmodified | 0.946 |
| Motion relink off | 0.946 |
| Z-flip TTA added (8 → 16 views) | 0.946 |
| Detection threshold 0.960 | 0.946 |
| Detection threshold 0.970 | 0.945 |
| Secondary detection weight 0.80 → 0.65 | 0.945 |
| Our own model in the secondary slot | 0.939 |

Both threshold directions lose, so 0.965 is already at the optimum. Extra TTA views buy nothing. Every change that trusts the learned association more (relink off, lower candidate threshold) failed, while the geometric repair always helped — which fits test being a different embryo.

## A trap inside the notebook worth knowing about

The notebook scores 8 train videos at the end, sweeps 7 post-processing candidates, and rewrites `submission.csv` with the winner. Two problems:

1. Those 8 videos are the ones the secondary model memorised.
2. The list excludes train stems that also appear in `TEST_DIR`. On your commit run that drops the 4 example videos. On the hidden rerun there is no overlap, so **a different 8 videos are scored and a different config can win**. The setup that produced your leaderboard score may not be the one printed in your log.

We only found this after pinning the selected config by hand and losing 0.002.

## Training our own model

We started from the public secondary checkpoint and adapted it for 6 hours at lr 1e-5 on 600 synthetic sequences (CC0 set by @José Freitas) mixed with 40 real videos. Swapped in: **0.939**, about 0.006 worse than the public weights.

## If you plan to train on Kaggle GPUs

On 2× T4 with the support-pack trainer, 64³ windows: batch 16 runs out of memory, and before it died it ran at **12–14 seconds per step**, about 35–40 minutes per epoch. An 8-hour session buys 12–25 epochs. The public primary had 402. That gap is not closable on Kaggle hardware.

## What we would do differently

Stop tuning inference and measure properly: train on one embryo, evaluate on the other. It is the only split in this dataset that is not memorised.

Happy to answer questions about any number here.

---

## Comments (0)

*(none)*
