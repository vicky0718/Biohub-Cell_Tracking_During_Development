# 213th Place: Two Coordinate Heads, One Better Step

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744562
- **Topic id**: 744562
- **Author**: Hammad Farooq (CONTRIBUTOR)
- **Posted**: 2026-09-30T09:51:06.656980400Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

## TL;DR

We finished **213th on the private leaderboard (0.923 private / 0.957 public, +40 places on the shake-up)** with a single-model pipeline built on the strongest public lineage. We did not train a new detector or linker. All of our gain came from one component: **the coordinate head that moves each detected peak before linking.** We ensembled two independently trained heads, then rescaled the averaged shift so it keeps the step length a single head would take.

| Submission | Public | Private |
|---|---|---|
| Public x138 lineage (V1284 head) | 0.953 | – |
| + our own B251 head (replaces V1284) | 0.954 | 0.922 |
| + head ensemble 0.7·B251 + 0.3·V1284 | 0.956 | 0.921 |
| **+ ensembled shift ×1.15 (final)** | **0.957** | **0.923** |

We also found a cheap trap that cost us a slot: writing float centroids drops the score even when the graph is identical. That one is at the end of this writeup.

---

## 1. Pipeline

Our base is the public 0.953 lineage: pilkwang's temporal UNet3D + edge transformer, the Harmonic Fusion post-processing, and the x138 additions (seed-mode flow relink, re-admission of discarded detections, and gap filling from sub-threshold peaks). We kept every public knob frozen, so any score change comes from our own edits.

![Pipeline overview](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F36060323%2Fe3cbeb3c8353d081e6e05b9a24734ed6%2Ffig1_pipeline.png?generation=1790761412444468&alt=media)

Key stages, in order:

- **Detection.** Two temporal UNet3D seeds (primary + seed314159) with the pipeline's xy-flip TTA. Adding more views (8-view flips or full D4) cost about 0.008 in our probes. Peaks come from harmonic fusion of the two seeds' logits at threshold 0.965.
- **Coordinate head.** A small MLP moves each integer peak by up to 2 µm, *before* linking. This is where all of our work went (section 2).
- **Association.** An edge transformer with 8-view edge-feature TTA and bidirectional (reverse-time) fusion at weight 0.15, then an ILP in SCIP (division weight 1.2, disappearance weight 2, 1200 s timeout per film).
- **Repair.** A tissue-flow motion relink (K = 12 neighbours within 40 µm) re-attaches broken tracks. Discarded detections are re-admitted, and sub-threshold peaks (> 0.3) fill short gaps.
- **Divisions.** A safe-division step with a DeepCenter veto, parent ≤ 9 µm, sister ≤ 14 µm and sister symmetry τ = 0.6, capped per frame and globally.
- **Output.** Short tracks (< 6 frames) are filtered, centres are clamped inside the volume, and coordinates are written as `int(round(·))`.

Runtime on 2×T4 is about 14 minutes on the visible test and well inside the 12 h limit on the hidden set. A 7.5 h repair deadline degrades gracefully if needed.

---

## 2. The coordinate head, and why two are better than one

The detector gives integer voxel peaks, and z is 1.625 µm per voxel. A coordinate head regresses a sub-voxel correction from frozen features at the peak:

- **Inputs (B251 head, 251 dims):** 32 UNet feature channels at the peak, plus 6 directional differences to its ±z/±y/±x neighbours (224 dims). Added to these is a 3×3×3 patch of the blended detector logits, expressed relative to the centre logit (27 dims).
- **Model:** `Linear(251→32) → SiLU → Linear(32→3)`, with a bounded output `d = 2h / (1 + |h|)`, so |d| ≤ 2 µm.
- **Training:** detection-to-GT offsets from 45 training movies.

The public V1284 head has the same shape without the logit patch (224 dims) and was trained on 20 movies. The important detail is **where** the shift is applied. It is added to the peak *before* the edge transformer and the ILP, so it changes which detections get linked, not just where the written centre lands. Applying the same head only to the output centres (graph unchanged) scored 0.946 versus 0.951 when applied before linking.

![Coordinate-head ensemble](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F36060323%2F2c44806b80535e156eabc2582b5e4710%2Ffig2_head.png?generation=1790761518580532&alt=media)

**Ensembling.** The two heads use different inputs and different training movies, so their errors are only partly correlated. A convex blend `0.7·B251 + 0.3·V1284` beat either head alone on the public LB.

**Rescaling.** When two heads disagree in direction, their average is *shorter* than either shift. The blend therefore systematically under-corrects. Multiplying the blended shift by 1.15 restores the step length, and it was our best submission on **both** splits.

![Head-ensemble weight sweep](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F36060323%2Fa6a662d6941a4fb2e6a2f5863ec1f225%2Ffig3_sweep.png?generation=1790761557320944&alt=media)

![Shift-scale sweep](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F36060323%2F0959952fb48e9f21b27122303707b581%2Ffig4_scale.png?generation=1790761608714252&alt=media)

Two honest notes from these sweeps:

1. **Public gains mostly did not transfer.** The public LB (29%) peaked at w = 0.65–0.7, but the private split is flat within 0.002 across all weights, and B251 alone was already 0.922 private. Only the ×1.15 rescale improved private as well (0.923).
2. **The step length changes the graph, not just positions.** On film `44b6_0b24845f` the node count after linking falls monotonically as the shift grows: +4.6% at ×0.85, +1.8% at ×1.0, −0.2% at ×1.15, all relative to B251. The adjusted edge Jaccard penalises over-predicting nodes, which is consistent with the longer step helping.

---

## 3. A trap worth knowing: float centroids

The reference metric (`royerlab/kaggle-cell-tracking-competition`, `csv_to_geffs`) casts `z, y, x` to `Float64`. With a sub-voxel head, writing the float centroid looks like a free win, so we tried it.

We changed exactly one line, `round(v, 3)` instead of `int(round(v))`. We verified that node ids, edges and `t` were byte-identical, and that rounding the float output reproduces the int file for 99.95% of nodes.

![Float-centroid trap](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F36060323%2F95a1a9656273a796e161bc877d71957b%2Ffig5_float.png?generation=1790761652484734&alt=media)

It scored without error and lost **0.008 public / 0.007 private**. Our best guess is that the Kaggle-side scorer truncates non-integer input rather than rounding. Truncation would move every centre by about −0.5 voxel on average, roughly 0.8 µm in z. We have not confirmed the mechanism. Either way: **keep the int write.** We posted this during the competition so others would not lose a slot to it.

---

## 4. What did not work (public LB)

| Idea | Result |
|---|---|
| 8-view detection TTA (flip-only or full D4) | −0.008 |
| Conditional two-model detection fusion near the threshold | −0.013 to −0.016 |
| Motion relink tight gate 6.0 / 6.5 / 7.0 µm | ±0.000 |
| Detection threshold 0.960, secondary edge weight 0.25 | ±0.000 |
| Mitosis-CNN orphan forks for divisions | ±0.000 to −0.003 |
| Disabling post-hoc divisions entirely | −0.035 (divisions matter) |
| Float centroids | −0.008 |

The detector and linker were saturated by the public stack, and nearly every knob we touched landed within noise. Coordinate geometry was the one lever with real headroom.

---

## 5. Takeaways

- **Model the geometry, not just the graph.** At a 7 µm matching radius and 1.625 µm z-voxels, sub-voxel centre correction *before* linking was worth more than any post-processing knob.
- **Averaging vectors shrinks them.** If you ensemble regressors of a displacement, check the norm of the average and rescale it.
- **Change one thing per submission.** Every probe in our log changed exactly one factor against a frozen parent, which is the only reason we can say anything about what worked.
- **Trust the split, not the leaderboard.** Most of our public gain was noise at the third decimal; the change that held on private was the one with a mechanism behind it.

## Acknowledgements

This solution stands on public work. Thanks to **pilkwang** (temporal UNet3D, DeepCenter and support pack), **Anvith Pothula** (x138 and the public V1284 head), **Igor Zharov** and **Raunak Dey** (Harmonic Fusion), **Seung Jae Lee** (LF-DCTTA), and **Corwin** (flow-compensated relink). Thanks also to the Biohub team for a genuinely hard and well-designed benchmark.

*Team KR_no_1: KR_no_1, URAD, Mirza Yasir Abdullah Baig, Hammad Farooq.*

---

## Comments (0)

*(none)*
