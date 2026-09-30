# 89th place: division reselection transferred to Private, our Public-picked chass

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744490
- **Topic id**: 744490
- **Author**: Takahiro Katsumata M.D. (CONTRIBUTOR)
- **Posted**: 2026-09-30T01:36:08.747989600Z
- **Votes**: 1
- **Comments**: 0

---

## Opening post

Team: Takahiro Katsumata (tkatsuma)

## 1. TL;DR

- Private 89th / 4,020 (0.928, silver; preliminary leaderboard). Final picks NLP-30 and NLP-40: Public 0.967 / 0.966, Private 0.928 / 0.927.
- We built on the public `thtennant/biohub-frontier947-readmit-v1` chassis (0.946) and added a coordinate head (+0.007 on both Public and Private), a learned division reselection (DIV-C2; +0.010 Public / +0.008 Private at the break-even τ 0.18, +0.013 / +0.007 at the final τ 0.30) and two small post-processing steps (about +0.001 each on Private, +0.001 together).
- Two things did not transfer: the division threshold we tuned on Public (flat on Private), and our choice of chassis. A chassis with our own third detector seed looked 0.004 lower on Public and was 0.007 higher on Private. Our best Private was 0.935 (C2-18s, Public 0.959), about 42nd–48th; the gold line was 0.941.
- Lesson: with about 12 Public videos, a Public gap of one or two divisions should not pick the chassis. Keep one final slot for a different chassis.

## 2. Starting point

Every final kernel is a patched copy of `thtennant/biohub-frontier947-readmit-v1`. It detects cells with two temporal 3D U-Net seeds (8-view TTA), scores edges with an edge transformer (edge-feature TTA, harmonic forward/reverse-time fusion) and links them with an ILP. A DeepCenter centre prior vetoes gap closing and divisions. Repairs follow: flow-based motion re-link, re-admission of discarded detections and gap filling. A geometric safe-division rule, a short-track filter and linefit smoothing (±2 frames) finish the output.

We reached this base through byte-exact reproductions of public notebooks, each submitted once against a pre-registered expected score (10 of 12 matched). Of our Public climb from 0.812 to 0.967, +0.132 came from reproducing public work and +0.023 from our own changes (third seed +0.002, coordinate head +0.007, division reselection +0.013, pruning +0.001). The public share is net of a −0.003 step when we moved to the readmit base, which dropped our third seed.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F380570%2F2ad4314fb01b2b9c6b9caccb655b7bdf%2Fpipeline_schematic.png?generation=1790741898529607&alt=media)

*Pipeline of our final submissions. Grey: public chassis; green: our additions, with the Public / Private change against the parent submission; orange dashed: our third U-Net seed, not in the final picks.*

## 3. What we added

| Submission | Change | Public | Private |
| --- | --- | --- | --- |
| Zero control | Public chassis, byte-identical output | 0.946 | 0.913 |
| CR-1R | + our coordinate head | 0.953 | 0.920 |
| C2-18 / C2-30 | + DIV-C2 at τ 0.18 / 0.30 | 0.963 / 0.966 | 0.928 / 0.927 |
| PRN-1 / LZ2 | C2-30 + 1% pruning / + z window 3 | 0.967 / 0.966 | 0.928 / 0.928 |
| NLP-30 / NLP-40 (final) | C2-30 / C2-40 + LZ2 + PRN 1% | 0.967 / 0.966 | 0.928 / 0.927 |

### 3.1 Coordinate head (CR-1R)

A small MLP moves each detection by up to 2 µm. Its input is seed-A U-Net features at the detection plus 6-neighbour differences (224 values → Linear 32 → SiLU → Linear 3). We trained it on 48 train videos with the `anvithpothula/biohub-x138` recipe. Centre error fell by 20.7% in video-grouped CV, and by 3.5% and 18.5% leave-one-embryo-out. Through the edge-scorer features, it also changes what the ILP keeps.

### 3.2 DIV-C2: division reselection

**Why divisions.** The division term has weight 0.1, but the hidden set rewards it: removing every division from the 0.926 public notebook cost −0.022 on Public against −0.0067 locally. Simply loosening the safe-division gates added annotated false positives 2–10× faster than true positives, so we learned which candidates to keep.

**Candidates.** The chassis's gates without the symmetry condition, and with divergence 1.0 µm instead of 2.25 µm. A candidate is a parent at t, its current child at t+1 and an orphan at t+1. The original rule's picks are always included.

**Model (M-A).** An L2 logistic regression on 18 standardised features: geometry (parent, sister and child distances, symmetry, sister separation growth at t+2 and t+3), track lengths and nearby track ends, ILP probabilities, the DeepCenter score and local density. It was trained on 157 labelled candidates (31 positive) from 40 train videos.

**Labels from the official metric.** For each candidate we replay the rest of the pipeline with and without it and score both with the official division scorer. A TP gain is positive, an FP gain without a TP gain is negative, and anything else is left unlabelled. This caught divisions predicted one frame early (a TP for the metric), which had been 20–30% of positives and were mislabelled.

**Selection and threshold.** We keep candidates with p ≥ τ, highest first, each parent and orphan used once. Adding a candidate that is right with probability p changes micro division J by (1/D)·[p − (1−p)·J], with D = TP + FP + FN, so the break-even is p* = J/(1+J). Train J = 0.214 gives τ ≈ 0.18. We then swept τ on Public:

| τ | 0.10 | 0.18 | 0.30 | 0.40 |
| --- | --- | --- | --- | --- |
| Public | 0.957 | 0.963 | 0.966 | 0.966 |
| Private | 0.927 | 0.928 | 0.927 | 0.926 |

Public rewarded a higher τ; Private did not. The reselection itself was real (+0.008 on Private on this chassis, +0.006 on the third-seed chassis in §6), but the τ tuning only fitted Public. On 155 extra train videos that M-A never saw, the production model gained +0.0053 (video average). On our 40 local videos, an oracle keeping only true candidates would add +0.0174 in total, a further +0.0142 over C2-30; at τ 0.30, M-A captured 18% of the oracle's total gain.

### 3.3 PRN: pruning nodes unlikely to be annotated

Adjusted edge J = J·(1 − 0.1·(N_pred − N_total)/N_total), and the bonus for fewer nodes has no cap. Only about 3% of predicted nodes match an annotated GT node, so removing nodes that are unlikely to be annotated costs little. A gradient-boosted classifier on 57 test-time features ranks the nodes. The features cover position, density, edge probabilities, which repair step made each edge, and component statistics. We remove the lowest 1% per video, typically short weak components at the lateral border.

Cross-embryo AUC was only 0.55–0.69, and the two train embryos are annotated differently (clustered versus scattered lineages). Beyond 1%, one embryo turned negative in leave-one-embryo-out, so we stopped there. PRN exploits the node-count term: it breaks no rule, but a metric fix would remove it.

### 3.4 LZ2: z-only smoothing window

LZ2 widens the linefit window to ±3 frames for z only; x and y stay at ±2. It gained +0.0019 micro locally, and Public did not change.

## 4. Metric and data notes

- **Micro averaging.** Division J pools TP/FP/FN over videos, and edge J weights each video by its TP+FP+FN. Video-average numbers misled us more than once. With a pooled division J near 0.2, losing one TP costs about 5× what removing one FP gains.
- **Public is about 12 videos.** The hidden test has 40 videos; Public is about 29% of it, so roughly 12 Public and 28 Private. One division TP is worth about +0.003 or more on Public. A bootstrap gave the reselection effect an SD of 0.0049 on 12 videos and 0.0021 on 28. We read 0.001 on Public as noise, yet still let gaps of 0.002–0.006 pick our chassis (§6).
- **Local evaluation is in-sample.** The seed-B detector's training metadata lists all 199 train videos, and Eric (@chengren01) reported the same. Up to 25 Sep, seven of our changes (post-processing, linking, blending and one early division rule) gained ≥ +0.005 locally, and none gained ≥ +0.002 on Public. Division reselection was the exception: C2-18 went from +0.0047 locally to +0.010 on Public.
- **Out-of-sample check.** 155 extra train videos were never used to fit M-A or to set our rules (the public detectors are in-sample on them, and our coordinate head was trained on 38 of them). That check made us drop NSD-1 (no second division inside a daughter's track; −0.0015 micro there) and keep GX (mutual-NN gate off) as an option only. On Private, each would have added +0.001 to +0.002: within noise, but in the wrong direction for us.

## 5. What did not work

| Idea | Result | Why (our reading) |
| --- | --- | --- |
| U-Net point features for divisions (M-C1, E-U, MC2A) | Public 0.958–0.964, vs 0.963–0.966 for M-A | In-sample features: AUC gain +0.068 in-sample, +0.006 on unseen videos |
| M-A retrained on 195 videos (DIV-C3) | −0.0042 locally; not submitted | Extra videos are 6% positive vs 20%; probabilities fell |
| Detection union + detection threshold 0.965 → 0.960 (from `amanatar/optimized-biohub-max-score`) | −0.0032 locally; not submitted | Nodes +3.8%, 98% far from annotated cells |
| Looser linking geometry (same notebook) | −0.0374 locally | The z gate widened to 17.5 µm, adding wrong z links in the dense embryo |
| Global flow stabilisation (DC-1, DC-2) | Public: DC-1 0.939 (−0.010 vs its parent); DC-2 0.817 / 0.777 | The flow transform was applied only at inference, to scorers trained on raw geometry |
| 5-seed head averaging, two input sets (Optuna-tuned heads failed our pre-submission check, so none was submitted) | Public 0.950 / 0.951, vs 0.952–0.953 for single-seed heads | Offline head ranking did not transfer |
| READMIT 0.965 → 0.94 | −0.00056 micro locally; not submitted | Final nodes +1.2%, none matching GT |
| Density-adaptive relink; group-flow or quadratic smoothing | Analysis only | Motion does not depend on density; group flow adds nothing beyond track smoothing; GT z acceleration is too small for a quadratic fit to help |

## 6. Final selection and what Private showed

The rule was fixed before the last scores came in. Slot 1 was NLP-30, to be swapped only for a variant at least 0.002 higher on Public. Slot 2 was NLP-40, built on C2-40, whose divisions are a subset of C2-30's. We added no hedge on a different chassis, because once the coordinate head was on, those variants had been 0.002–0.006 lower on Public every time. That was the costly decision.

| Submission | Public | Private |
| --- | --- | --- |
| CR-1R without head (2 detector seeds + readmit) | 0.946 | 0.913 |
| + our third seed, no head (3S16) | 0.948 | 0.922 |
| CR-1R (+ coordinate head) | 0.953 | 0.920 |
| CR-3S16 H2 (third seed + coordinate head) | 0.951 | 0.929 |
| C2-18 (CR-1R + reselection, τ 0.18) | 0.963 | 0.928 |
| **C2-18s** (CR-3S16 H2 + reselection, τ 0.18) | 0.959 | **0.935** |
| C2-30s (CR-3S16 H2 + reselection, τ 0.30) | 0.960 | 0.934 |
| NLP-30 / NLP-40 (**final picks**) | 0.967 / 0.966 | **0.928 / 0.927** |
| NLP-30+GX / COMBO-1 (NLP-30 + NSD-1) | 0.966 / 0.967 | 0.930 / 0.929 |

- **Third detector seed:** +0.009 on Private with or without the head (Public +0.002 / −0.002). With reselection on top, it was +0.007 on Private and −0.004 on Public. Local evaluation could not referee this, because train is in-sample for every seed.
- Across our 20 submissions with Public ≥ 0.955 (all from 25–29 Sep), the Public–Private rank correlation was 0.06 (Spearman).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F380570%2F90d19ca759de38473bfd46a6242ba894%2Fpublic_private_scatter.png?generation=1790741919163733&alt=media)

*Public vs preliminary Private for our 43 submissions with Public ≥ 0.934, on the same scale for both axes. Among the 20 with Public ≥ 0.955, Public did not predict the Private order (Spearman 0.06). C2-18s, on the third-seed chassis, was below the final picks on Public and above them on Private, but still 0.006 short of the gold line.*

- **Lesson:** build an out-of-sample yardstick for chassis choices early, and keep one final slot for the best submission on a different chassis.

## 7. Process

- **Pre-registered cards.** Before each submission, a committed card fixed the change, the local safety conditions and how to read the Public score. Waivers and variants chosen after seeing results (such as τ 0.40) were logged.
- **Visible-run guards.** Stages before the change had to be byte-identical to the parent, activation markers present, no fallback, runtime within the card's limit (usually ≤ 1.10×) and a valid CSV.
- **AI agents.** Anthropic Claude handled design, audits and some implementation, and OpenAI Codex most of the implementation. A human authorised every submission and chose the final slots.

## 8. Acknowledgements

- **Hosts:** thanks to the Biohub team (Thibaut Goldsborough, Jordão Bragantini, Xiang Zhao, Gordon Leary, Teun Huijben, Ilan da Silva Theodoro, Kyle Harrington, Chi-Li Chiu and Loïc A. Royer) and to Kaggle (Walter Reade, María Cruz) for the data and the carefully documented metric.
- **Datasets and notebooks:** thanks to pilkwang for the datasets our pipeline runs on (seed-A and seed-B U-Nets, edge transformer, DeepCenter) and for the two-seeds logit-blend notebook we reproduced.
- **Notebooks:** thanks to takumashiga, thtennant, anvithpothula, redoctopusk, flexonafft, sjlee101, rockerritesh, evgendvorkin, nusrati, stephennedumpally, alioman, qrz1201 and analyticaobscura for the public notebooks we reproduced and built on, and to amanatar for the variants we tested.
- **Discussion:** posts shaped how we read local-versus-Public transfer, the metric and the split. Thanks to Eric (@chengren01), imissher (@tule12345), hikaggler (@hikarukai), Justin CH123 (@justinch123), Quantizr (@quantizr), hengck23 (@hengck23) and Georgy Mamarin (@georgymamarin), and to John Taylor (AI) (@johntaylorai) for discussion 743929 (the READMIT 0.94 tip).

---

## Comments (0)

*(none)*
