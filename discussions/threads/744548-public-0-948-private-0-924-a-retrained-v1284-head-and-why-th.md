# Public 0.948 → private 0.924: a retrained V1284 head, and why the public board pointed the wrong way

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744548
- **Topic id**: 744548
- **Author**: Korokke3 (CONTRIBUTOR)
- **Posted**: 2026-09-30T08:25:28.070876400Z
- **Votes**: 1
- **Comments**: 0

---

## Opening post

Now that private is out, here is a result that surprised me. It may help anyone thinking about the public/private gap.

**Setup.** I started from the public x138 notebook (Harmonic Fusion + frontier947 post-processing + anvithpothula's V1284 coordinate-refinement head). The V1284 head is a small MLP. It takes frozen detector features at each detection centre and predicts a sub-voxel shift (at most 2 µm) toward the true centroid. The published head was trained on 20 movies. I ran the pipeline on all 199 training movies in a capture mode that matches detections to the sparse GT, which gave 128,931 pairs. I then retrained the same head (3 seeds, 5 epochs).

**What the boards said** (x138 pipeline; only the refinement head changes, except in the last row):

| Refinement | Public | Private |
|---|---|---|
| original head (x138 as published) | 0.953 | 0.917 |
| retrained heads only | **0.948** | **0.924** |
| 0.5 original + 0.5 retrained | 0.954 | 0.923 |
| 0.85 original + 0.15 retrained | 0.956 | 0.919 |
| same + ILP_DIVISION_WEIGHT 0.4 + READMIT_MIN_SCORE 0.94 | 0.958 | 0.918 |

The public board ranked these configurations almost exactly in reverse. My worst public submission was my best private one, 0.007 above x138.

**What would have caught it.** Leave-one-embryo-out CV on the training data (train on 44b6 and test on 6bba, then the reverse) preferred the 50/50 blend, which scored 0.923 on private. I overrode it with public-board tuning and picked the public-best pair, so I missed bronze by 0.001. My takeaway is that when the hidden test is new embryos and public is 29%, public differences of 0.001-0.003 are mostly the public embryo's quirks. At least one final pick should come from group-CV.

A small physics note: the gain comes largely from absolute position accuracy, including per-embryo offsets (the mean GT-minus-detection z-offset is +0.17 µm for 44b6 and +0.84 µm for 6bba), not only from frame-to-frame jitter. A head trained on de-meaned targets scored 0.001-0.003 worse on both public and private.

- Notebook (runs as a drop-in on the x138 pipeline): https://www.kaggle.com/code/korokke3/biohub-retrained-v1284-head
- Heads plus capture and training scripts (CC0): https://www.kaggle.com/datasets/korokke3/biohub-v1284-head-199-movies

Thanks to anvithpothula, pilkwang, flexonafft and thtennant, whose public work this builds on.

---

## Comments (0)

*(none)*
