# Rule-based is surprisingly strong? (currently 7th/344teams / gold zone, no learning)

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716952
- **Topic id**: 716952
- **Author**: ISAKA Tsuyoshi (MASTER)
- **Posted**: 2026-07-01T08:50:08.239056100Z
- **Votes**: 47
- **Comments**: 7

---

## Opening post

I ran a few experiments locally, submitted several, and tracked CV vs LB. Bottom line: **without any modeling (learning), a pure rule-based pipeline reached the gold zone (7th/344teams, LB 0.826 at time of writing)**. Sharing a short table and the notebook.

https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline

**Validation strategy (short)**
- The metric is essentially **edge Jaccard** (7 µm point matching → edge agreement). A naive baseline showed **CV ≈ LB**, so I fixed that as the target.
- Train has **only 2 embryos** (`44b6`/`6bba`); learned models are hard to validate for generalization, so I went **rule-based** first (DoG-blob detection + Hungarian linking).
- **Submit only on a new CV best**, adding a (CV, LB) point to a scatter each time to monitor the proxy.

**Results (all rule-based, no learning)**

| # | Method | CV(edge) | LB(public) | Notebook |
|---|--------|:--------:|:----------:|----------|
| 1 | DoG-blob detection + Hungarian linking | 0.682 | 0.663 | [v1](https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline?scriptVersionId=331719029) |
| 2 | Tuned DoG scales + 8 µm linking | 0.791 | 0.786 | [v2](https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline?scriptVersionId=331729770) |
| 3 | + gap closing | 0.807 | 0.784 | [v3](https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline?scriptVersionId=331733252) |
| 4 | + division edges | 0.810 | 0.778 | [v4](https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline?scriptVersionId=331746701) |
| 5 | **multi-scale DoG** | 0.824 | **0.826** | [v5](https://www.kaggle.com/code/isakatsuyoshi/biohub-rule-based-baseline?scriptVersionId=331751321) |

**Detection was the biggest lever.** Just taking DoG at multiple scales and using the scale-space max jumped **LB from 0.786 → 0.826 (+0.040)**. In contrast, **division edges hurt** (0.784 → 0.778) and gap closing was roughly neutral.

**CV-LB plot** (attached): detection improvements track **CV ≈ LB (~1:1)**. Graph-level add-ons (like divisions) can raise CV yet lower LB — **redrawing the scatter each submission is how you catch that**.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F5927283%2F34dc10e74c0ce84bb947c72ba8cbb719%2Fcv_lb_correlation.png?generation=1782895694211724&alt=media)

**Takeaway**: with careful validation, a **rule-based** pipeline can reach the gold zone here — no deep model required. I also tried a learned 3D U-Net, but with only 2 embryos it didn't generalize, so I set it aside. Notebook is public — feedback welcome!

---

## Comments (7)


### Timmy Juicehouse (EXPERT) — 2026-07-01T09:10:37.060Z — 3 votes

Thanks for your feedback!

It looks like the correlation between CV and LB isn't actually very good. The 3D-UNet method is very time-consuming to train, while my own heuristic strategy is quite fast. I'm waiting for biohub's confirmation before considering my next submission plan.

### Adarsh (MASTER) — 2026-07-01T12:29:36.690Z — 4 votes

Its too early but yeah, rule-based DoG is surprisingly good compared to DL approaches right now.

#### ↳ Tom (MASTER) — 2026-07-01T16:18:30.327Z — 3 votes

> I think a lot of stuffs can play around with those rules. Might need to reformulate the problem statement. I believe a certain learning signal can make deep learning model powerful.

### Timmy Juicehouse (EXPERT) — 2026-07-03T10:53:01.280Z — 2 votes

**Let me add to my results:**

| **baseline** | CV(all train dataset using official metric) | lb|
| ---- | ---- | ---- |
| my heuristic 1 (~2-hour inferece)| 0.7448   | 0.834 |
| my heuristic 2(~3-hour inferece)| 0.8213   | 0.846 |

### Tom (MASTER) — 2026-07-01T09:44:45.717Z — 2 votes

Hi, may I know your division jacard CV?

#### ↳ ISAKA Tsuyoshi (MASTER) — 2026-07-01T09:57:32.680Z — 2 votes

> Thanks! My division Jaccard CV is basically 0, because my best submissions don't predict divisions at all.

### BeanTard (CONTRIBUTOR) — 2026-07-02T20:43:37.407Z

doesnt seem to be very calibrated, but structural priors are usually very good, the idea is to have these structural priors in the model and have the model learn off of them and get more.
