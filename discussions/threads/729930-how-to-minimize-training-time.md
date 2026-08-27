# How to minimize training time

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/729930
- **Topic id**: 729930
- **Author**: Rajmukund mehta (CONTRIBUTOR)
- **Posted**: 2026-07-27T09:29:37.742488700Z
- **Votes**: 3
- **Comments**: 6

---

## Opening post

Hello,so I am new to handling image dataset .my question is how do you guys train model on such large datasets multiple times as it takes a lot time just for 50 epoch.is there any trick or something people use that i should know.

---

## Comments (6)


### Davit Khantadze (CONTRIBUTOR) — 2026-08-02T12:47:25.233Z

could you share training time per epoch? For me, after introducing some tricks, it takes 2 hours currently.

### mikelou1 (EXPERT) — 2026-07-28T08:31:28.733Z

There isn't really a trick to make training quicker. I suppose you could train it in FP16/BF16 but that reduces quality. I just train it on my gpus and it takes ~4 hours for 50 epochs.

#### ↳ Davit Khantadze (CONTRIBUTOR) — 2026-08-02T13:07:46.997Z

> wanted to check: you train baseline model 50 epochs in 4 hours? For me, after introducing mixed precision, 1 epoch takes around 2 hours. Not sure that the difference can be explained by GPU alone.

#### ↳ ↳ mikelou1 (EXPERT) — 2026-08-02T13:38:16.137Z

> > What GPU do you train it on?

#### ↳ ↳ Davit Khantadze (CONTRIBUTOR) — 2026-08-02T13:51:24.410Z

> > T4, from kaggle. Does this tell you something?

#### ↳ ↳ mikelou1 (EXPERT) — 2026-08-03T16:09:45.540Z

> > Yeah that explains it. T4 has very bad fp32 [its basically 25x slower]. Try fp16 or something
