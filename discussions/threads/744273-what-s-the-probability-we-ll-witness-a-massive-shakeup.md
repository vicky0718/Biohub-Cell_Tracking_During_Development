# What's the probability we'll witness a massive shakeup?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744273
- **Topic id**: 744273
- **Author**: sghwr (CONTRIBUTOR)
- **Posted**: 2026-09-29T07:39:43.911014100Z
- **Votes**: 4
- **Comments**: 7

---

## Opening post

if the division event is roughly equal with train-199 videos, then i think it's actually a very small amount and pretty weak for shakeup. the division-Jaccard term is computed from very few ground-truth events per video, so it swings a lot on just one or two forks going right or wrong — a small amount in current public lb may just be division-count luck. and there may be also human annotation issues in gt labels.

and for post-processing term, we A/B'd the same edge-only post-processing tweak twice on the public LB after it looked like a solid win offline, and both times it came back flat. Local gains on train-like data don't seem to transfer reliably. 

All things considered, I’m pretty anxious right now, as I suspect our solution has heavily overfitted to the public test set.

What do you think? All opinions are welcome!😊

---

## Comments (7)


### You WeiLin (CONTRIBUTOR) — 2026-09-29T12:24:52.043Z — 4 votes

I’m also surprised. My best-performing version locally scores poorly on the LB, while the one with a high LB score ranks dead last in my local tests.

### Cyrus (MASTER) — 2026-09-29T11:53:18.943Z — 3 votes

Since the algorithm is sequential and involves multiple steps, there is a risk of overfitting to the public test set. This risk could be even higher if the public test distribution is somehow close to a subset of the training data. Considering these factors, I think there is a fairly high probability of a leaderboard shakeup. I would not necessarily expect a major shakeup within the top 10. The bigger changes may happen around solutions based on the public code anchor, especially those that do not rely on proper cross-validation and appear to be heavily overfitted to the public leaderboard.

#### ↳ Gabriel (CONTRIBUTOR) — 2026-09-29T19:27:34.973Z — 1 votes

> I am pressimistic about this because my local testing is not good.

### Satwik (MASTER) — 2026-09-29T09:06:21.600Z — 5 votes

I believe the public LB has very few divisions. Everything that I had good local results on division (validating on a held out 41 video set, then the entire 199 set, and an external dataset gave consistent gains) but caused either no change or a slight drop in my LB scores. My guess is that most of the public LB we see is an easier embryo, probably less dense, including divisions. Take it with a grain of salt though. If you have robust divisions, especially in dense, crowded conditions, I think there won't be much of a shake up for you. Another thing to consider is the public model, which largely seems to have been tuned for the LB. It might fall apart on private LB, given the other embryo covers most of private LB. Good luck!

#### ↳ Civitasmass (EXPERT) — 2026-09-29T11:32:20.087Z — 1 votes

> same.
> I've tried numerous methods locally, but on the public LB, the gains are either negligible or even negative.

### Yassine Alouini (GRANDMASTER) — 2026-09-29T09:13:45.520Z — 4 votes

Very highly for one major reason: many submissions are forks and tweaks of existing notebooks.

### OmerZalman (CONTRIBUTOR) — 2026-09-29T22:26:20.263Z

really high, its already going to happen, pretty sure everyone from 4000-1st will move some way bc the private set is so much different according to the comp organizers
