# Does CV match LB in this competition?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160
- **Topic id**: 730160
- **Author**: Chester Yuan (EXPERT)
- **Posted**: 2026-07-28T09:30:03.094942400Z
- **Votes**: 5
- **Comments**: 5

---

## Opening post

Does CV match LB in this competition?

---

## Comments (5)


### Mendrika Ramarlina (MASTER) — 2026-07-28T18:48:20.387Z — 1 votes

Directionally, yes, but the public LB is more optimistic by almost 10%. That being said, movie-to-movie variability is large: ±0.14, with 18% coefficient of variation. Worst movie scores 0.460 and the best is at 0.984. The public split might have more of the easy movies compared to my hold out set.

#### ↳ william.wu (MASTER) — 2026-07-29T08:53:08.440Z

> What's your CV strategy? Leave one embryo out?

#### ↳ ↳ Mendrika Ramarlina (MASTER) — 2026-08-02T09:17:32.777Z — 2 votes

> > Leave one embryo out for model selection

### Adarsh (MASTER) — 2026-07-29T05:01:03.153Z

if ur using any of the baseline notebooks and creating a cv strategy on top, I guess not because from my understanding, most of the baseline notebooks use the model trained on all available training data, leading to leakage.

My suggestion would be to use two separate models : one trained on only one embryo, the other used for validation and the other model is trained on both embryo's which would be used for submission.

#### ↳ Mark_RowSet (CONTRIBUTOR) — 2026-08-03T04:13:21.763Z — 2 votes

> Adarsh is right and I can put numbers on it.
> 
> The public weights were trained on all 199 annotated videos. Their split_manifest.json lists 199 under train, and the 40 in its own test list are all inside that same set. So if you're validating on train videos with those checkpoints, you're scoring the model on data it memorized.
> 
> This cost me a week. I ablated the post-processing in the public pipeline and measured +0.0184 from turning one stage off. Baseline guard reproduced the shelf score exactly, I verified the config actually applied, the whole thing looked clean. Submitted it and got 0.909 against my 0.912 baseline. The sign flipped.
> 
> Took me a while to figure out why it inverts instead of just shrinking. Those stages are corrective, they exist to repair model errors. On memorized videos there isn't much left to repair, so they only perturb predictions that were already right, and switching them off looks like a gain. On unseen embryos the base predictions are worse and the same stage earns its keep. A training-set harness will keep telling you to delete the components that matter most at test time.
> 
> Leave-one-embryo-out is the right call, though as Adarsh says it means training your own model, since the released checkpoints have seen everything.
