# How are your local CVs like? It feels like a huge gap, atleast for me.

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739352
- **Topic id**: 739352
- **Author**: Gunjan Haldar (EXPERT)
- **Posted**: 2026-09-03T18:36:29.896847600Z
- **Votes**: 3
- **Comments**: 6

---

## Opening post

Can we expect a huge shakeup? if so, what are your thoughts/opinions.

---

## Comments (6)


### Pavel (MASTER) — 2026-09-04T10:33:46.103Z — 2 votes

I have an edge weight model which gives boost around 0.02-0.04 on each of 5 folds, with similar scores hold up when trained on one embryo and tested on another.

It gives approx +0.000 on LB though.

### mikelou1 (EXPERT) — 2026-09-04T00:02:43.187Z — 1 votes

I have like 0.96 CV and 0.93 LB for some reason

#### ↳ hengck23 (GRANDMASTER) — 2026-09-04T00:56:03.757Z — 7 votes

> it is not useful to state/anayse " 0.96 CV "
> you should break it down:
> ```
> edge_jaccard = ???
> adjusted_edge_jaccard = ???
> 
> node recall = ...
> estimated no of nodes = ...
> division = ...
> 
> breaks in tracks ....
> link to wrong trajectory ....
> 
> ```
> 
> if you do this you will know if there is overfitting etc

#### ↳ Kim Jinseo (CONTRIBUTOR) — 2026-09-05T05:38:43.977Z

> How did you evaluate your CV? Wouldn't evaluating CV using the data we trained with be prone to overfitting?

### sghwr (CONTRIBUTOR) — 2026-09-06T03:12:03.350Z

Somehow, I observed a CV‑LB shift in my exps; nearly all my local CV scores are more pessimistic than the public LB by about ~0.013‑0.02 (・∀・*). (Edge Jaccard only; haven’t specifically developed division J yet.)

### unknown — 2026-09-06T03:09:50.730Z

*(empty)*
