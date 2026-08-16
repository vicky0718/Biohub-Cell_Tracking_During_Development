# Only 2 groups of embryo_id？

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793
- **Topic id**: 716793
- **Author**: Timmy Juicehouse (EXPERT)
- **Posted**: 2026-07-01T03:15:26.558908700Z
- **Votes**: 14
- **Comments**: 6

---

## Opening post

@thibautgoldsborough

Hi, biohub team. During my EDA, I found only two groups of embryo_ids. I would like to ask about the situation with the embryo_ids in the test set. If there are multiple groups of embryo_ids in the test set, how should the training results be evaluated and how did you consider the design of the training set?

---

## Comments (6)


### Thibaut Goldsborough (CONTRIBUTOR) — 2026-07-01T22:17:13.263Z — 4 votes

Hi, indeed there are two unique embryo_ids in the training set. You can assume the test sit is roughly similar in size, with no overlap in embryo_ids between train and test sets.

#### ↳ Timmy Juicehouse (EXPERT) — 2026-07-02T00:19:38.363Z — 1 votes

> So my understanding is that the test set consists of another two groups of embryos of the same number, right?

#### ↳ Tom (MASTER) — 2026-07-02T02:34:45.883Z — 1 votes

> Hi @thibautgoldsborough, the test set also consists of two groups of embryos? and the distribution is similar to train set: 71:128?

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-03T06:19:28.050Z

> > I probed the test set and found that its size is significantly smaller than the training set. Of course, it's also possible that the embryonic cell development nodes in the test set are simpler. Assuming the development process is the same, the size is roughly only 2/3 or even less.

#### ↳ ↳ Sangram Patil (EXPERT) — 2026-07-03T07:20:22.253Z — 1 votes

> > This leaderboard is calculated with approximately 29% of the test data. The final results will be based on the other 71%, so the final standings may be different.
> > 
> > `You can assume the test sit is roughly similar in size`
> > 
> > 199 * 29% = 57.71 = 58 ids PBLB with 1 unique embryo_ids
> > 
> > 199 * 71% = 141 ids PVLB with (1 in PuB & 1 Unq = 2 unique embryo_ids)
> > 
> > is my assumption!!

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-04T11:58:10.993Z — 1 votes

> > thanks, Sangram. I did think about this possibility, but from my deep learning CV experiments, the public leaderboard scores don't correlate well with local CV. Plus, if the public LB is based on one group of embryos and the private LB uses a completely different group (or even more), it's difficult for me to judge the training/test set split design."
