# Has anyone experienced extremely long scoring times?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/717228
- **Topic id**: 717228
- **Author**: Rahul Parmeshwar (CONTRIBUTOR)
- **Posted**: 2026-07-01T17:09:11.807075500Z
- **Votes**: 3
- **Comments**: 5

---

## Opening post

Hi, my last two submissions failed because scoring timed out, the submissions.csv was produced but after that the scoring element of kaggle timed out after 12 hours. Anyone else experienced this?

---

## Comments (5)


### Sergio Alvarez (MASTER) — 2026-07-01T17:37:42.173Z — 4 votes

Hi @rahulparmeshwar. Here's the snippet from the overview page:
>When a notebook is submitted for rerun, a new hidden test set is swapped in. The size of the hidden test set is approximately the same size as the training dataset.

The training set has 199 image volumes (T=100, Z=64, Y=256, X=256), so the hidden test set will be roughly that same size. I haven't submitted, but ~200 second per volume should fit in 12h

#### ↳ Rahul Parmeshwar (CONTRIBUTOR) — 2026-07-01T23:17:47.330Z

> Understood! Thanks Sergio!

### Arunava Nag (CONTRIBUTOR) — 2026-07-04T08:24:37.790Z

facing this issue, were you able to find a solution

#### ↳ Rahul Parmeshwar (CONTRIBUTOR) — 2026-07-04T10:38:54.043Z — 1 votes

> Yeah, but it wasn’t on my end, it just takes 5 hours for me to run the full validation on scoring, so you’ll just have to wait because there’s 199 validation images

#### ↳ ↳ Arunava Nag (CONTRIBUTOR) — 2026-07-04T12:36:33.480Z

> > Thanks. Mine is just timing out error after 14hours.
