# division jaccard

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/737577
- **Topic id**: 737577
- **Author**: nusrati (CONTRIBUTOR)
- **Posted**: 2026-08-26T08:00:08.592129900Z
- **Votes**: 1
- **Comments**: 2

---

## Opening post

For me division_jaccard never rises above 0.0004, at any weight tested. Even at 0.5, where the ILP does find 1 real division (div_tp=1), it does so alongside 3260 false ones. div_j = tp/(tp+fp+fn) ≈ 1/3260 - swamped to near-zero by the FP volume.

This means weight-tuning cannot produce a real division_jaccard contribution. It can only choose between "suppress everything, div_j=0" (weight 1.0) or "let a few real ones through, buried under thousands of fake ones, div_j still ≈0" (weight 0.5–0.65). Neither path gets me a meaningful positive division score.

I highely need help with thisl

---

## Comments (2)


### Davit Khantadze (CONTRIBUTOR) — 2026-08-26T11:55:37.080Z

Hi Jawad, sorry for this question , which is not related to your post: I saw your score; does this mean you got this score with division jaccard being 0? Currently I am in 0.85 and wondering how to proceed next. Is it mainly about training or inference tuning; i trained 50 epochs.

#### ↳ nusrati (CONTRIBUTOR) — 2026-08-26T12:56:28.023Z — -1 votes

> Dont consider my current score for this post. What im doing is basically having 2 approaches. 
> 
> 1) Going with public model and tuning it around to grab a pinch of score somehow.
> 2) Training from scratch myself. I started doing so 2 days ago. As public one hits a final checkpoint. No further work induces it. So this post is about the scratch model im working on.
