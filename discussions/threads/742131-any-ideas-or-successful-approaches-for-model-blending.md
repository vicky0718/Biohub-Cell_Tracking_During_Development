# Any ideas or successful approaches for model blending?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742131
- **Topic id**: 742131
- **Author**: sghwr (CONTRIBUTOR)
- **Posted**: 2026-09-20T04:21:00.360209900Z
- **Votes**: 2
- **Comments**: 1

---

## Opening post

its just ~10 days before competition ends,and we managed to have two private model structures that are slightly weaker than the current best public models on the leaderboard, with a gap of only around 0.015–0.020.
We have been experimenting with different blending and ensemble strategies, hoping that the private models could provide complementary signals to the stronger public pipeline. However, none of our attempts has improved the Public LB so far. we're stucked😭
I'd be grateful if I could get some advice on two issues:

1. Has anyone successfully improved their Public LB using model blending or ensembling in this competition?

2. If so, would you be willing to share some general ideas about what worked, such as the blending stage, matching strategy, confidence calibration, or how you handled detection nodes, edges, and divisions?
We are not asking for private code or exact parameters. Even high-level observations about successful or unsuccessful approaches would be very helpful.

Thanks! ❤

---

## Comments (1)


### Yunk-S (CONTRIBUTOR) — 2026-09-20T07:33:21.217Z

卡了兄弟😭  我感觉开源的代码已经差不多优化到极限了
