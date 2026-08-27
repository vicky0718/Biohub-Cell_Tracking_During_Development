# How much points are you guys getting

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734192
- **Topic id**: 734192
- **Author**: mikelou1 (EXPERT)
- **Posted**: 2026-08-10T12:04:24.138409800Z
- **Votes**: 3
- **Comments**: 2

---

## Opening post

Hey guys! I'm really curious how much Edge Jaccard and Division Jaccard you guys have. For the public evaluation, my Division Jaccard is roughly 0.03 and edge 0.898.

---

## Comments (2)


### Arul Prasad S P (CONTRIBUTOR) — 2026-08-10T14:14:20.740Z — 1 votes

Nice thread — I'd like to compare, but I want to be sure I'm reading your numbers the same way you are.

1. Is the 0.03 the raw division_jaccard, or its weighted contribution to the score? The official metric is score = adj_edge_jaccard + 0.1 * division_jaccard (SCORE_DIVISION_WEIGHT in tracking_cellmot/metrics.py). So 0.03 raw adds only 0.003 to your total, whereas 0.03 after weighting means division_jaccard ≈ 0.30. Those are very different worlds, and your edge number reads quite differently depending on which you meant.

2. How did you get the split on the public LB at all? The leaderboard only returns the combined number, so I assume this is either a local measurement or an ablation. If it's local — does your local edge/division split track the LB closely? Mine doesn't; my local harness is noticeably miscalibrated against the public score, and I'd be interested to know whether yours behaves better.

One thing that may be useful in return: summarise() drops the division term entirely when a submission contains no divisions at all (score = edge_jaccard if not has_divisions). So a fork-free submission returns your pure adjusted edge Jaccard, and the division term follows by subtraction — one submission for an exact split.

#### ↳ mikelou1 (EXPERT) — 2026-08-10T14:19:06.043Z — 1 votes

> 0.03 is already weighted [so 30%], I got it on public lb by submitting a division only and reducing it slightly for matched edges.
