# I measured my own division_jaccard (0.22). What's yours?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739516
- **Topic id**: 739516
- **Author**: hikaggler (EXPERT)
- **Posted**: 2026-09-04T12:23:52.430593200Z
- **Votes**: 4
- **Comments**: 0

---

## Opening post

The score is adj_edge_jaccard + 0.1 × division_jaccard, but I had no idea what my division part was, so I spent one submission to find out.

I submitted the same pipeline with the division-creating step turned off — two division-related settings
changed, everything else identical.

 - normal: LB 0.929
 - divisions off: LB 0.907
 - difference 0.022 = 0.1 × division_jaccard
 - so my division_jaccard ≈ 0.22

What surprised me is that two people on the same 0.948 can be in completely different situations:

 - if your division_jaccard is 0.30, you only need adj_edge_jaccard 0.918
 - if it is 0.15, you need adj_edge_jaccard 0.933
Those two people should be working on opposite things, and the leaderboard shows them the same number.

My question: for anyone at 0.945 or above — roughly what is your division_jaccard?
A rough number is fine, and "I ignore divisions" is just as useful an answer.

---

## Comments (0)

*(none)*
