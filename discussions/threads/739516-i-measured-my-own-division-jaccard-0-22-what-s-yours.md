# I measured my own division_jaccard (0.22). What's yours?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739516
- **Topic id**: 739516
- **Author**: hikaggler (EXPERT)
- **Posted**: 2026-09-04T12:23:52.430593200Z
- **Votes**: 5
- **Comments**: 1

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

## Comments (1)


### José Luiz Luna-Xavier (CONTRIBUTOR) — 2026-09-07T17:35:28.277Z — -3 votes

Hi @hk4165692682,                                                                                                                                                                                                   Nice trick isolating the division term via a paired run — and the 0.948-with-two-completely-different-profiles example is a good illustration of a real problem: the combined score isn't identifiable from a single number.

One thing worth flagging on the estimate itself, though: the subtraction only gives you the true division_jaccard if edge_jaccard stayed exactly constant between the "normal" and "divisions off" runs. But parent→daughter edges are part of the edge count too — if turning off division creation means your linker only keeps one of the two daughter links instead of two, the dropped daughter becomes an edge false negative, not just a division miss. That would pull edge_jaccard down as well, so your 0.022 delta would be splitting between two effects, not attributable to division_jaccard alone. Depending on how your ablation is wired, 0.22 could be an upper or lower bound rather than the actual value — hard to say without knowing the exact mechanism.

If it helps: the official repo (royerlab/kaggle-cell-tracking-competition) ships evaluate_datasets(), which returns edge_jaccard and division_jaccard as two separate fields, computed locally against the train .geff ground truth. That gets you the exact split without spending a submission or needing to assume anything stays constant — probably worth switching to that instead of the paired-LB-diff method going forward.

As for your actual question — I don't have a submitted division_jaccard yet (still in the candidate-graph/EDA stage on my end), so I can't give you a real number. What I can share: on the training set, a simple nearest-parent candidate graph (R≈9.9µm, K≤10) only reaches ~88% coverage on the 151 GT division forks, versus ~99.7% on plain continuation edges — so there's a real, measurable gap concentrated on divisions before any classifier even gets involved. That's a ceiling on division_jaccard, not the metric itself, but it suggests the rare-event nature of divisions (a few hundred events among hundreds of thousands of annotated nodes) is probably the dominant bottleneck for most people's division score, more than model architecture per se.
