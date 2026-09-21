# Stuck at 0.947 after ten board tests. What I measured, and one question about the relink stage 

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742266
- **Topic id**: 742266
- **Author**: Justin CH123 (CONTRIBUTOR)
- **Posted**: 2026-09-21T03:39:55.992547600Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

Hi all. I have been stuck on the 0.947 plateau for a week. I wanted to share what I measured in case it saves someone time, then ask one question.

I tested ten single-knob changes on top of the public Harmonic Fusion pipeline and every one lost on the board. The clearest pattern was that the loss tracked the change in predicted node count on the test set, not my offline validation score. The change with my best offline edge score (+0.0011) scored worst on the board (-0.005), and it was also the one that dropped the most nodes on one test video. A change that moved the node count by only four nodes still lost 0.001. For this pipeline, offline edge J on the training videos was not a usable signal for me.

I also noticed that in the public post-processing the final edges come from the Hungarian motion relink in filter_output_graph, which replaces the ILP edge set. The network probability only enters that cost through the learned bonus, worth at most about 1 um against a median true step of about 1.8 um.

My question is whether anyone found a gain at that relink stage specifically, for example feeding dense edge probabilities into the relink cost, without changing which cells are kept. More broadly, for anyone above 0.95 willing to say, was most of your gap over the public stack on the detection side or the linking side?

Thanks, and good luck in the last week.

---

## Comments (0)

*(none)*
