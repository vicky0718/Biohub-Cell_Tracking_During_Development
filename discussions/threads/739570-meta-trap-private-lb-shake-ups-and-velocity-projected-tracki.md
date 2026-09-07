# Meta Trap, Private LB Shake-ups, and Velocity-Projected Tracking 🚀

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739570
- **Topic id**: 739570
- **Author**: Rishavendra Sharma (CONTRIBUTOR)
- **Posted**: 2026-09-04T17:45:43.778331300Z
- **Votes**: 2
- **Comments**: 2

---

## Opening post

Hey everyone! As we get closer to the end of the competition, I’ve been doing a deep dive into the architecture of the public 0.940+ notebooks. I wanted to share some observations and spark a discussion about the current meta and where we are heading.


1. The Public Baseline "Trap" (Are we all overfitting?) If you look closely at the top-scoring public scripts, they are heavily reliant on the biohub-tracking-support-pack datasets. Rather than running lightweight inference natively, these notebooks often shell out to a zipped GitHub repo to run heavy ILP solvers, or they just run post-processing on pre-computed .geff tracks. Because a massive portion of the leaderboard is essentially post-processing the exact same set of baseline detections, doesn't this guarantee a massive Private LB shake-up?

Question 1: Are you guys actually training your own 3D UNets from scratch for your final submissions, or is the current meta purely focused on ensembling and pruning these public ILP graphs?

2. Fast-Motion Occlusions vs. Static ILP I noticed that the public ILP gap-closers (which heavily rely on static XYZ distance or bidirectional harmonic distance) tend to drop tracks when cells move rapidly and occlude each other. To combat this, I’ve been experimenting with Velocity-Projected Gap Closing. Instead of measuring the distance from where a cell disappeared, I calculate its EMA (Exponential Moving Average) velocity and project its position forward by velocity * gap_frames. Connecting the projected position to the target target drastically improves recovery on fast-moving occlusions.

Question 2: How is everyone else handling these fast-motion occlusions? Are you incorporating explicit motion prediction into your cost matrices, or are you relying entirely on the ILP solver's spatial tolerance?

3. The DeepCenter Veto The DeepCenter veto logic in the public scripts is a brilliant heuristic for rejecting spurious wide divisions. However, by aggressively pruning these graphs, it feels like we might be discarding genuine biological outliers that will be present in the Private test set.

Question 3: Has anyone found a sweet spot for tuning the DeepCenter gap threshold, or found a better architectural way to validate tricky divisions without just blindly pruning them?

Would love to hear your thoughts. Good luck to everyone in the final stretch!

---

## Comments (2)


### Mendrika Ramarlina (MASTER) — 2026-09-05T05:23:27.347Z — 1 votes

A lot of us, myself included are for sure overfitting the LB. Trust you local CV, good principles still apply.

#### ↳ Rishavendra Sharma (CONTRIBUTOR) — 2026-09-05T08:08:30.167Z — 1 votes

> The 50ep heuristics are great for a quick public score, but sticking to solid local CV principles is definitely the only way to survive the remaining 79% of the private data. Good luck in the final stretch!
