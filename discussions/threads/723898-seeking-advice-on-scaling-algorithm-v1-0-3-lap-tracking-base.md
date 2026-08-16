# Seeking Advice on Scaling Algorithm v1.0.3 LAP Tracking Baseline (Current Local CV: ~0.590) – Beyond Frame-to-Frame Euclidean Assignment

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723898
- **Topic id**: 723898
- **Author**: Harshul Gupta (CONTRIBUTOR)
- **Posted**: 2026-07-08T16:57:53.942132900Z
- **Votes**: 4
- **Comments**: 0

---

## Opening post

Hi everyone,I am currently working on optimizing a production baseline for this challenge. My current iteration uses a scale-space Laplacian of Gaussian (LoG) response for 3D peak detection on downsampled spatial volumes (::2 factor on Y and X), followed by a multi-frame Linear Assignment Problem (LAP) solver with constant velocity projection vectors. While this approach cracked the local validation plateau and scored 0.590, I am noticing some clear failure modes where the tracking accuracy drops significantly. I would love to get your insights or advice on how to tackle these specific bottlenecks:
1. Handling High-Density Track Swapping. In areas where cell clusters become highly dense, linear cost matching solely based on physical Euclidean distance (even with velocity anchoring) causes the solver to swap tracks during cell crossovers.Question: Has anyone successfully integrated morphological features (like local intensity gradients or volume ratios) directly into the LAP cost matrix to regularize distance constraints?
2. Fine-Tuning Mitosis & Daughter Cell Linkage Right now, my division detection triggers when an unassigned "orphan" node appears near a historical track boundary. However, during rapid cell divisions, capturing the sudden split and assigning correct parent-daughter relationships degrades our TRA score. Question: What are your best practices for optimizing the temporal window size or building a dedicated matrix for splitting events rather than handling it as a post-processing heuristic?
3. Alternative Vectorized Graph Solvers Network X is clean but introduces bottlenecks when resolving global constraints (like enforcing in_degree <= 1 globally across hundreds of frames).Question: Are there specific graph solver libraries (e.g., Google OR-Tools, scipy.optimize.milp) or minimum-cost max-flow frameworks that you’ve found to be faster and memory-stable within Kaggle's 12-hour constraint?

Would love to hear your thoughts, structural tips, or any open-source benchmarks you've found helpful for 3D cell tracking!

Thanks in advance, and good luck to everyone competing!

---

## Comments (0)

*(none)*
