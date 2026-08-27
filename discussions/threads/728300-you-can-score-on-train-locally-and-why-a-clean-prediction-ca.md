# You can score on train locally, and why a clean prediction can go above 1.0

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728300
- **Topic id**: 728300
- **Author**: Busya PRIME (CONTRIBUTOR)
- **Posted**: 2026-07-22T19:26:45.587719500Z
- **Votes**: 7
- **Comments**: 0

---

## Opening post

The metric here is not one of the usual traccuracy presets, and there is no way to read your number without spending a submission. It turns out you can, because the organisers published the scorer.

The score for a dataset is

    score = adjusted_edge_jaccard + 0.1 * division_jaccard

Predicted nodes are matched to ground truth nodes by optimal assignment on the physical centroid distance, capped at 7 microns, with the anisotropic scale taken from each dataset (z is 1.625 microns per voxel, y and x are 0.40625). A predicted link counts only when both of its endpoints match ground truth nodes that are themselves linked. Then the base Jaccard is scaled:

    adjusted = max(0, jaccard * (1 - 0.1 * (N_pred - N_true) / N_true))

with `N_true` read from `estimated_number_of_nodes` in the ground truth metadata.

That last term explains something that confused me at first. The ground truth is sparsely labelled. Across the 199 train datasets the median is about 659 labelled nodes while the estimated true count is far higher, so a faithful prediction has fewer nodes than the estimate, the ratio goes negative, and the factor comes out above 1. Scoring above 1.0 is not a bug and the evaluation page says as much. Treat 1.0 as a reference point, not a ceiling.

Two practical notes. The division term is only weighted 0.1, but it is also the part most trackers drop entirely, so it is a cheap tenth once your links are solid. And the scorer needs only the small ground truth graphs, not the image volumes, so it runs in seconds on CPU.

The organisers' scoring code is public under BSD 3 Clause at github.com/royerlab/kaggle-cell-tracking-competition. I wrapped it in a drop in function that takes a submission and the train directory and returns the exact number, with worked examples of what each kind of mistake costs: https://www.kaggle.com/code/busyaprime/score-your-cell-tracker-locally

---

## Comments (0)

*(none)*
