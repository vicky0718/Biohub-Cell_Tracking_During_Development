# ⚠️ Don't write float centroids - same graph, 0.954 → 0.946 on the public LB

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744093
- **Topic id**: 744093
- **Author**: Hammad Farooq (CONTRIBUTOR)
- **Posted**: 2026-09-28T16:29:09.221666500Z
- **Votes**: 8
- **Comments**: 0

---

## Opening post

A short warning, because this one cost us a submission slot and it looks like a free win.

The idea: the reference metric in royerlab/kaggle-cell-tracking-competition casts z, y, x to Float64 (csv_to_geffs.build_graph_from_rows). Most of the strong public pipelines now refine centres to sub-voxel precision with a coordinate head, then throw that away with int(round(...)) at the final write. With z at 1.625 µm per voxel, writing the float centroid looks like it should help.

What we did: we took one of our 0.954 submissions and changed exactly one thing - node coordinates written as round(v, 3) instead of int(round(v)). Same node ids, same edges, same t, byte-identical graph. Before submitting we checked that rounding the float output reproduces the int submission for 99.95% of nodes (the rest are .5 ties).

Result on the public LB:
int coordinates: 0.954
float coordinates (3 dp): 0.946

It scored normally (no scoring error) and lost 0.008.

Our best guess: the Kaggle-side scorer does not treat non-integer input the way the GitHub reference suggests. If it floors / truncates instead of rounding, every centre moves by about -0.5 voxel on average (~0.8 µm in z), which plausibly costs this much on borderline matches. We have not confirmed the mechanism - a clarification from the hosts would be welcome.

Takeaway: keep int(round(...)) on the final write, even if your coordinates are refined upstream. The sample submission uses ints for a reason.

Hope this saves someone a slot in the last days. Good luck everyone!

---

## Comments (0)

*(none)*
