# The linking radius is 8.4 µm, and divisions are one link in 853

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/733973
- **Topic id**: 733973
- **Author**: Luka Duvanov (CONTRIBUTOR)
- **Posted**: 2026-08-09T10:37:14.977001900Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

Two numbers that decide a tracker's architecture, measured across all 199 training movies and 128,883 ground-truth links:

Frame-to-frame displacement: median 1.82 µm, p95 5.34 µm, p99 8.38 µm. A nearest-neighbour linker with an 8.4 µm radius reaches 99% of true links. In voxels that is ~21 in x/y but only ~5 in z — voxels are 1.625 × 0.40625 × 0.40625 µm, so a radius expressed in voxels is wrong in one axis by 4×.
Divisions: 151 in total, i.e. one link in 853, and 112 of the 199 movies contain none at all.
Two things that cost me time and needn't cost yours:

.geff is plain zarr — the geff package is not in the Kaggle image and you do not need it. Node properties live at nodes/props/<name>/values. Twelve lines of zarr do it.
The ground truth follows ~22 tracks per movie, not every cell, but 100% of its edges span exactly one frame — no gap-closing to learn, and an unannotated blob is not a missed detection.
Also: use the intensity quantiles shipped in the zarr metadata for normalisation. p99.9 is 2145 against a maximum of 4319, so min-max squashes everything useful into the bottom half.

https://www.kaggle.com/code/nekkon/the-linking-radius-is-8-4-um

---

## Comments (0)

*(none)*
