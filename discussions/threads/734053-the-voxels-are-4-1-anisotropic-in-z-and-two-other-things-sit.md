# The voxels are 4:1 anisotropic in Z, and two other things sitting in zarr.json

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734053
- **Topic id**: 734053
- **Author**: maximo lorenzo y losada (CONTRIBUTOR)
- **Posted**: 2026-08-09T16:58:04.843170900Z
- **Votes**: 1
- **Comments**: 0

---

## Opening post

Before downloading the full 5 GB I read the shipped metadata, and three things in it seem worth flagging early. All of this is about 1 KB of JSON per dataset and none of it needs the arrays.

1) Voxels are not cubes. The OME-NGFF coordinateTransformations give scale = [T 1.0, Z 1.625, Y 0.40625, X 0.40625]. So Z / Y = exactly 4.0 — one voxel step in Z covers four times the physical distance of one step in Y or X.

This matters more here than it would in a segmentation task. Tracking links detections between frames, and nearly every linking rule is a distance: nearest neighbour, a gating radius, an assignment cost, a graph edge weight. Computed on raw voxel indices Z is understated 4x, so a tracker will happily link cells that are physically far apart in depth while rejecting closer ones in plane. Two detections 4 voxels apart in Z are 6.5 micrometers apart; 4 voxels apart in Y are 1.63.

Every dataset I could see shares the same scale vector, but the notebook prints how many distinct ones it finds rather than hardcoding it.

2) The intensity distribution is heavy-tailed, and the metadata hands you the quantiles so you do not have to compute them over 5 GB:

0.0 -> 15, 0.1 -> 75, 0.9 -> 497, 0.99 -> 1478, 1.0 -> 4319

The range is 15 to 4319, but 90% of voxels sit at or below 497, which is 11.2% of that range. Min-max scaling to [0,1] therefore puts 90% of your data below 0.112. Clipping at the 99th percentile before scaling keeps far more dynamic range where the voxels actually are.

3) Chunking is one timepoint per chunk: shape (100, 64, 256, 256) uint16, chunk_shape (1, 64, 256, 256), blosc/zstd with bitshuffle. Zarr decompresses a whole chunk regardless of how little you asked for, so reading one full timepoint touches 1 chunk while reading a single Z-slice across all time touches 100 chunks — 100x the I/O for 64x less data. Time-major access is nearly free; anything that walks time at fixed depth pays for the entire array.

One more, on the submission: sample_submission.csv carries row_type, source_id and target_id, so it is a graph. Nodes hold positions, edges hold the links between them. A perfect detector with no linking scores nothing.

Notebook with all of it computed in-cell from the shipped zarr.json:
https://www.kaggle.com/code/maximolorenzoylosada/your-voxels-are-not-cubes

No method here, just the metadata read. If I have misread the coordinateTransformations convention — in particular whether the scale is meant to be applied the way I have — I would like to be corrected.

---

## Comments (0)

*(none)*
