# Biohub — Cell Tracking During Development: the terrain

Compiled 2026-08-13, at the start of our work on this contest.

## What the contest is

Detect and track cells through **3D space + time** in light-sheet microscopy videos of
**zebrafish embryos** during early development. Run by the **Royer Group at CZ Biohub**
— the lab behind [Ultrack](https://github.com/royerlab/ultrack) (Nature Methods, 2025),
which is the reference method for this exact problem. Ground truth is billed as the
largest publicly available cell-tracking annotation set, released CC0.

- Kaggle page: <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development>
- Official baseline: <https://github.com/royerlab/kaggle-cell-tracking-competition>
- Launched **2026-06-29**; final submission reported as **2026-09-29** *(confirm on the
  Kaggle page — this came from press coverage, not the rules page)*.
- ~73 teams at the time of writing. Small field compared to ROGII.

## Data

```
/kaggle/input/competitions/biohub-cell-tracking-during-development/
├── train/   {name}.zarr + {name}.geff     (images + ground-truth tracks)
└── test/    {name}.zarr                   (images only)
```

- **Images**: OME-Zarr, dimensions `(T, Z, Y, X)`, `uint16`.
  Voxel scale `(Z, Y, X) = (1.625, 0.40625, 0.40625)` µm/px — **4× anisotropic in Z**.
  Zarr attrs carry precomputed intensity quantiles (`image_statistics.quantiles`),
  so normalisation needs no full pass over the data.
- **Tracks**: `.geff` files read by [tracksdata](https://github.com/royerlab/tracksdata).
  Nodes are approximate cell centres `(t, z, y, x)` in **voxel** units; edges link a cell
  at `t` to the same cell at `t+1`; a division is one node at `t` with **two** outgoing
  edges to `t+1`.
- **Annotation is sparse** — only a subset of cells in each video is annotated. This one
  fact drives the entire metric analysis in [02-metric-findings.md](02-metric-findings.md).
- The GEFF metadata carries an `estimated_number_of_nodes` extra, which the scorer uses
  as the node budget (see below).

## Submission

This is a **CSV-upload** competition, not a notebook-runtime one. You run inference
wherever you like, then upload one CSV. Round trip:

```
predict → one .geff per test dataset → geffs_to_csv.py → submission.csv → upload
```

Schema (`scripts/geffs_to_csv.py`), one flat table over all datasets, with an `id` index column:

| column | node rows | edge rows |
|---|---|---|
| `dataset` | dataset name | dataset name |
| `row_type` | `"node"` | `"edge"` |
| `node_id` | node id | `-1` |
| `t, z, y, x` | integer coords | `-1` |
| `source_id, target_id` | `-1` | node ids |

Coordinates are rounded to **integers** in submission space, so the CSV round-trips
exactly. Node ids are remapped on reconstruction — matching is spatial, not by id.

## Scoring

```
score = adjusted_edge_jaccard + 0.1 · division_jaccard
```

- **Node matching**: predicted → GT nodes by optimal bipartite assignment on centroid
  distance, in **microns**, cutoff **7 µm**. Each predicted node matches at most one GT node.
- **Edge TP**: a predicted edge whose *both* endpoints match GT nodes joined by a GT edge.
- **Edge FN**: every GT edge without such a match.
- **Edge FP**: a predicted edge that is *not* a TP but whose source matches a GT node with
  outgoing GT edges, **or** whose target matches a GT node with incoming GT edges.
  (`pred_valid = out_valid OR in_valid`, then `FP = valid_pred_edges − TP`.)
- **Node-count adjustment**: `adj_J = max(0, J · (1 − 0.1 · (N_pred − N_total) / N_total))`
  where `N_total` is the GEFF's `estimated_number_of_nodes`.
- **Aggregation**: edge and division counts are micro-averaged (summed, *then* Jaccard);
  the *adjusted* edge Jaccard is per-sample and weight-averaged by sample size
  `TP+FP+FN`. See `summarise()` in `src/tracking_cellmot/metrics.py`.

Silent filters applied to your graph before scoring — worth knowing, they are free
cleanup you don't have to do yourself, but also constraints you cannot escape:

- edges not spanning exactly `t → t+1` are **dropped** (no gap-bridging, see findings);
- duplicate `source→target` pairs are de-duplicated;
- several predicted edges collapsing onto one GT edge are reduced to one;
- out-degree is capped at **2** per node (extra outgoing edges dropped by edge id).

## The official baseline

`royerlab/kaggle-cell-tracking-competition` — end-to-end detection + linking:

1. `TemporalUNet3D` — 3D U-Net with temporal attention, per-voxel features plus a
   1-channel detection map; centres via max-pool local-max suppression.
2. `SimpleNodeTransformer` — cross-attention over the pooled per-node features, scoring
   every `(t, t+1)` node pair; greedy assignment with `max_parents=1, max_children=2`.
3. Sparse supervision: only annotated edges contribute to the loss.
4. Optional `td.solvers.ILPSolver` post-processing for global, flow-consistent linking.

Its README says the released weights were **not trained to convergence** — "expect gains
from training longer". Its default `--det-threshold` is **0.99**, justified in the CLI
help as keeping precision up because the detector is poorly calibrated on sparse GT.
[Our metric probes say that instinct is backwards](02-metric-findings.md) — that is the
first thing to test.

## Our situation

- Compute: **Kaggle GPU notebooks** (data already mounted, no download needed).
- Starting from zero: not joined, no data pulled, no submissions.
- This container has no GPU and no Kaggle credentials, so it is for logic, analysis,
  harness code and notebook authoring — every measurement on real data runs on Kaggle.
