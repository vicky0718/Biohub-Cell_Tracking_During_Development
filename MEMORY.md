---
name: biohub-metric-structure
description: "Biohub cell-tracking contest — verified structure of the scoring metric and the strategy that follows from it"
metadata: 
  node_type: memory
  type: project
---

Started 2026-08-13. Contest: [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(Royer Group / CZ Biohub, launched 2026-06-29, deadline reported 2026-09-29, ~73 teams).
Track cells in 3D+time zebrafish embryo light-sheet videos. **CSV-upload** contest, not
notebook-runtime. Compute = Kaggle GPU notebooks. Workspace = this repo (`vicky0718/Biohub-Cell_Tracking_During_Development`).

`score = adjusted_edge_jaccard + 0.1 · division_jaccard`. Nodes matched to GT by bipartite
assignment on centroid distance ≤ **7 µm**; edges must span exactly t→t+1; out-degree
capped at 2. Data: OME-Zarr (T,Z,Y,X), voxel scale (1.625, 0.40625, 0.40625) µm — 4×
anisotropic in Z. GT tracks in `.geff` (tracksdata). Official baseline:
`royerlab/kaggle-cell-tracking-competition` (TemporalUNet3D detector + SimpleNodeTransformer
linker, released weights explicitly **not** trained to convergence).

**Metric structure, verified by running the official scorer** (`probes/*.py`, all
numbers reproduced; full write-up in `notes/02-metric-findings.md`):

1. **FPs are nearly free — the GT is sparse.** Adding 300 fictional tracks (1800 nodes) to
   a perfect prediction left the score at exactly 1.0000. An edge only counts, as TP *or*
   FP, if an endpoint matched an annotated node. Duplicate detections inside the 7 µm
   radius are free too.
2. **The node budget is a weak two-sided multiplier**: `×(1 − 0.1·(N_pred−N_total)/N_total)`
   against the GEFF's `estimated_number_of_nodes`. 2× budget → ×0.90; **under-predicting
   gives an uncapped bonus** (0.2× budget → ×1.08). You'd need 11× over-prediction to zero out.
3. **A missed detection is unrecoverable.** Non-consecutive edges are dropped, so a t→t+2
   bridge scores *identically* to no edge at all. One missed detection = 2 permanent FN.
   This is the hard ceiling; detection recall is the one thing that can't be bought back.
4. **A wrong link costs exactly 2× a missing link** (1 FP + 1 FN vs 1 FN) — confirmed at
   scale: a 250-cell synthetic run produced 565 FP and 565 FN, one of each per error. But
   this argues for *aggressive* linking, not caution, since declining forfeits the edge
   anyway: link whenever `p > N/(N+D+1) ≈ J/(1+J)` — **~⅓ at J≈0.5**, rising toward ½ only
   as the score approaches perfection.
5. **Divisions: don't force them.** A division one frame late cost 0.30 of edge Jaccard to
   earn at most 0.1×div. Missing one is strictly better than mistiming it.

**Strategy:** detect aggressively (the baseline's default `--det-threshold 0.99`, justified
in its own CLI help as protecting precision, is pushing the axis that barely matters —
sweeping it down is experiment #1); link aggressively near p≈⅓; hedge ambiguous links
across both allowed children (bounded downside: the whole 0.1 division term) — untested
hypothesis; chase divisions last. Global assignment beats greedy NN by a lot on synthetic
dense data (J 0.726 vs 0.512), consistent with the baseline shipping an ILP option and with
Ultrack's design.

**Caveat carried forward:** these are toy-graph probes. Directions are structural (they
follow from code paths in `metrics.py`); magnitudes need real data. `notebooks/01_recon.ipynb`
is written and dry-run offline but **not yet run on Kaggle** — it measures annotation
density, the node budget, inter-frame displacement, cell spacing, division counts, and the
linking-only ceiling (perfect detections + NN linking) that splits the score into detection
vs linking. Nothing gets tuned until those numbers are on record.

Method carried over from ROGII (see `chat/memory/rogii-validation-harness.md` in the
`vicky0718/rogii` repo): honest fixed harness, preregistered gates, gains promoted only when
sign-stable across folds, LB movement is not evidence.
