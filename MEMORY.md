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

**Domain intel (2026-08-14, literature + source research; full write-up `notes/03-domain-intel.md`).**
Kaggle itself is unreachable from our environment (SPA + API 401) so none of it comes from the
competition's own pages. Key items: (a) **the sparse GT comes from a second fluorescence
channel we are not given** — Ultrack's dual-channel trick; the baseline's data path is literally
`./data/dense_channel`. Labels are "random" genetically but daughters inherit them, so expect
**clonal clumping and division enrichment**, plus possible **depth bias** (a cell must be visible
in the sparse channel too) — all three now tested in recon §5b. (b) Anisotropy is **exactly 4:1**,
so XY/4 gives an isotropic 1.625 µm grid (what the baseline does). (c) **Z accuracy is worth ~4× XY**:
a 2-slice Z error is 3.25 µm, 46% of the 7 µm match budget. (d) Ultrack's ILP read from source —
one-parent + flow-conservation + mutually-exclusive-nested-hypotheses constraints, edge weight
`φ(w)=w^4`; the lab's own zebrafish `max_distance` is **5 µm**, not 7. **Set the linking radius from
motion, not from the metric** — 7 µm is 4–17× expected per-frame displacement; the binding
constraint is nucleus spacing (~6–10 µm). (e) **Frame interval is unknown and is the most
consequential gap** — recon §5c infers it from GT displacement and cross-checks against division rate.
(f) $60k prize pool. No dataset paper exists.

**RECON RAN 2026-08-14 (full write-up `notes/04-recon-results.md`).** 199 train / 4 test
datasets, all T=100, (Z,Y,X)=(64,256,256), two embryos by prefix (`44b6_` 71, `6bba_` 128).
No dataset_splits.json ships. 133,318 annotated nodes / 128,883 edges / 151 divisions.
**FLAG: all 4 test dataset names also appear in train, and we hold their GT** — check
sample_submission.csv and the data description before drawing any conclusion; if the visible
test set is the scored one, tell the organisers rather than exploit it.
**Headline: detection is essentially the whole contest.** Linking ceiling on perfect
detections = Hungarian 0.9915 / greedy 0.9847 — optimal assignment beats nearest-neighbour by
just +0.0068, and NN is the true successor **99.80%** of the time. At most ~0.015 edge Jaccard
exists in the entire linking problem. Annotated fraction median **0.0356** (1 cell in 28).
Displacement p50 1.82 µm, p99 8.38 µm, only 2.1% exceed the 7 µm cutoff → **linking radius
8-10 µm**. Median |dZ| is **exactly 0** (sub-voxel), so introduced Z error is pure noise.
adj_edge_jaccard 1.0825 confirms the uncapped under-prediction bonus on real data — but it
does NOT justify under-detecting: annotated cells are a random subset so edge recall goes as
f², and maximising f²(1+0.1(1−f)) gives f=1. **Detect everything.** Divisions 1.17 per 1000
edges — ignore them.
**Two predictions FALSIFIED:** clonal clumping not found (annotated cells are ~uniform or
*dispersed*, only 5 of ~120 datasets clumped), and divisions are rare rather than enriched.
Depth bias IS real but different in shape — annotations occupy a per-dataset Z *slab* with
whole deciles at exactly zero, not a gradient.

**Caveat carried forward:** these are toy-graph probes. Directions are structural (they
follow from code paths in `metrics.py`); magnitudes need real data. `notebooks/01_recon.ipynb`
is written and dry-run offline but **not yet run on Kaggle** — it measures annotation
density, the node budget, inter-frame displacement, cell spacing, division counts, and the
linking-only ceiling (perfect detections + NN linking) that splits the score into detection
vs linking. Nothing gets tuned until those numbers are on record.

Method carried over from ROGII (see `chat/memory/rogii-validation-harness.md` in the
`vicky0718/rogii` repo): honest fixed harness, preregistered gates, gains promoted only when
sign-stable across folds, LB movement is not evidence.
