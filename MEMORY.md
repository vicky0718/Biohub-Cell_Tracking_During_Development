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

**Recon follow-up (2026-08-15), from the inventory in `recon_summary.json`.**
(a) **The GT is tracks, not scattered points** — 4,586 annotated tracks over 199 datasets,
median 21/dataset, median length 35 of 100 frames, ~6.6 annotated cells/frame (`44b6_`
median 4 tracks, `6bba_` 30). Detection errors lose contiguous runs of edges, so per-dataset
scores have fatter tails than the f² model implies.
(b) **⭐ The test set is two datasets, and one is a node-budget trap.** `metrics.summarise`
weight-averages per-dataset `adj_edge_jaccard` by `TP+FP+FN`, so the two `6bba_` test
datasets carry **~95%** of the score (55.6% + 39.7%) and the two `44b6_` ones (2 tracks,
~50 edges each) 4.7%. Those two `6bba_` datasets have node budgets **11× apart** — 64 vs
698 cells/frame — so one fixed detection density gives ratio 10.0 and multiplier **exactly
0.0** on 39.7% of the weight. Across all 199 the budget spans 20.8×. `Config.budget_fill`
now caps detections per frame at each dataset's own `estimated_number_of_nodes / T`. This
does not contradict "detect everything": that was recall; this is precision, cheap to ~50%
and fatal below ~10%.
(c) **Hedging a second child is dead.** Extra edges pay only when `m/k > J/(1+J)` — 49.6%
at the ceiling — and the 2nd-nearest neighbour is the true successor **0.19%** of the time.
Closes the untested hedging hypothesis in **Strategy** above. Division FPs are scoped to
matched nodes, but the extra edges are not.

**`harness/purescore.py` (2026-08-15) — the Kaggle blocker is gone.** `tracksdata` needs
numpy>2, Kaggle pins numpy<2, and installing it rewrites numpy under the live kernel (two
recon runs died that way), so we could not score anything on Kaggle. The edge metric is now
reimplemented on numpy/scipy alone, transcribed from the source; `probes/verify_purescore.py`
shows it reproduces the official TP/FP/FN **exactly** on 7 structured cases (duplicates,
merges, out-degree overflow, skip/backward edges, unmatchable frames) plus 40 randomized
fuzz cases — 0 mismatches, with adj_edge_jaccard and summarise agreeing to 1e-9. Exact only
for **fork-free** predictions (then `division_jaccard` is 0 by construction); `Harness`
routes forking predictions to the official scorer and raises if it is unavailable.
Interchange is now `harness.tracks.Tracks` (plain arrays) rather than tracksdata graphs, so
`harness/`, `pipeline/` and the submission writer all import without it. Verified by
executing every cell of `02_classical_baseline.ipynb` with tracksdata blocked — it produces
a validated submission.csv.

**FIRST REAL SCORES, 2026-08-16** (`02` on Kaggle, all 199 train datasets, full 100 frames,
~4.6 h; write-up `notes/05-first-sweep.md`). Kaggle dataset mount:
`/kaggle/input/datasets/vigneshnehru/biohub-cell-tracking/biohub-cell_tracking_during_development`.

1. **Experiment #1 CONFIRMED, hugely.** det_threshold 0.99 (the official baseline's own
   default) scores **0.0490**; 0.15 scores **0.5327**. PROMOTE, +0.4836 pooled, positive in
   all five folds. The baseline's precision-protecting default costs ~0.48 of score.
   **The threshold is now saturated**: 0.15→0.05 bought 146k nodes and +0.0014 recall, so
   the missing 15.5% of GT nodes are not threshold-limited.
2. **❌ My node-budget cap was FALSIFIED.** cap ON 0.5327 vs cap OFF **0.5552** — REJECT,
   −0.0226, regressing all five folds. §9's mechanism was right but its premise was wrong:
   this detector runs *under* budget (pooled ratio −0.111), so the cap only ever cut real
   detections. `budget_fill` now defaults to None; the machinery stays for a future
   detector that over-produces.
3. **⭐⭐ The headroom is in LINKING, not detection — this reverses recon §7.** At the best
   arm node_recall=0.845 but edge_J=0.528, where recall²×ceiling predicts **0.707**.
   **Shortfall 0.179** (~42k wrong links vs ~38k missing; edge precision ~0.68), and that
   is a lower bound since edge endpoints are not matched independently. Recon's "linking is
   solved" was measured with *perfect* detections — 6.6 nodes/frame, 25 µm apart. The real
   field is ~210 nodes/frame at ~8 µm spacing with 15% of true successors missing, so a
   9 µm radius converts a miss into a miss *plus* a false positive.
4. **`test/` ships images only (4 .zarr, 0 .geff)** — no node budget at test time.
5. **🚩 The leak is CONFIRMED.** `sample_submission.csv` names exactly the four datasets
   whose ground truth ships in `train/`. Anyone can score ~1.0 by echoing it.
   **Report to the organisers; do not exploit.** Treat any LB position as meaningless.

**Status:** the metric findings (§1-5 above) remain toy-graph probes. Recon and the sweep
numbers ARE real data. Best known config: `det_threshold=0.15, min_separation_um=6.0,
link_radius_um=9.0, budget_fill=None` → **0.5552** on 199 train datasets. Still no
leaderboard score. `notebooks/03_linking.ipynb` (link_radius × min_separation grid on a
fixed 60-dataset subset, plus a local score of the four test datasets that predicts the LB
before spending a submission) is written and executed against synthetic data under Kaggle's
constraints, not yet run for real.

Method carried over from ROGII (see `chat/memory/rogii-validation-harness.md` in the
`vicky0718/rogii` repo): honest fixed harness, preregistered gates, gains promoted only when
sign-stable across folds, LB movement is not evidence.
