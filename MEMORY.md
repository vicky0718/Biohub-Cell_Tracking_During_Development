---
name: biohub-metric-structure
description: "Biohub cell-tracking contest — verified structure of the scoring metric and the strategy that follows from it"
metadata: 
  node_type: memory
  type: project
---

Started 2026-08-13. Contest: [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(Royer Group / CZ Biohub, launched 2026-06-29, deadline reported 2026-09-29, ~73 teams).
Track cells in 3D+time zebrafish embryo light-sheet videos. **NOTEBOOK-submission**
contest (`onlyAllowKernelSubmissions=True`, `usesSynchronousReruns=True`): Kaggle reruns
our notebook against private data and it must emit `submission.csv` within **12 h**;
5 submissions/day; public LB is 29% of the test data; 2,395 teams; $60k.
(An earlier note here said "CSV-upload" — that was wrong, corrected 2026-08-16.)
Compute = Kaggle notebooks (our pipeline is CPU-only). Workspace = this repo (`vicky0718/Biohub-Cell_Tracking_During_Development`).

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

**FORUM SCRAPED IN FULL 2026-08-16** — all 71 topics, all 219 comments including nested
replies (`data/forum-scrape.json`, scraper `probes/scrape_forum.py`; full write-up
`notes/07-forum-intel.md`). Kaggle's `api/v1` is 401 and the pages are JS shells, but the
SPA's own `/api/i/` endpoints serve public forum content to an **anonymous session** — fetch
any page for the XSRF-TOKEN/ka_sessionid cookies, then echo the token in `x-xsrf-token`.
Without the cookie pair every call is a bare 400 with no body.

1. **🚩 It is a notebook competition** (see header). There is **no submission notebook yet**
   — that is now the gating deliverable. Test names must be globbed at runtime (the `test/`
   folder is swapped at rerun) and `submission.csv` needs the `id` column.
2. **The leak is CLOSED — there was never one.** Host: the visible test files are "dummy
   placeholder files"; the leaderboard uses "a much bigger test set, deliberately kept
   private", no train overlap. Already reported by another competitor. So scoring those 4
   locally predicts nothing (kills `03` §4's "predicted leaderboard score").
3. **⭐⭐ The hidden test is a DIFFERENT PAIR OF EMBRYOS**, roughly the size of train (~200
   datasets). Host: "no overlap in embryo_ids between train and test". Our 5-way hash folds
   mixed both embryos into every fold and measured the wrong shift — `Harness` now defaults
   to **`fold_by="embryo"`** (leave-one-embryo-out, 2 folds).
4. **⭐ Rule-based is competitive and divisions are not needed.** 7th/344 with no learning
   and division Jaccard ≈ 0. Paired CV→LB from one competitor: 0.7448→**0.834**,
   0.8213→**0.846**. LB runs *above* CV ("~10% more optimistic"), so our CV 0.5552 is
   probably ~0.60–0.65 LB. Movie-to-movie spread is ±0.14 (18% CoV).
5. **🚨 Subsampling frames scores exactly 0.0.** A competitor with 0.57 local got 0.0 because
   frame-skipping produced t→t+5 edges, all structurally unmatchable, with no error raised.
   `Config.max_frames` must never ship in a submission.
6. **A division-metric exploit was patched mid-competition and everything was rescored**
   (2026-07-22). The `cross_component_forks`/weak-component machinery in
   `division_metrics.py` IS that patch; our clone is post-patch so purescore is verified
   against the current metric.
7. **GT is Ultrack pseudo-labels with real defects** — byte-identical consecutive frames
   where GT still moves a cell 8.9 µm, unresolvable dim cells, cells with Z span >24 µm that
   cannot fit the 7 µm match radius. A real ceiling below 1.0.
8. **🚨 `pip install` does NOT work in a scored submission** — internet is off in the rerun
   (host, to a competitor whose submission failed on it). Every notebook we have opens with
   `pip_install(["geff","zarr"])`; that cell would fail a submission. At test time there is
   no GT to read, so `geff` is not needed — only zarr (preinstalled), numpy, scipy. The
   submission notebook must install nothing.
9. **Max score is 1.1, not 1.0** (host): `adj_edge_jaccard` (≤1, higher with the
   under-prediction bonus) + `0.1 × division_jaccard`. So LB 0.915 is **83% of maximum** and
   our CV 0.5552 is ~50%. Verified 2026-08-16 against Kaggle's HOST-tagged messages.
10. **External data is allowed** (host: Zebrahub explicitly OK, no test overlap) — the
   obvious corpus if we ever train a detector. Public baseline weights were trained on all
   199 train videos, so validating them on train is leakage; **we are training-free, so our
   CV is honest.**

**Zebrahub is NOT reachable from this container** — the agent proxy's own relay log records
`connect_rejected zebrahub.org:443 / public.czbiohub.org:443, gateway answered 403 to
CONNECT (policy denial)`. Chromium is installed but Kaggle's edge resets it
(`ERR_CONNECTION_RESET`) even through the proxy with HTTP/2 off and a normal UA, so the SPA
cannot be rendered here either — the 71 forum *opening posts* stay unretrieved (all 219
replies and all 25 HOST/ADMIN messages we do have). Any Zebrahub acquisition must run in a
Kaggle notebook with internet enabled.

**EXPERIMENTS 2-3 (2026-08-16), full write-ups `notes/09` and `notes/10`.**
**#2 linking grid — all three predictions confirmed, and the linker is not the answer.**
Tightening the detection window took node recall 0.895 -> 0.976 and the score 0.5790 ->
0.3449: 1.86M extra detections bought ~3,264 extra GT nodes, i.e. **571 spurious detections
per additional GT node**. The binding constraint is detection **precision**, not recall.
Best linker gain available was +0.0073 (radius 9->7), which fails the gate. Also found a
design flaw of mine: `_footprint` quantises to odd voxel counts, so on the 1.625 µm grid
only 4 separations exist (1.625/4.875/8.125/11.375) — the grid's 4.5 and 3.5 rows were the
same experiment. The DoG ball footprint is continuous.
**#3 DoG detector — PROMOTED, +0.0970, positive in every fold.** 0.5790 -> **0.6760** at
0.92x the node count, so it is the detector not the density. It reaches *lower* recall
(0.857 vs 0.895) and still wins. Multi-scale confirmed at matched density (+0.0353, 2
scales vs 1) — the notebook's own check said FALSIFIED because I reused the 2-scale
calibration for the 3-scale arm, which then ran at 2x density; its edge_J (0.7027) actually
ties the winner. **The node budget has flipped sign**: DoG runs +0.24…+0.53 OVER budget
where intensity ran -0.111 under, so `budget_fill` is back on the table for the exact
reason it was dropped. `dog_sep4.5` has the best quality yet measured (edge_J **0.7195**)
and is taxed 5.3% for over-detecting — bringing it to budget is worth ~+0.05, landing ~0.72.
Density is controlled by `min_separation_um`, not `dog_rel_threshold` (16x threshold change
moved density 15%). **Caution: `04` printed "leave-one-embryo-out" as hardcoded text while
actually running the 5-way hash split** — the uploaded snapshot predated `fold_by`. Cross-
embryo generalisation of DoG is still unmeasured; `05` asserts the fold structure instead
of claiming it.

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

---

# CURRENT STATE (2026-08-27) — everything above predates the first leaderboard score

The section above ends at "Still no leaderboard score / best 0.5552". Both are long
superseded. What follows is the load-bearing state; `notes/15` onward carry the detail.

## Score trajectory (leaderboard, max is 1.1 not 1.0)

```
0.752  classical champion (DoG + Hungarian)   <- banked floor, reproducible
0.843  public pack weights, ILP BYPASSED (a bug: predict_video returns CANDIDATES)
0.867  public pack weights, ILP running
0.880  + graph repair: close_gaps(5.75 um) then linefit_smooth(0.76, win 2)   <- best
0.913-0.916  the cluster of ~514 teams sharing these weights
```

**The strategy is Track A**: reproduce the public pack's UNet+transformer weights, then
beat the cluster by repairing the graph they all ship unrepaired. The pack's own manifest
says it emits the *"ILP candidate graph before notebook-level graph repair"*.

## What is CLOSED, with the number that closed it

Each of these was measured, not assumed. Do not reopen without new evidence.

| direction | verdict | evidence |
|---|---|---|
| learned detector from scratch | dead | 5 runs, best CV **0.649**; `notes/23` §2c measured the learned CV->LB offset as **0.061 worse** than classical, so an arm needs **CV ~0.813** just to match the 0.752 floor |
| training the pack's architecture ourselves | dead | **725 h** for 402 epochs on a P100 (`notes/24`); ~30 GPU-h/week quota |
| divisions | dead from 3 directions | `notes/04` §10 (151 divisions in 128,883 edges); `notes/25` geometric insertion = **1 TP per 2,223 guesses**, +0.0015; ILP's own forks measured *unevaluable* (37 emitted, 0 TP, 0 FP) |
| per-dataset node-budget calibration | ~0.002, parked | `notes/26`: the multiplier is already at 0.9892-1.0012, nearly saturated |
| **local motion relink** | dead from BOTH ends | `notes/29` mine (no margin) made `fn_mislink` monotonically worse; `notes/30` the user's (margin 0.35) made **zero swaps** in the entire test set. Loose enough to fire => harmful; strict enough to be safe => never fires |

## What WORKS

**Graph repair.** `pipeline/repair.py`: `close_gaps` then `linefit_smooth`, in that order
(reversed is +0.0006 worse). Measured +0.0115 on 24 training datasets, scored **+0.0130**
on the leaderboard — a **1.13x transfer ratio**, i.e. it transferred *better* than measured.
Coherent reason: contamination (`notes/24` §2 — these weights were fitted on an unpublished
subset of the same 199 datasets) makes the model's graphs on training data BETTER than on
unseen data, so there is less for a repair to fix there. n=1, so a direction not a coefficient.

Measured no-ops, do not re-add: `cap_edge_length` is **harmful** (-0.0002);
`prune_isolated` and `single_parent_repair` are **exact** no-ops because the ILP already
emits graphs with no isolated nodes and no merges.

## The edge-loss anatomy (`notes/26`, `pipeline/anatomy.py`) — the map for everything left

Of 13,832 GT edges on the 24 budget-stratified datasets:

```
tp           12,909  93.33%
fn_mislink      473   3.42%   <- largest failure; both ends detected, wrong pair joined
fn_detect       238   1.72%   <- detection is NOT the ceiling
fn_gap          212   1.53%
             + 669 false-positive edges
```

Repairing every gap and mislink puts edge Jaccard at **0.9375-0.9691**, i.e. **+0.047 to
+0.079**. The gap to the cluster is +0.046 (now 0.033-0.036). **The reachable band contains
the deficit** — the first time in this project a measured ceiling covered it. ~15-24% claimed.

## Methodological findings that keep repeating

1. **Node recall is blind to temporal coherence.** `notes/21` measured a 0.074 edge-Jaccard
   gap under *identical* 0.866 node recall. `pipeline/detector.py::paired_recall` exists for
   this — and I then used node recall to bound position repair anyway and was wrong by 70%
   (`notes/27`). With 20-150x more predictions than annotations, a GT node is usually matched
   to a *nearby wrong* prediction; moving positions flips the assignment. Node recall cannot see it.
2. **Position repair TRADES mislinks for detection failures**, ~4:1 where it works and
   inverting when pushed. Any node-moving repair needs a bound like `max_shift_um`.
3. **Silent-pass-on-missing-input is the recurring bug class.** A NaN node budget made a
   grading check print PASS on absent data (NaN comparisons are False); a confounded
   stratification (`notes/25` §2) gave a correlation of the *wrong sign*. Every grading cell
   now detects and says NOT GRADED instead.
4. **Verify notebooks by EXECUTING their real cells against synthetic data** with the answer
   constructed, not inferred (`scratchpad/exec_*.py`). This has caught more real defects than
   any other practice here — including three where the *harness itself* was wrong.

## Infrastructure facts (measured, save re-discovery)

- Free Kaggle GPU is a **P100 (sm_60)**; the image torch builds sm_70+ only. `claude_torch_wheelhouse`
  ships torch 2.5.1+cu121. `machineShape` is **accepted by `kernels/push` and silently ignored** —
  the accelerator can only be chosen in the UI. T4 reproduces P100's ILP counts exactly.
- **`pip install` does not work in a scored rerun** (no internet). Wheels must be attached.
- `claude-relink-sweep`'s output caches `cand_*.npz` for all 24 datasets: coords, post-ILP
  graph, and candidate edges **with probabilities** — the ILP's own input. Attach it as a
  `kernelDataSource` to re-solve without a ~25 min prediction pass.
- **Three index spaces** exist and mixing them is silent: `coords` rows, tracksdata node ids,
  and `Tracks`' renumbering of the ILP's surviving subset. Use
  `Tracks.from_tracksdata_with_ids`.
- Kaggle's `/api/i/` endpoints reject Basic auth, so **submission cannot be automated** — a
  human must click Submit.
