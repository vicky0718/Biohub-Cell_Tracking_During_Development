# End-to-end scouting of the competition discussions

Read of all 71 threads / 219 comments in `threads/`, plus the competition metadata and a
public-leaderboard snapshot in `raw/`. Scraped 2026-08-16, competition closes 2026-09-29.

Every claim below is sourced to a thread id (`threads/<id>-*.md`), to
`raw/competition.json`, or to `raw/leaderboard.json`. Where a thread contradicts something
in our `MEMORY.md`, the contradiction is stated plainly and the source given, so the call
on which to believe is yours.

---

## 0. The headline

Our facts are good. Our position is not.

The measurements in `notes/04-recon-results.md` are independently corroborated by the forum
— thread 733973 reports 128,883 GT links, 151 divisions, displacement p50 1.82 µm / p99
8.38 µm, and ~22 tracks per movie, which are our recon numbers to the digit. Thread 728300
derives the same adjusted-Jaccard structure and the same >1.0 behaviour we found. So the
terrain is mapped correctly.

But **our best score, 0.5552, would sit at rank ~2004 of 2402 on the public leaderboard**
(`raw/leaderboard.json`), and three of our operating assumptions about the *contest itself*
are wrong in ways that invalidate parts of the harness. The gap is not a modelling subtlety;
it is a plain detector gap plus a validation target aimed at the wrong thing.

---

## 1. Where we actually stand

Public LB snapshot, 2,402 teams (`raw/leaderboard.json`):

| Score | Rank | Percentile |
|------:|-----:|-----------|
| 0.950 (top) | 1 | — |
| 0.940 | 10 | top 0.4% |
| 0.930 | 27 | top 1.1% |
| 0.920 | 71 | top 2.9% |
| **0.910** | **744** | **top 30.9%** |
| 0.900 | 1019 | top 42.4% |
| 0.850 | 1522 | top 63.3% |
| **0.5552 (us)** | **~2004** | **top 83.4%** |

Median 0.890, p25 0.746, p75 0.913.

The shape matters more than our rank. Between 0.910 and 0.920 the leaderboard collapses from
744 teams to 71 — that band is the **public-notebook plateau**, and thread 730924 names it
exactly: a clean single-seed TemporalUNet3D pipeline gives 0.908, a two-seed logit blend
0.910, and the public two-seed notebook 0.911. Roughly 670 teams are sitting on the same
stack. Real separation starts above 0.92.

For calibration against our own local numbers, the CV↔LB evidence is mixed but usable:

- 716952 (ISAKA): CV 0.682→LB 0.663, CV 0.824→LB 0.826 — near 1:1 for detection changes.
- 716952 (Timmy): CV 0.7448→LB 0.834, CV 0.8213→LB 0.846 — LB well *above* train CV.
- 730160 (Mendrika): "directionally yes, but public LB is more optimistic by almost 10%";
  movie-to-movie spread ±0.14, 18% CV, worst movie 0.460, best 0.984.

So our 0.5552 probably maps to something in the low 0.6s on the LB. Still bottom quintile.

---

## 2. Three things `MEMORY.md` has wrong

### 2.1 ⭐ This is a notebook-rerun competition, not a CSV upload

`MEMORY.md` line 11 says "**CSV-upload** contest, not notebook-runtime." That is backwards.
From `raw/competition.json`:

```
onlyAllowKernelSubmissions: True      usesSynchronousReruns: True
maxCpuRuntimeMinutes: 720             maxGpuRuntimeMinutes: 720
requiredSubmissionFilename: submission.csv
```

Confirmed repeatedly in the forum: internet is disabled during the scored rerun (716704,
727462), packages must be pre-staged as datasets rather than pip-installed (716704, 726955),
and a hidden test set is swapped in at rerun time (723921, 717228).

**Consequences for us.** The deliverable is not a CSV, it is a notebook that must detect and
link **~200 unseen volumes inside 12 hours** with no internet. Reported end-to-end rerun
times: 5–7 h typical (730149, 726526), 9–12 h (734237), and genuine timeouts at 12 h
(724914, 717228). Our `pipeline/classical.py` has never been costed against that budget, and
`harness/purescore.py` — built specifically to dodge the `tracksdata`/numpy conflict on
Kaggle — is a scoring-side tool that the rerun does not need at all. The runtime budget is a
hard constraint we have not yet treated as one.

### 2.2 ⭐ The "leak" is not a leak, and it was reported six weeks ago

`MEMORY.md` line 143 records: "🚩 **The leak is CONFIRMED.** `sample_submission.csv` names
exactly the four datasets whose ground truth ships in `train/`. Anyone can score ~1.0 by
echoing it. **Report to the organisers; do not exploit.**"

The four visible test clips are **debug placeholders**. LeeWhieldon reported the identical
finding on 2026-07-08 (723921, and again in 716062), and the organisers answered directly:

> "Indeed these are dummy placeholder files to help you ensure that your submission notebook
> actually produces a .csv file without erroring out. The actual leaderboard score is
> obtained from a much bigger test set, that is deliberately kept private, and I assure you
> there is no overlap between that hidden test set and the train set."
> — Thibaut Goldsborough (organiser), 2026-07-10, thread 716062

Corroborated by fnands (723921): the placeholders are replaced in submission mode, which is
why internet is disabled — "so you can't exfiltrate the real data."

**Consequences for us.** (a) No report to the organisers is owed; the issue is raised,
answered and closed. (b) Echoing `sample_submission.csv` scores 0.000, not ~1.0 — thread
732674 shows exactly that. (c) **`notebooks/03_linking.ipynb` is built on a false premise.**
`MEMORY.md` describes it as producing "a local score of the four test datasets that predicts
the LB before spending a submission." Those four datasets are not scored by anything. That
part of the notebook should be deleted, not debugged.

### 2.3 The competition is far bigger and further along than our notes assume

`MEMORY.md` line 10 says "~73 teams". Actual, from `raw/competition.json`: **2,397 teams,
2,590 competitors, 10,438 joined users, 32,949 submissions**. Public LB is 29% of the test
data (`leaderboardPercentage: 29`), so a 71% private shakeup is pending. Deadline
2026-09-29; team-merger and new-entrant deadlines 2026-09-22. Prize pool $60k over 7 prizes
— that part we had right.

---

## 3. Authoritative organiser answers

Collected from the threads; these are the load-bearing ones.

| Question | Answer | Source |
|---|---|---|
| Are the 4 test clips the real test set? | No — debug placeholders; hidden test is bigger and disjoint | 716062, 723921 |
| Test set composition | Two embryos, roughly train-sized, **no embryo_id overlap with train** | 716793 |
| Same protocol across embryos? | Yes — same instrument, same developmental stage; each embryo a separate imaging session | 724386 |
| Is Zebrahub public data allowed? | **Yes**, imaging *and* `*_tracks.csv`; no overlap with test | 734330 |
| Cell Tracking Challenge data allowed? | Organisers of CTC granted permission on request, citation required | 729057 |
| Self-trained weights as a private dataset? | Permitted; must be reproducible if you win | 724502 |
| Can score exceed 1.0? | Yes — max ~1.1, "theoretically possible to obtain slightly above" | 725015 |
| What preprocessing was applied? | Multi-view fusion, views linearly scaled to a reference; custom Tomer-2012-style microscope | 724582 |
| Metric source code | `royerlab/kaggle-cell-tracking-competition`, plus a CSV→geff script added after 716062 | 716062, 727154 |

Two of these are directly actionable and neither is in our notes: **Zebrahub is explicitly
allowed as external training data**, and **the test embryos are different embryos**.

---

## 4. Data facts the field found that we do not have

These are all measured claims from other competitors, not ours. Ranked by how much they
would change what we do.

### 4.1 ⭐ Frozen frames — a systematic acquisition artifact, 6bba only (724283)

g john rao byte-compared every consecutive frame pair in all 199 training videos:

```
Group     Videos   Affected   Duplicate pairs   Pair rate
44b6          71          0                 0        0%
6bba         128        114               947     7.47%
Total        199        114               947     4.81%
```

57.3% of all training videos contain at least one frozen transition; 89.1% of `6bba` videos
do. And it is *scheduled*: `6bba_05b6850b`, `6bba_07477033`, `6bba_5b28472a` all freeze
after exactly frames 4, 12, 27, 42, 52, 57, 59, 62, 66, 76 — implying they are crops from
one master volume. A second group shares a different schedule.

Why it matters twice over. First, hengck23's warning (724283): a model can memorise the
freeze schedule and fail if the hidden test freezes differently. Second, and worse, thread
729082 shows the naive exploit does not even work — in `6bba_fc516dc6` frames t=81 and t=82
are byte-identical yet the GT node moves **8.9 µm**, beyond the 7 µm matching radius. So
copying detections across a frozen pair can *fail to match*. Our recon never looked for this.

### 4.2 The ground truth has real errors, at low rate

- 729053 (hengck23): edge annotations are not all correct; error rate low, detectable by
  checking neighbouring cell motion.
- 732474 (Tim Krige): built a GUI, found GT disagreeing with careful human tracking; also
  GT missing splits.
- 726381: boundary cells at `y=0` are annotated as **two separate tracks** across a
  visibility gap, so a correctly-continuous track is charged FP+FN. Concrete case given in
  `44b6_12dfb391`.
- 732474 (xaxipiruli): the Zebrahub `*_tracks.csv` lineages are **Ultrack output, not manual
  curation** — pseudo-labels carrying Ultrack's own biases around divisions and dense
  regions. Relevant now that Zebrahub is confirmed legal.

### 4.3 Normalisation and I/O facts sitting in the shipped metadata (734053)

- Intensity quantiles ship in `zarr.json`: 0.0→15, 0.1→75, 0.9→497, 0.99→1478, 1.0→4319.
  90% of voxels sit at or below 497 — 11.2% of the range — so **min-max scaling puts 90% of
  the data below 0.112**. Clip at p99 before scaling. (733973 adds p99.9 = 2145.)
  Our `det_threshold=0.15` is applied to whatever normalisation `pipeline/classical.py`
  uses; if that is min-max, the threshold is landing in a badly compressed part of the range
  and its measured optimum is an artifact of the scaling, not of the data.
- Chunking is `(1, 64, 256, 256)` — one timepoint per chunk, blosc/zstd + bitshuffle. Reading
  a full timepoint touches 1 chunk; reading one z-slice across all time touches 100. **Any
  access pattern that walks time at fixed depth costs 100× the I/O.** Directly relevant to
  fitting 200 volumes in 12 h.
- José Freitas (732103) adds: the official pipeline downsamples XY by a **stride**
  (`vol[:, ::4, ::4]`), not a block mean — so the detector sees strided noise, not averaged.

### 4.4 The two embryos are wildly heterogeneous (732103 comment, Ace)

Annotation sparsity ~1% vs ~9%; division counts 26 vs 125 across their samples. With the
test set being two *different* embryos (716793), this is the generalisation axis that
decides the private leaderboard.

### 4.5 ⭐ The public checkpoints were trained on all 199 videos (730160)

Mark_RowSet: the released weights' `split_manifest.json` lists all 199 under `train`, and
the 40 in its own test list are inside that same set. Validating on train movies with those
checkpoints scores a model on data it memorised. His concrete cost:

> "I ablated the post-processing in the public pipeline and measured +0.0184 from turning one
> stage off... Submitted it and got 0.909 against my 0.912 baseline. The sign flipped."

His explanation is the useful part: corrective post-processing stages have nothing to repair
on memorised videos, so switching them off looks like a gain — **a training-set harness will
tell you to delete the components that matter most at test time.**

This does not currently bite us (`pipeline/classical.py` is training-free), but it is a trap
laid directly across the path we are on, since the obvious next step is to adopt the public
detector. Community consensus is **leave-one-embryo-out**, which means training our own.

---

## 5. Metric findings: what the forum confirms, and what it adds

Confirmed independently, matching `notes/02-metric-findings.md`:

- Over-detection is nearly free; cost is exactly `1 − 0.1 × over-prediction` (733877, 728300).
  733877 states the break-even cleanly: **10% more nodes needs only a 1% relative gain in
  edge Jaccard to pay for itself** — "most people's thresholds are too high."
- Under-prediction gives an uncapped bonus, so clean predictions score above 1.0 (728300,
  725015). Max ~1.1.
- Divisions do not pay if forced: 716952 measured LB 0.784 → **0.778** on adding division
  edges; 724323 measured 0.728 → **0.695** with `DETECT_DIV=True`.

Added by the forum, not in our notes:

- **Duplicate detections cost ~9% and buy nothing** (733877). Node matching is one-to-one, so
  a twin one voxel away matches nothing while still counting against the node budget. This
  directly contradicts `MEMORY.md` §1's "duplicate detections inside the 7 µm radius are free
  too." Consequence: **taking the union of two models' detections without a merge pass pays
  in full** — relevant the moment we ensemble.
- **A one-to-one linker forfeits the entire division term by construction** (733877): no node
  ever has two outgoing edges, so `division_jaccard` is exactly 0.000. That is 0.1 of the
  available 1.1 gone before the tracker sees an image. Our `pipeline/classical.py` uses
  `linear_sum_assignment` — we are in exactly this position.
- **`summarise()` drops the division term entirely when a submission contains no divisions**
  (734192, Arul Prasad). So a fork-free submission returns pure adjusted edge Jaccard, and
  the division term follows by subtraction — **an exact edge/division split for one
  submission**. That is a free diagnostic and we should use it.
- **The metric was patched 2026-07-17, commit `aa65e90`** (727154, 728324, 733877), closing
  three holes: divisions now require directed local topology rather than shared weak
  connectivity; edges spanning more than one frame are dropped; merged edges are collapsed.
  All submissions were rescored by 2026-07-23. **Any local CV number from before 17 July is
  not comparable with one from today** — we should confirm which scorer commit
  `harness/scorer.py` and `harness/purescore.py` were transcribed from.
- **Non-consecutive edges silently score 0.0, with no error raised** (728613). Diana Daher
  submitted a frame-strided pipeline, got exactly 0.0 twice, and traced it to `t=0→5` edges
  being structurally unmatchable. Fix: process a **contiguous block** of frames from t=0
  rather than striding. This is the single most likely way for a runtime-budget optimisation
  to destroy a submission silently, and it is exactly the shortcut we would reach for.

### The division exploit, for the record (727154, 714101, 723655)

An `augment_dataset` post-processing cell in a public notebook added a hub node at
`(t=-1000, z=y=x=-10000)` joining all track roots into one weakly-connected component, plus
synthetic fork chains at the same sentinel coordinates. Under the pre-patch matching this
pushed division Jaccard from ~0 to ~1.0, worth nearly the full +0.1. It is closed and now
*loses* score (733877 measures 0.9839→1.0639 old vs 0.9839→0.9794 new). Noted so we
recognise it if we encounter a fork of that notebook.

---

## 6. Idea inventory, ranked by expected value to us

**Tier 1 — cheap, evidenced, directly addresses our gap**

1. **Replace raw-intensity peak detection with multi-scale DoG/LoG scale-space maxima.**
   Our detector is a `maximum_filter` NMS on raw intensity (`pipeline/classical.py:124`).
   ISAKA (716952) measured single-scale DoG + Hungarian at LB 0.663, and **multi-scale DoG
   at 0.826** — the scale-space max alone was +0.040, and "detection was the biggest lever."
   José Freitas (732103) measured DoG recall at **0.91–0.94 on real annotated nuclei**. Our
   node recall is 0.845. This is the largest single evidenced gap and needs no training.
2. **Fix intensity normalisation before re-tuning any threshold** (734053). Clip at p99 using
   the shipped quantiles. Our `det_threshold=0.15` optimum was measured under whatever
   scaling we currently use and may not survive the fix — re-run the sweep after.
3. **Spend one submission on the fork-free split trick** (734192) to get our exact
   edge/division decomposition on the real hidden test, and one to establish that our
   notebook completes inside 12 h at all.
4. **Audit for non-consecutive edges before every submission** (728613). Cheap assertion,
   catastrophic failure mode.

**Tier 2 — structural, medium cost**

5. **Leave-one-embryo-out validation** (730160), replacing our 5-fold split over 199
   datasets. Two embryos, test is two *different* embryos; our current folds measure
   interpolation within seen embryos, which is not the question being scored.
6. **Divisions as a stage-2 problem, not a linker output** (726924, hengck23): learn tracks
   with one birth and one death, then decide splits in post-processing with appearance
   change and longer track cues — "it is difficult even for humans to decide if there is cell
   division based on two frames." This is the concrete path to the 0.1 our Hungarian linker
   currently forfeits by construction.
7. **Pre-train on external data, fine-tune on real** — Zebrahub is explicitly allowed
   (734330), CTC is permitted with citation (729057), and there is a free CC0 synthetic set
   with **165,267 labelled divisions** (732103), ~540× the mitosis supervision in the
   competition GT. Caveat from its own author: division rate deliberately inflated (4.07% vs
   ~0.26% real), texture/contrast the weakest axes, value is in the labels not the realism.
8. **Handle the frozen frames explicitly** (724283, 729082) — but note the GT can still move
   8.9 µm across a byte-identical pair, so copy-forward is not the answer.

**Tier 3 — ideas the field is exploring, unproven here**

9. Temporal affinity fields / flow-based linking (723655, hengck23) — predict a local vector
   field to guide the graph optimiser; build flow GT from optical flow + sparse tracks.
10. Reformulate linking as segmentation: render detections in a 3D `(z,y,x)` volume colour-coded
    by `t`, then a plain U-Net labels each point Start/Inside/End/Division — turning 4D conv
    into 3D conv (723655).
11. Learnable peak detection: softmax over a pixel's neighbourhood instead of binary
    classification, with unannotated pixels excluded from the loss — a direct answer to sparse
    supervision (723655).
12. Pseudo-labelling by consensus across several open-source trackers (ultrack, byotrack,
    trackastra, laptrack), feeding short 2–3 frame tracks into an ILP for long tracks (723655).
13. CoTracker-style joint tracking (726924) — but two competitors report it does not address
    the failure mode that costs points: Jawad Ahmed measured the residual errors as
    **short-range ambiguities inside a single frame pair**, not long-horizon drift. Bharat got
    it working on z-MIP 2D projections only.
14. HOCT, the hosts' own higher-order tracker (726521) — **tried and reported negative**
    (728551): over-links, produces spurious forks, ~45 min/movie, not submission-viable when
    fed synthetic ball masks instead of real morphology.
15. Node-count regression as a density-estimation problem, "like human head counting", to nail
    the adjusted-Jaccard numerator (725015, hengck23).

---

## 7. Hazards to avoid

- **Scoring 0.0 with a valid CSV** — non-consecutive edges (728613).
- **Timing out at 12 h** on ~200 hidden volumes (724914, 717228, 726526). Youri Matiounine
  (724917) settles the diagnosis: local scoring of all train data takes ~1 minute, so runtime
  is *prediction*, not scoring. Time your pipeline on the 4 visible clips and scale by ~50×
  (734237).
- **Validating a corrective post-processing stage on train movies with public weights** — the
  sign can invert (730160).
- **Merging two detectors' outputs without a dedup pass** — pays the ~9% duplicate cost
  (733877).
- **Trusting the public notebook plateau.** 744 teams sit at ≥0.910 and 71 at ≥0.920;
  mikelou1 (735352): "public notebooks are really overtuned to the lb." With 71% of the test
  set held back, a large shakeup is likely.
- **Comparing pre- and post-17-July local scores** (733877).

---

## 8. What I would do next, in order

1. Correct `MEMORY.md` on the three points in §2 — the submission model, the non-leak, and
   the scale of the field. These are cheap and they currently misdirect every downstream
   decision.
2. Delete the four-test-dataset scoring path from `notebooks/03_linking.ipynb`; it predicts
   nothing.
3. Swap the detector to multi-scale DoG and re-run the threshold sweep on p99-clipped
   intensities. On the public evidence this is worth roughly +0.25, and it needs no GPU.
4. Rebuild the harness folds as leave-one-embryo-out.
5. Cost the full pipeline against the 12-hour rerun budget before optimising score further —
   a 0.9 that times out scores nothing.
6. Only then take on divisions, as a stage-2 classifier over completed one-to-one tracks.

## 9. One open question I could not close

`MEMORY.md` §4 notes `test/` ships images only (4 `.zarr`, 0 `.geff`), and concludes "no node
budget at test time." That is right, but the adjusted-Jaccard multiplier still uses
`estimated_number_of_nodes` from the *hidden* ground truth, which we never see. So the node
budget applies at test time; we simply cannot read it. Whether the real hidden test exposes
any per-dataset size hint is not answered anywhere in the forum. Worth a direct question to
the organisers if we ever revisit `budget_fill`.
