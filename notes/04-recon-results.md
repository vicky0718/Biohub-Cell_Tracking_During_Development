# Recon results — the first real numbers

`01_recon.ipynb`, run on Kaggle 2026-08-14. All 199 train datasets, no images loaded.
Raw output in `recon_summary.json`.

---

## 0. ~~The thing to check before anything else~~ — RESOLVED, no leak

> **CLOSED 2026-08-16 (`notes/07-forum-intel.md` §2).** The host answered this on the
> forum: the visible `test/` files are **dummy placeholders** so competitors can check
> their submission notebook produces a CSV without erroring. The leaderboard scores a
> *much bigger private test set* with **no overlap with train**. Reading 1 below was
> right. Nothing to report, nothing to exploit. The section is kept for the record.


**All 4 test dataset names also appear in the train list**, and we have ground truth for
all four:

| dataset | in train? | GT nodes | GT edges |
|---|---|---|---|
| `44b6_0113de3b` | yes | 52 | 50 |
| `44b6_0b24845f` | yes | 51 | 49 |
| `6bba_05b6850b` | yes | 861 | 845 |
| `6bba_05db0fb1` | yes | 1,229 | 1,183 |

These are opaque hashes; a collision is not plausible. So the visible `test/` folder holds
the same four videos as `train/`, whose annotations we already have.

**Do not act on this yet.** Two readings, and they lead to opposite strategies:

1. The visible `test/` is a *worked example* — a handful of videos so competitors can build
   and validate a submission — and the leaderboard scores against hidden data. Then this is
   a convenience, nothing more.
2. The visible `test/` is the actual scored set. Then the public leaderboard is trivially
   saturated by echoing train GT, which would be a competition-breaking flaw the organisers
   would want to know about.

**Resolve it by reading `sample_submission.csv`** (which datasets does it name?) and the
competition's data description / discussion. If reading 2 holds, report it to the
organisers rather than quietly exploiting it — and note it says nothing about how a hidden
private set would score. Either way, this does not change the modelling work below.

> **Section 0 of `02_classical_baseline.ipynb` now does this check on the Kaggle mount** —
> it lists the folders, reads `sample_submission.csv`, and prints the overlap. It also
> answers the other open question the notebook needs: whether `test/` ships `.geff`
> metadata, which is where the per-dataset node budget of §9 comes from. If it does not,
> the budget cap has nothing to key off on the datasets we are scored on.

## 1. Inventory

- **199 train datasets, 4 test.** Every one is `T=100`, `(Z,Y,X) = (64, 256, 256)`.
- Two embryos, by name prefix: **`44b6_`** (71 datasets) and **`6bba_`** (128).
- **No `dataset_splits.json`** ships with the data — the harness's deterministic fold
  fallback is what we use.
- Totals: **133,318 annotated nodes, 128,883 edges, 151 divisions.**

The two prefixes behave differently and should be treated as separate regimes:

| | `44b6_` | `6bba_` |
|---|---|---|
| annotated fraction | ~0.001–0.05 | ~0.01–0.20 |
| annotated nodes/dataset | 50–1,350 | 200–1,950 |
| mid-movie track ends at volume border | ~100 % | ~40–90 % |

## 2. Annotation density — the sparsity is extreme

**Annotated fraction: min 0.0013, median 0.0356, max 0.2021.** About **1 cell in 28** is
annotated in the median dataset; in the worst, 1 in 770.

`estimated_number_of_nodes` ranges 3,783 → 78,644 per dataset (≈38–786 cells per frame),
against 50–1,950 annotated nodes. The node budget is one to two orders of magnitude larger
than the ground truth.

This is the measured version of the metric finding: the scorer only ever sees edges
touching that ~3.5 % minority.

## 3. Motion — and what the linking radius should be

Pooled over 128,883 GT edges:

| p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|
| 1.82 µm | 2.73 | 4.14 | 5.34 | 8.38 | 60.76 |

Only **2.1 %** of true links move further than the 7 µm match cutoff.

Component split: **median |dZ| = 0.000 µm** (exactly zero — under one Z voxel, so Z motion
is quantised away by the 1.625 µm grid), median |dXY| = 1.22 µm. p99 |dZ| = 6.50 µm = 4 Z
voxels.

**Linking radius: ~8–10 µm** covers p99 with margin. The domain note's guess of 8 µm was right.

## 4. Cell spacing — the annotated set is sparse in space too

Median nearest-neighbour distance between annotated cells: **24.99 µm**, with only 0.62 %
of pairs closer than the 7 µm match radius. Since ~1 cell in 28 is annotated, true cell
spacing is roughly `25 / 28^(1/3) ≈ 8 µm` — consistent with the ~6–17 µm nucleus estimate.

## 5. ⭐ Nearest neighbour is almost always the right link

**The nearest annotated cell in `t+1` is the true successor 99.80 % of the time**
(128,470 / 128,732). Rank histogram: `{1: 128470, 2: 250, 3: 9, 4: 2, >5: 1}`.

## 5b. Two of my predictions were wrong

Both sparse-annotation biases predicted from the dual-channel labeling story failed:

- **Clonal clumping: NOT FOUND.** Observed vs uniform-null NN spacing came out **~uniform
  or *dispersed*** in the overwhelming majority of datasets (ratios 0.9–1.5); only 5 of
  ~120 scored as CLUMPED. Annotated cells are if anything *more* spread out than a uniform
  random draw. The clonal-inheritance argument does not survive contact with the data.
- **Division enrichment: NOT FOUND — the opposite.** Only **151 divisions in 128,883 edges
  = 1.17 per 1,000**, i.e. **0.117 %** of annotated nodes divide. Divisions are *rare*, not
  over-represented.

**Depth bias: real, and stronger than expected — but not the shape predicted.** Annotations
are not spread through Z; in most datasets they occupy a *slab*, with whole deciles at
exactly 0.00 (e.g. `44b6_40c45f5a`: `0 0 0 .25 .68 .07 0 0 0 0`). Some are extreme
(`6bba_b1ae37b9`: median z = 5 / 64). This is not a gentle attenuation gradient — it is a
hard restriction of the annotated region, and it varies per dataset in a way we cannot
observe at test time.

## 6. Divisions are not worth chasing

1.17 divisions per 1,000 GT edges. The term is capped at 0.1 of the score and a mistimed
division costs more edge Jaccard than it earns. **Ignore divisions entirely** until
everything else is done.

## 7. ⭐⭐ The linking-only ceiling — detection is essentially the whole contest

Perfect detections (GT nodes fed back in), scored on all 199 datasets:

| linker | edge Jaccard | TP / FP / FN | adj edge Jaccard |
|---|---|---|---|
| **Hungarian** (optimal assignment) | **0.9915** | 128,178 / 392 / 705 | 1.0825 |
| **greedy** (nearest neighbour) | **0.9847** | 128,556 / 1,674 / 327 | 1.0750 |

**Optimal assignment beats greedy by only +0.0068.**

Two conclusions, and they are the most decision-relevant numbers we have:

1. **Given the detections, tracking is solved.** Plain nearest-neighbour linking is within
   0.7 % of optimal assignment, and optimal assignment is within 0.9 % of perfect. There is
   at most ~0.015 of edge Jaccard available in the entire linking problem. An ILP,
   Ultrack, a learned association transformer — all of them are competing for that sliver.
2. **Everything else is detection.** Whatever the leaderboard shows, the gap between it and
   ~0.99 is almost entirely cells not found.

Note also `adj_edge_jaccard = 1.0825 > 1` — the uncapped under-prediction bonus from the
metric findings, confirmed on real data, because 133k GT nodes sit far below the ~5M total
node budget.

### The budget bonus does *not* justify under-detecting

Tempting, and wrong. Annotated cells are a random subset, so detecting a fraction `f` of
all cells catches fraction `f` of the annotated ones; an edge needs **both** endpoints, so
edge recall goes as roughly `f²`. The budget multiplier is `1 + 0.1(1 − f)`. Maximising
`f²·(1 + 0.1(1−f))` gives `dJ/df = 2.2f − 0.3f² > 0` for all `f ≤ 1`.

**The optimum is f = 1. Detect everything.** The bonus is far too weak to pay for lost recall.

## 8. What the annotation actually *is*: a handful of long tracks

Derived from the inventory (`tracks = nodes − edges + divisions`, exact for a forest).

**4,586 annotated tracks across 199 datasets — a median of 21 per dataset.**

| | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| tracks per dataset | 1 | 6 | **21** | 33 | 83 |
| nodes per track | 12.0 | 24.7 | **35.3** | 52.0 | 100.0 |
| annotated cells per frame | 0.7 | 2.9 | **6.6** | 9.4 | 20.1 |

And again the two embryos diverge sharply:

| | `44b6_` | `6bba_` |
|---|---|---|
| tracks per dataset (median) | **4** (range 1–25) | **30** (range 6–83) |
| annotated nodes (median) | 214 | 826 |

This reframes §2. The sparse GT is **not** a random 3.5 % of cells sampled independently
per frame — it is a few dozen cells followed for a long time. Median track length is 35
frames of a 100-frame movie, and only 8 of 199 datasets have tracks that typically run the
full length. Consequences:

- **Errors are correlated, not independent.** Lose one cell to a detection miss and you
  lose a contiguous run of its edges, not one scattered edge. Per-dataset scores therefore
  have much fatter tails than an independent-sample model suggests.
- **The "random subset" assumption behind the `f²` argument in §7 is about *which cells*
  get annotated, and that still looks unbiased in XY — but it is now clearly *not* a
  per-frame independent draw.** The `f²` scaling holds; the variance around it does not.
- **156/199 datasets have GT spanning all 100 frames**; 29 start after `t=0` and 25 end
  before `t=99` (shortest span 40 frames). Frames outside the annotated span contribute
  nothing at all — every predicted edge there is free.

## 9. ⭐⭐⭐ The test set is really *two* datasets — and one is a node-budget trap

The organisers' own aggregation (`metrics.summarise`, read from source): `edge_jaccard`
and `division_jaccard` are **micro**-averaged (TP/FP/FN summed, then Jaccard), while
`adj_edge_jaccard` — the term that actually scores — is the per-dataset adjusted Jaccard
**weight-averaged by `w = TP + FP + FN`**. So each dataset's influence is proportional to
its own edge denominator, ≈ its GT edge count for any sane prediction.

| dataset | GT edges | **weight** | node budget | cells/frame |
|---|---|---|---|---|
| `44b6_0113de3b` | 50 | 2.4 % | 25,755 | 258 |
| `44b6_0b24845f` | 49 | 2.3 % | 32,795 | 328 |
| `6bba_05b6850b` | 845 | **39.7 %** | **6,362** | **64** |
| `6bba_05db0fb1` | 1,183 | **55.6 %** | **69,800** | **698** |

Two things follow, and the second is the most actionable number in this document.

1. **The `44b6_` test datasets are noise.** 2 annotated tracks each, ~50 edges, 4.7 % of
   the weight combined. Do not tune on them. The leaderboard is ~95 % the two `6bba_` ones.
2. **The two datasets that *are* the leaderboard have node budgets 11× apart** — 64 vs 698
   cells per frame in identically-sized volumes. A detector with one global threshold
   cannot serve both. Detect at `6bba_05db0fb1`'s density on `6bba_05b6850b` and you
   predict ~70,000 nodes against a 6,362 budget: ratio 10.0, multiplier
   `max(0, 1 − 0.1·10.0) = **0.0**`. That dataset's adjusted Jaccard goes to **zero**, and
   it is 39.7 % of the score. The leaderboard number halves regardless of how good the
   tracking was.

The spread is not peculiar to the test set — across all 199 datasets the budget runs
**3,783 → 78,644 (20.8×)**, median 17,909:

| p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|
| 5,534 | 7,436 | 17,909 | 32,681 | 57,292 |

Simulating a detector that emits a fixed node count everywhere:

| fixed `N_pred` | median multiplier | p10 | datasets zeroed |
|---|---|---|---|
| 17,909 (median) | 1.000 | 0.776 | 0 / 199 |
| 32,681 (p75) | 0.918 | 0.509 | 0 / 199 |
| 57,292 (p90) | 0.780 | 0.065 | **16 / 199** |

**The detector must adapt its output count to local density.** Concretely: read
`estimated_number_of_nodes` from the dataset's own GEFF metadata — it ships with the
data, for test too — and cap detections per frame at roughly `est_total / T`. This turns
`Config.max_per_frame` from a nicety into a required guard.

### This does not contradict "detect everything"

§7 argued `f = 1` is optimal, and it still is. That argument was about **recall** — what
fraction of *true* cells you find. This section is about **precision** — how many nodes you
emit that aren't cells. With recall `f` and precision `p`, `N_pred ≈ f·N_total/p`, so

```
score ≈ f² · (1 − 0.1·(f/p − 1))
```

At `f = 1`: `p = 1.0 → ×1.00`, `p = 0.5 → ×0.90`, `p = 0.2 → ×0.60`, `p = 0.1 → ×0.10`,
`p ≈ 0.09 → ×0`. Precision is cheap down to ~50 % and fatal below ~10 %. Metric finding §1
("FPs are nearly free") is true of the *Jaccard* and false of the *multiplier* — the
multiplier is the only thing that ever charges for a spurious detection, and it charges
in proportion to how sparse the crop truly is.

## 10. Hedging divisions is dead — the arithmetic, on real numbers

`notes/02-metric-findings.md` left an untested hypothesis: hedge an ambiguous link across
both allowed children, on the theory that the downside is bounded by the 0.1 division term.
It isn't, and the recon numbers kill it.

Adding `k` speculative edges of which `m` are true changes `J = T/D` (with `D = T+F+N`) to
`(T+m)/(D+k−m)`. That beats `J` exactly when

```
m / k  >  J / (1 + J)
```

— the same rule as before, but the constant is set by *where you already are*. At the
linking ceiling `J = 0.9847`, the bar is **m/k > 0.496**: a hedged second child must be
correct **half the time**. A second child is correct when the cell actually divided, which
is **0.117 %** of annotated nodes (§6), and the second-nearest neighbour is the true
successor in 250 of 128,732 cases = **0.19 %** (§5). Off by two and a half orders of magnitude.

Hedging every link at the ceiling: TP 128,556→128,806, FP 1,674→~129,700,
**J 0.985 → ~0.497**. Hedging only the most ambiguous 1 %, optimistically recovering half
the rank-2 cases: **J → 0.977**, still below 0.985. There is no fraction that pays.

The one thing that softens the blow — and it is not enough — is that division FPs are
*scoped like edges*. From `division_metrics._pred_division_fork_sets`: a predicted fork
counts as evaluable only if the forking node **matched a GT node with out-degree ≥ 1**.
Forks on unannotated cells are invisible to the division term. But the extra *edges* are
still `pred_valid` whenever either endpoint matched, so the edge term charges for them.
**Never emit a second child** until there is a division classifier with >50 % precision,
which is a much later problem than detection.

Related correction to the `~⅓` figure in the metric notes: the threshold `J/(1+J)` was
quoted at `J ≈ 0.5` (synthetic data). Evaluate it at your *actual* `J`. Early on, with
detection recall poor and `J ≈ 0.4`, the bar for a first link is ~0.29 — link aggressively.
The bar for a *second* link is the same number, but the probability being tested is
"did this cell divide", not "is this the successor", and that probability is 0.001.

---

## What this changes

1. **Detection recall is the entire game.** Build the best detector; do not spend time on
   clever linking. Nearest-neighbour or a simple radius-limited assignment is already ~99 %
   of the achievable linking score.
2. **Detect aggressively** — now supported by measurement, not just metric structure. The
   node budget cannot justify holding back on *recall*.
3. **But cap detections per frame at `est_total / T`, per dataset** (§9). The node budget
   varies 20.8× across datasets and 11× between the two test datasets that carry 95 % of
   the weight. A fixed global threshold zeroes out the sparse crops. This is the one place
   where a spurious detection is not free, and it is where the leaderboard is decided.
4. **Linking radius ≈ 8–10 µm**, from the p99 of 8.38 µm. Not 7 (the metric cutoff), not 25.
5. **Skip divisions, and never hedge a second child** (§10). A hedged link must be right
   >49.6 % of the time to pay; the second-nearest neighbour is right 0.19 % of the time.
6. **Z centroid accuracy still matters most** — median true |dZ| is *zero*, so any Z error
   we introduce is pure noise against a signal that does not move in Z between frames.
7. **Watch the two embryos separately.** `44b6_` and `6bba_` differ enough in density and
   border behaviour that a pooled number can hide a regression in one — and the test set is
   effectively `6bba_` only, so a `44b6_`-driven pooled gain is not evidence.
8. **Validate on datasets that look like the test set.** Weight local validation the way
   the scorer does (by `TP+FP+FN`), and check the two budget regimes — very sparse
   (~64 cells/frame) and very dense (~700) — separately.
