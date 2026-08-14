# Recon results — the first real numbers

`01_recon.ipynb`, run on Kaggle 2026-08-14. All 199 train datasets, no images loaded.
Raw output in `recon_summary.json`.

---

## 0. The thing to check before anything else

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

---

## What this changes

1. **Detection recall is the entire game.** Build the best detector; do not spend time on
   clever linking. Nearest-neighbour or a simple radius-limited assignment is already ~99 %
   of the achievable linking score.
2. **Detect aggressively** — now supported by measurement, not just metric structure. The
   node budget cannot justify holding back.
3. **Linking radius ≈ 8–10 µm**, from the p99 of 8.38 µm. Not 7 (the metric cutoff), not 25.
4. **Skip divisions.** 1.17 per 1,000 edges.
5. **Z centroid accuracy still matters most** — median true |dZ| is *zero*, so any Z error
   we introduce is pure noise against a signal that does not move in Z between frames.
6. **Watch the two embryos separately.** `44b6_` and `6bba_` differ enough in density and
   border behaviour that a pooled number can hide a regression in one.
