# The leaderboard, and the detector we should have been using

Source: the discussion archive on branch `claude/scrape-competition-discussions-f865sg`,
merged into `discussions/`. It has what my own scrape could not reach — the **71 opening
posts** and a **full leaderboard snapshot** (`discussions/raw/leaderboard.json`).

Two things in it force corrections to `notes/06` and `notes/07`, and one gives us a
concrete, measured path from 0.5552 to ~0.82.

---

## 1. 🚨 The leaderboard is far more compressed than I assumed

2,402 scored teams:

| | score |
|---|---|
| rank 1 | 0.950 |
| top 1 % (rank ~25) | 0.930 |
| top 5 % (rank ~121) | 0.916 |
| top 10 % (rank ~241) | 0.915 |
| top 25 % (rank ~601) | 0.913 |
| **median (rank ~1,202)** | **0.890** |
| top 75 % | 0.746 |

Medal cutoffs: **gold 0.935** (14 teams), **silver 0.916** (105), **bronze 0.915** (120).

A quarter of the field sits between **0.913 and 0.916** — that is a fork cluster, everyone
running the same public learned notebook. The leaderboard is not a smooth gradient of
skill; it is a cliff at the public baseline.

Where our work actually lands:

| | score | rank | percentile |
|---|---|---|---|
| our CV 0.5552 | 0.555 | ~2,004 | bottom 17 % |
| our CV + 0.09 (the optimistic mapping) | 0.645 | ~1,919 | bottom 20 % |
| the rule-based frontier, 0.846 | 0.846 | ~1,529 | bottom 36 % |
| the learned public pipeline, 0.915 | 0.915 | ~139 | top 5.8 % |

### ❌ Correction to `notes/07` §4

I wrote that our 0.5552 "probably maps to roughly 0.60–0.65 LB — behind the frontier, but
the gap to the 0.834 heuristic is a tuning gap, not an architecture gap", and that
**"rule-based is genuinely competitive"**. Both need withdrawing.

- **"7th of 344 teams"** was posted **2026-07-01** at **LB 0.826**. There are now 2,402
  teams and 0.826 would rank about 1,600th. The claim was true when written and is stale
  now.
- **"LB runs above CV by ~10 %"** came from two competitors. The author of the rule-based
  pipeline — the one closest in shape to ours — measured the opposite and published the
  scatter: **CV ≈ LB, roughly 1:1**, for detection changes. Their table below shows
  0.682/0.663, 0.791/0.786, 0.824/0.826. For a pipeline like ours, **CV is the estimate**,
  so 0.5552 means about 0.555, not 0.65.

The honest read: we are in the bottom fifth, and the gap to the pack is an
**architecture** gap, not a tuning gap.

## 2. ⭐⭐ The rule-based author's measured ladder — and it is all detection

From the opening post of *"Rule-based is surprisingly strong?"* (which my API scrape could
not retrieve), all rule-based, no learning:

| # | method | CV (edge) | LB |
|---|---|---|---|
| 1 | DoG-blob detection + Hungarian linking | 0.682 | 0.663 |
| 2 | tuned DoG scales + 8 µm linking | 0.791 | 0.786 |
| 3 | + gap closing | 0.807 | 0.784 |
| 4 | + division edges | 0.810 | 0.778 |
| 5 | **multi-scale DoG** | **0.824** | **0.826** |

Read that top row again: **their very first version, plain DoG plus Hungarian, scored CV
0.682 against our 0.5552.** Same linker, same match radius, same metric. The entire
difference is the detector — ours thresholds raw normalised intensity, theirs runs a
Difference-of-Gaussians.

Their own summary: *"Detection was the biggest lever. Just taking DoG at multiple scales
and using the scale-space max jumped LB from 0.786 → 0.826 (+0.040)."*

Two of their rows also settle experiments we had open:

- **Gap closing: roughly neutral** (CV +0.016, LB −0.002). I called it "the highest-value
  repair available" in `notes/06` §4. Downgrade that — it stays implemented and off.
- **Division edges: actively harmful** (LB 0.784 → 0.778). Confirms recon §6 and §10 more
  strongly than our own reasoning did.

And a data point that bears directly on the pretraining plan: *"I also tried a learned 3D
U-Net, but with only 2 embryos it didn't generalize, so I set it aside."*

## 3. What DoG does that an intensity cut cannot

Implemented as `Config.detector="dog"` (`detect_frame_dog`), ported from their published
kernel. Three differences, each with a reason:

1. **Band-pass instead of absolute intensity.** DoG responds to blob-scale structure and
   suppresses the smooth background, so a dim nucleus sitting on a bright region still
   peaks. An absolute cut cannot separate those. This is precisely the 15.5 % of GT nodes
   `notes/05` §1 found the threshold could no longer reach.
2. **Per-frame normalisation** (percentiles 1.0 / 99.7) instead of whole-movie zarr
   quantiles, so bleaching does not drag the operating point through the movie.
3. **Scale-space maximum** over several sigma pairs, catching nuclei of different sizes —
   recon noted cells with Z spans over 24 µm alongside ordinary ones.

Plus a smaller fix that came with it: the non-maximum window is now an **ellipsoidal ball**
in physical space, not a box. A box reaches `r` along the axes but `r√3` into the corners,
and on a 4:1 anisotropic grid that error is not symmetric between Z and XY.

Measured on synthetic volumes with dim nuclei on a bright gradient, **at a matched
detection budget** — which is the only fair comparison, since the node budget charges for
every detection:

| detector | detections | dim-nucleus recall |
|---|---|---|
| intensity | 11 (capped) | **0.17** |
| **DoG multi-scale** | 11 | **0.67** |
| intensity | 153 (uncapped) | 0.67 |

Four times the recall per detection spent, or 14× fewer detections for the same recall.
That is the mechanism working exactly as claimed.

## 4. Independent confirmations from the synthetic-dataset author

The `[Free Dataset] 18.5 GB` opening post contains measurements that cross-check ours:

- **median step 1.86 µm/frame** — recon §3 measured 1.82 µm independently. 
- **annotated fraction ~2.8 %** — recon measured a median of 3.56 %. 
- **the official pipeline downsamples XY by a stride, `vol[:, ::4, ::4]`, not a block
  mean** — which is exactly what our `downsample=(1,4,4)` does. Confirmed correct.
- **a classical DoG detector recovers 0.91–0.94 of the real annotated nuclei.** Our
  `node_recall` is 0.845. So roughly **0.07–0.10 of recall is sitting in the detector
  swap**, before any tuning.
- sister separation 7.24 µm; lag-1 directional persistence +0.30, so a velocity prior has
  real signal for the linker.
- ~304 division events across the 199 videos (recon counted 151 dividing *nodes* — 304 is
  the edge count, 2 per division; consistent).

Dataset and generator: <https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset> ·
explorer: <https://www.kaggle.com/code/josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking>

---

## What to do

1. **Swap the detector.** `detector="dog"` is implemented and tested but **unmeasured on
   real data**. It is the largest known gain available: a directly comparable pipeline
   scored 0.682 where we score 0.555, and 0.824 with multi-scale. Run it through the
   harness at `fold_by="embryo"` and gate it against 0.5552.
2. **Then tune scales**, since their multi-scale step alone was +0.040.
3. **Leave gap closing and divisions off** — both measured neutral-to-harmful by someone
   who tried them on this exact metric.
4. **Read CV as LB.** For a rule-based pipeline the mapping is ~1:1, measured across five
   paired submissions. Stop hoping for a +0.09 bonus.
5. **The pretraining plan is aimed at the right wall.** The rule-based author abandoned a
   3D U-Net because two embryos were not enough to generalise; that is exactly the
   constraint external pretraining removes. But note the ceiling: a *tuned* rule-based
   pipeline tops out near 0.826, which is still bottom-40 % of today's board. Reaching the
   0.915 cluster needs the learned detector; reaching past it needs the learned detector
   to be better than the one everybody forked.
