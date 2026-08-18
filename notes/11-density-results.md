# Experiment #4 — the cap works, the gate rejects it, and it is the wrong instrument

`05_density.ipynb`, run on Kaggle 2026-08-17, 3.9 h on the fixed 60-dataset subset.
Folds verified leave-one-embryo-out for the first time: **fold 0 = `44b6` (21), fold 1 =
`6bba` (39)**. Subset node budget: 1,299,232.

| arm | SCORE | edge_J | recall | nodes | ratio |
|---|---|---|---|---|---|
| **sep4.5 + cap** | **0.6957** | 0.6931 | 0.885 | 1,176,957 | −0.051 |
| sep5.0 + cap | 0.6953 | 0.6914 | 0.871 | 1,115,344 | −0.075 |
| sep5.5 + cap | 0.6943 | 0.6894 | 0.857 | 1,058,422 | −0.098 |
| sep6.0 + cap | 0.6907 | 0.6850 | 0.847 | 1,015,538 | −0.116 |
| sep6.5 + cap | 0.6876 | 0.6809 | 0.831 | 958,435 | −0.141 |
| sep3.5 + cap | 0.6781 | 0.6768 | 0.899 | 1,243,919 | −0.027 |
| sep6.0 no cap (`04` champion) | 0.6760 | 0.7053 | 0.857 | 1,160,180 | +0.239 |
| sep4.5 no cap | 0.6671 | **0.7195** | 0.904 | 1,444,570 | +0.528 |
| 3 scales, density-matched | 0.6473 | 0.6426 | 0.786 | 920,952 | −0.138 |

**Best is 0.6957**, past the 0.682 the public rule-based pipeline reached with plain DoG.
Running total: 0.5790 (intensity) → 0.6760 (DoG) → 0.6957.

---

## 1. Prediction 1: the cap pays pooled, and the gate rejects it

| | pooled | `44b6` (fold 0) | `6bba` (fold 1) |
|---|---|---|---|
| cap at sep 4.5 | **+0.0286** | **−0.0286** | +0.0389 |
| cap at sep 6.0 | **+0.0147** | **−0.0205** | +0.0211 |

**REJECT on both.** The cap helps one embryo and hurts the other, by almost exactly
opposite amounts.

This is the first time leave-one-embryo-out has actually been in force, and it caught
something the five-way hash split would have hidden: a change whose sign depends on which
embryo you are looking at. The hidden test set is a **different pair of embryos**
(`notes/07` §3), so we have no basis for assuming it behaves like `6bba` rather than
`44b6`. That is precisely the risk the gate exists to catch, and `05` is the run where it
earned its keep.

### ❌ …and then my notebook used the cap anyway

The sweep cell picks `USE_CAP` by comparing **pooled scores** — `results["sep4.5_cap"].score
> results["sep4.5_nocap"].score` — and ignores the `gate()` verdict printed immediately
above it. So every arm in §2 and §3 ran with a setting the gate had just rejected.

It does not invalidate the separation sweep, which is an internally consistent comparison
at fixed cap. But the notebook drew a conclusion the harness had explicitly refused, which
is exactly the failure mode the gate was built to prevent. Fixed in `06`: the selector
reads the verdict, not the pooled number.

## 2. Prediction 2: an interior optimum — but it is a plateau, not a peak

CONFIRMED: 3.5 → 0.6781, **4.5 → 0.6957**, 5.0 → 0.6953, 5.5 → 0.6943, 6.0 → 0.6907,
6.5 → 0.6876.

The peak is interior, so the prediction holds. But **4.5 / 5.0 / 5.5 span 0.0014** — far
inside the ±0.14 per-movie noise the forum reports. Treating 4.5 as "the answer" would be
over-fitting a flat surface. The honest statement is that separation belongs in **4.5–5.5
µm**, and only 3.5 is clearly wrong (too dense: recall 0.899 but edge_J collapses to
0.6768).

## 3. Prediction 3: three scales is decisively worse — FALSIFIED

0.6473 vs 0.6957, **−0.0484, regressing both folds**. Not a density artefact this time:
the arm got its own calibration, and to match 265 detections/frame the third scale forced
`min_separation` up to **7.5 µm**, which crushed recall to 0.786.

Adding the finer `(1.0, 3.0)` pair generates a mass of small-scale responses that then have
to be suppressed by a much larger window, and the window takes real nuclei with it. Two
scales is the answer; the `04` verdict on multi-scale (2 beats 1, +0.0353) stands and 3
does not extend it.

## 4. ⭐ The cap is the wrong instrument for the job

What the cap actually trades, at sep 4.5:

| | no cap | cap | Δ |
|---|---|---|---|
| edge Jaccard | 0.7195 | 0.6931 | **−0.0264** |
| budget multiplier | 0.9472 | 1.0051 | **+0.0579** |
| score | 0.6671 | 0.6957 | +0.0286 |

It buys the multiplier by **destroying edge quality**. That is inherent to how it works: it
truncates detections per frame by intensity rank *before* linking, so it has no way to tell
a spurious peak from a real cell and removes both.

**`sep4.5` without the cap still has the best detection+linking quality we have ever
measured — edge Jaccard 0.7195.** It is only 145,338 nodes (10.1 %) over budget. If those
nodes could be removed without touching a single edge, the score would be **0.7195** — a
further **+0.0238** over the current best.

There is already a tool for exactly that, implemented in `notes/06` and **never once
measured**: `prune_isolated_nodes`, which drops detections that ended up with **no edge at
all**. Those nodes cost budget and contribute nothing. Pruning is *informed* — it acts
after linking, on evidence — where the cap is *blind*.

It is not provably free (node matching is a bipartite assignment, so a pruned node may be
winning a match a linked node would otherwise take), which is why it must be measured. But
the mechanism is right and the target is precise: remove ~10 % of nodes, keep the edges.

## 5. Where the remaining classical headroom is

The public rule-based ladder (`notes/08` §2) against ours:

| | theirs | ours |
|---|---|---|
| plain DoG + Hungarian | 0.682 | — |
| **tuned DoG scales + 8 µm linking** | **0.791** | — |
| multi-scale DoG | 0.824 | — |
| our best | | **0.6957** |

We are past their plain-DoG rung and short of their *tuned-scales* rung by **~0.10**. Our
scales — `(1.5, 4.0)` and `(2.5, 6.0)` — are their published **defaults**, not their tuned
values. Scale tuning is the largest classical lever left, worth roughly four times what
pruning is.

---

## What to do next

1. **Measure `prune_isolated_nodes`** on `sep4.5` without the cap. Precise target: strip
   ~10 % of nodes without losing edges, for up to +0.024. The informed alternative to the
   cap the gate rejected.
2. **Tune the DoG scales.** ~0.10 sits between our configuration and the public tuned one,
   and it is the same detector.
3. **Do not chase the separation value.** 4.5–5.5 is a plateau; pick 5.0 as the centre and
   stop.
4. Standing: still no leaderboard score and no submission notebook. At CV ≈ LB, 0.6957 is
   about rank 1,850 of 2,402 against a median of 0.890. The classical ceiling in public
   evidence is ~0.85, so **this path cannot reach the pack** — it is building the honest
   baseline the pretraining work has to beat.
