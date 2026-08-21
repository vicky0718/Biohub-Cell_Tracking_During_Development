# Phase 0b — at matched count the win mostly evaporates, and the reason is data

`claude_detector_matched`, Kaggle 2026-08-21, 2,423 s on a P100. Same data, same model,
same seeds as phase 0. Two changes: `threshold=1e-6` so the **cap** decides the count, and
25 epochs instead of 12.

## At genuinely matched count (1.0× DoG's per-frame count)

| train on | eval on | loss | recall | DoG | delta |
|---|---|---|---|---|---|
| 6bba | 44b6 | **masked** | 0.8615 | 0.7696 | **+0.0919** |
| 6bba | 44b6 | pu | 0.8275 | 0.7696 | +0.0579 |
| 44b6 | 6bba | pu | 0.8721 | 0.8776 | −0.0055 |
| 44b6 | 6bba | masked | 0.8621 | 0.8776 | −0.0155 |
| 6bba | 44b6 | naive | 0.6890 | 0.7696 | −0.0806 |
| 44b6 | 6bba | naive | 0.7936 | 0.8776 | −0.0840 |

**Only two of six arms beat DoG.** Phase 0's headline — "every learned arm beats DoG" —
does not survive the control it claimed to have.

## 1. 🚨 The notebook graded the wrong predictions

I rewrote the pre-registered predictions in the header and **left the summary cell
checking the previous notebook's three**. So the printed verdicts (P1 CONFIRMED, P2
CONFIRMED, P3 FALSIFIED) answer questions this notebook did not ask. Graded by hand
against what the header actually declared:

| # | claim | verdict |
|---|---|---|
| 1 | at 1.0× cap **every** learned arm beats DoG | **FALSIFIED** — 2 of 6 |
| 2 | `naive`'s directional split narrows | **CONFIRMED** — gap 0.2084 → **0.0034** |
| 3 | recall per detection falls as the cap rises | **CONFIRMED** — every arm, monotone |

A pre-registered prediction that is never actually graded is not pre-registration. Fixed
by generating the summary cell from the same list the header renders, so they cannot drift
again.

## 2. What actually changed from phase 0, and what I cannot attribute

| train | loss | phase 0 delta | phase 0b delta |
|---|---|---|---|
| 6bba | masked | +0.0400 | **+0.0919** |
| 6bba | pu | +0.0447 | +0.0579 |
| 6bba | naive | +0.1228 | **−0.0806** |
| 44b6 | pu | +0.0116 | −0.0055 |
| 44b6 | masked | +0.0062 | −0.0155 |
| 44b6 | naive | −0.0856 | −0.0840 |

**Phase 0b changed two things at once** — the threshold *and* the epoch count — so the
phase-0-to-0b difference is not attributable to either. That is a second design error in
the same notebook, and it means the only clean comparison here is *between losses within
phase 0b*, which share both settings.

Within that clean comparison the ranking is unambiguous: **`masked` > `pu` >> `naive`**.

## 3. ⭐ `naive` degrades with training, exactly as the theory requires

`naive`'s two directions were +0.1228 / −0.0856 in phase 0 (gap 0.2084) and are
−0.0806 / −0.0840 here (gap **0.0034**). It converged to uniformly bad.

Its loss kept falling throughout (0.0364 → 0.0182), and that is the point: **its objective
is to suppress unannotated cells, so optimising it harder makes the detector worse.**
`masked` improved over the same extra epochs (+0.0400 → +0.0919) and `pu` improved
(+0.0447 → +0.0579). Same data, same schedule, opposite direction of travel.

This is the strongest evidence yet for `notes/16` §4, and it is a dose-response in
*training time* to complement `notes/17` §2's dose-response in *annotation density*.

## 4. The binding constraint is data volume, and it is severe

Counting the actual supervision in each training set:

| embryo | positive voxels/frame | ≈ cells/frame | **annotated cells in the whole training set** |
|---|---|---|---|
| `44b6` | 17.3 | ~2.5 | **~590** |
| `6bba` | 54.3 | ~7.8 | **~1,860** |

That is the entire supervised signal — a few hundred to under two thousand cells. And the
weak direction is exactly the small one: models trained on `44b6` (~590 cells) lose to DoG
in both losses, while models trained on `6bba` (~1,860 cells, 3×) win by up to +0.0919.

We used **24 of 199 datasets and 10 of 100 frames** — 1.2 % of the available frames. The
corpus holds 133,318 annotated nodes. There is roughly **80× more supervision available**,
and every signal here says the model is data-starved rather than capacity-starved.

## 5. The operating point: 1.0×, not 0.5×

Recall per detection is monotone decreasing in the cap (prediction 3), so 0.5× always
looks best on that ratio. It is the wrong objective. Taking `masked` trained on `6bba`:

| cap | recall | dets | node ratio | budget mult | ≈ score driver |
|---|---|---|---|---|---|
| 0.5× | 0.6306 | 121 | −0.50 | ~1.05 | 0.63 × 1.05 = 0.66 |
| **1.0×** | **0.8615** | 243 | ~0.00 | ~1.00 | **0.86** |
| 1.5× | 0.8671 | 361 | +0.50 | ~0.95 | 0.82 |

The under-budget bonus tops out at ×1.1 and cannot repay a 0.23 loss of recall. **1.0× is
the operating point**, and 1.5× buys +0.0056 of recall for a 5 % multiplier penalty.

---

## What to do next — phase 1

1. **Train on everything.** All 199 datasets, ~30 frames each, still leave-one-embryo-out
   so each CV fold is scored by a model that never saw it. That is ~9× more supervision
   for the weak `44b6` direction and ~25× for `6bba`.
2. **`masked` is the loss.** Best arm, improves with training, and unlike `pu` it needs no
   per-dataset prior — which matters because that prior varies 20× between embryos.
3. **Operate at 1.0× of predicted budget**, reusing the budget regression from `notes/14`
   (10.7 % median error, and the node-ratio arithmetic above is insensitive to that).
4. **Then score end-to-end**, gated against the standing champion 0.7070. Recall is a
   proxy; the metric is the metric, and nothing here has been through it yet.
5. Measure UNet inference time on CPU during that run — the submission has no internet and
   the image's torch cannot use a P100, so CPU inference may be the only offline path.
