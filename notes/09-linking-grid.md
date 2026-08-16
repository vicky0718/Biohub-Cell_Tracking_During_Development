# Experiment #2 — the linking grid, and why the detector is the problem

`03_linking.ipynb`, run on Kaggle 2026-08-16, 2.2 h on a fixed hash-chosen 60-dataset
subset. Raw output in `linking_results.json`.

**Subset is representative**: incumbent scores 0.5790 on the subset vs 0.5552 on all 199
— a +0.0238 gap, inside the ±0.03 tolerance the notebook set beforehand.

---

## 1. ❌ A flaw in my grid design: two of the three rows were the same experiment

| sep \ radius | 9.0 | 7.0 | 5.0 | 4.0 |
|---|---|---|---|---|
| **6.0** | 0.5790 | **0.5862** | 0.5703 | 0.5413 |
| **4.5** | 0.3290 | 0.3391 | 0.3449 | 0.3416 |
| **3.5** | 0.3290 | 0.3391 | 0.3449 | 0.3416 |

Rows 2 and 3 are identical to four decimals, with identical node counts. Not coincidence —
`_footprint()` rounds the window to an odd number of voxels, and after `downsample=(1,4,4)`
the grid is isotropic at 1.625 µm, so only four separations are reachable at all:

| nominal `min_separation_um` | actual window |
|---|---|
| 1.0 | 1.625 µm |
| **2.5 – 5.0** | **4.875 µm** |
| **5.5 – 8.0** | **8.125 µm** |
| 8.5 – 11 | 11.375 µm |

The sweep asked for {6.0, 4.5, 3.5} and tested **{8.125, 4.875, 4.875}**. Prediction 2
("smaller separation raises recall") is still confirmed, but on **two** points, not three,
and about an hour of the run was a duplicate arm. My error in designing the grid — the
quantisation was visible in `_footprint` the whole time.

**Consequence beyond this run:** `min_separation_um` is not continuously tunable in the
intensity detector. Anything between 2.5 and 5.0 is the same setting. The ball footprint
added for the DoG path does not have this defect (3.5/4.5/6.0 µm → 33/81/203 voxels), which
is one more reason to move there.

## 2. ⭐⭐⭐ Recall is not the objective — and this is the clearest result we have

Tightening the window from 8.125 µm to 4.875 µm:

| | 8.125 µm | 4.875 µm | Δ |
|---|---|---|---|
| predicted nodes | 1,259,152 | 3,123,727 | **×2.48** |
| **node_recall** | 0.895 | **0.976** | **+0.081** |
| edge Jaccard | 0.5861 | 0.4174 | **−0.169** |
| budget multiplier | 0.988 | 0.826 | −0.162 |
| **SCORE** | **0.5790** | 0.3449 | **−0.234** |

Decomposed against the ~40,196 GT nodes in the subset:

- extra detections spent: **1,864,575**
- extra GT nodes found: **~3,264**
- → **571 spurious detections per additional ground-truth node found**

That is the whole story of this pipeline. Our intensity detector *can* reach 97.6 % recall,
but only by spraying peaks, and the cost is catastrophic on both terms of the metric: the
linker drowns in distractors (−0.169 edge Jaccard) and the node budget starts charging
(×0.826, i.e. 2.7× over budget).

**The binding constraint is detection precision, not detection recall.** `notes/05` §3
identified the linking gap and proposed tightening the radius to fix it; this run shows the
radius is a second-order knob and the real lever is not emitting the junk in the first
place.

This is exactly the mechanism `notes/08` §3 measured on synthetic data: at a *matched*
detection budget, DoG recovers 0.67 of dim nuclei where intensity thresholding recovers
0.17. Recall per detection spent is the quantity that matters, and it is the quantity DoG
improves.

## 3. The three pre-registered predictions

1. **"Score improves as the radius falls below 9."** ✅ **CONFIRMED** — 7.0 beats 9.0
   (0.5862 vs 0.5790). But the gain is small, and it **fails the gate**: pooled +0.0073
   with fold 3 at −0.0016. Under the promotion rule this is a REJECT, and given the DoG
   swap is worth an order of magnitude more, it is not worth another run to chase.
   (Note: this ran under the *old* five-way hash folds; the notebook predates the switch to
   `fold_by="embryo"`, so the regressing fold measures within-embryo variance, which
   `notes/07` §3 established is the wrong question anyway.)
2. **"Smaller separation raises recall."** ✅ CONFIRMED, 0.895 → 0.976 — and it is a trap,
   see §2. Recall rose and the score fell by a third.
3. **"The two interact — the best radius is tighter at higher density."** ✅ CONFIRMED:
   best radius is 7.0 at the 8.125 µm window and 5.0 at the 4.875 µm window. Denser
   detections need a tighter linker, precisely as predicted.

Getting all three right and still concluding "do something else" is the useful outcome
here: the predictions were about the linker, and the answer is that the linker is not where
the score is.

## 4. The node budget, and a caveat on `notes/05` §2

On this 60-dataset subset the incumbent runs at ratio **+0.121 — over budget**, not the
−0.111 measured across all 199. The subset is a different mix of budgets, so the sign of
the node-budget term is subset-dependent. The full-199 ablation remains the more reliable
measurement and `budget_fill=None` stays the default, but "we are comfortably under budget"
should be read as a statement about the full training set, not a universal property.

At the 4.875 µm window we are **2.7× over budget** on the subset, which is well into the
region where the multiplier bites.

## 5. Per-dataset variance is as large as the forum said

Scoring the four placeholder test datasets (which `notes/07` §2 established predict
nothing about the leaderboard, but do exercise the path):

| dataset | edge J | node recall |
|---|---|---|
| `44b6_0113de3b` | 0.5156 | 0.962 |
| `44b6_0b24845f` | 0.2264 | **0.373** |
| `6bba_05b6850b` | 0.6514 | 0.920 |
| `6bba_05db0fb1` | 0.5351 | 0.866 |

Node recall from 0.373 to 0.962 across four crops of the same two embryos. The forum's
"±0.14, 18 % coefficient of variation" is if anything an understatement. Any conclusion
drawn from a handful of datasets is noise.

---

## What this changes

1. **Stop tuning the linker.** All three predictions confirmed, total available gain
   +0.0073, and it does not pass the gate. Done with this axis.
2. **`min_separation_um` in the intensity path is a four-valued knob.** Do not sweep it
   continuously; the DoG ball footprint is continuous and should be used instead.
3. **The next experiment is the detector, and the metric to watch is recall per detection**,
   not recall. Concretely: compare arms at matched node counts, because this run shows the
   score difference between two detectors at different densities is dominated by density,
   not by detector quality.
4. **The measured target is 0.682** — what a plain DoG + Hungarian pipeline scored for the
   rule-based author (`notes/08` §2) against our 0.5790 on this subset.
