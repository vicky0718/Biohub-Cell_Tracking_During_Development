# Phase 1b-score — the detector finds more cells and tracks them worse

`claude_detector_score`, Kaggle 2026-08-22, 5,201 s. `pu` selected automatically (only loss
covering both embryos), each fold scored by the model that never saw it. Champion
reproduced exactly: **0.7070, drift +0.0000**. Budget regression reproduced exactly:
10.7 % median error.

| arm | SCORE | edge_J | mult | node recall | nodes | gate |
|---|---|---|---|---|---|---|
| **champion** | **0.7070** | 0.7128 | 0.9918 | 0.866 | 1,205,332 | — |
| unet_cap1.2 | 0.6451 | 0.6518 | 0.9897 | **0.889** | 1,410,923 | reject |
| unet_cap1.0 | 0.6434 | 0.6391 | 1.0066 | 0.866 | 1,203,147 | reject |
| unet_cap0.8 | 0.6011 | 0.5863 | 1.0252 | 0.792 | 970,888 | reject |

**All three predictions FALSIFIED. No arm passes the gate. The champion stands at 0.7070.**

And yet this is the most informative run in the project, because of one row.

---

## 1. ⭐⭐ At identical node count and identical node recall, the UNet's edges are 0.074 worse

| | `unet_cap1.0` | `champion` |
|---|---|---|
| predicted nodes | 1,203,147 | 1,205,332 |
| node recall | **0.866** | **0.866** |
| edge Jaccard | **0.6391** | **0.7128** |

Same number of detections. Same fraction of ground-truth cells found. **−0.0737 of edge
Jaccard.** It is not density and it is not recall — *the detections themselves link worse.*

`unet_cap1.2` makes it starker: it finds **more** GT cells than the champion (0.889 vs
0.866) and still scores 0.0619 lower.

## 2. The mechanism: temporal incoherence, and it is quantitative

An edge is a true positive only if **both** endpoints match GT. So edge recall is bounded:

- if per-frame detections were **temporally independent**, P(both matched) → `r²`
- if the **same cells** were found every frame, P(both matched) → `r`

Placing each arm between those bounds:

| arm | node recall `r` | edge recall | `r²` (independent) | `r` (perfect) | **position** |
|---|---|---|---|---|---|
| champion | 0.866 | 0.7798 | 0.7500 | 0.8660 | **+26 %** |
| unet_cap0.8 | 0.792 | 0.6593 | 0.6273 | 0.7920 | +19 % |
| unet_cap1.0 | 0.866 | 0.7307 | 0.7500 | 0.8660 | **−17 %** |
| unet_cap1.2 | 0.889 | 0.7490 | 0.7903 | 0.8890 | **−42 %** |

**DoG sits above the independence bound; the UNet sits below it.**

Above means DoG's detections are *correlated* across frames — find a cell at `t` and you
very likely find it at `t+1`, because a deterministic band-pass filter responds the same
way to the same structure. That correlation is worth +26 % of the available range and
nobody designed it; it is a free consequence of using a fixed filter.

Below the independence bound means something worse than flicker: **the linker is actively
mispairing.** The edge confusion matrix agrees — at matched node count the UNet produces
5,215 false-positive edges against the champion's 3,418, a 53 % increase, while also
producing more false negatives.

> **DoG's determinism is a tracking asset the learned detector threw away.** A per-frame
> UNet is free to jitter in position and membership between frames, and every such jitter
> costs an edge at both ends.

This is why the public 0.915 pipeline is a **Temporal**UNet3D and not a UNet3D. That
detail read as an architecture footnote in `notes/15`; it is the whole point.

## 3. A concrete secondary suspect, one line long

`predict_dataset` applies `refine_centroids(vol, c, voxel_um)` unconditionally, where `vol`
is the normalised **intensity** volume. For DoG that is coherent — the coordinates are
intensity-blob maxima being refined against intensity. For the UNet the coordinates come
from `peaks_from_prob`, which already returns a **sub-voxel centroid of the probability
plateau**, and they are then shifted again by an intensity-weighted centre of mass in a
2.5 µm window.

That second shift is uncalibrated for a probability peak, and — critically — it varies
frame to frame with intensity noise. It is a plausible *source* of the jitter in §2, and it
applies only to the learned path.

It is not the whole story: the near/far miss split shows the UNet's node localisation is if
anything *better* than DoG's (86.1 % near share vs 89.8 %, on more matched nodes). But it
costs one line to test and it only ever hurts the arm that is failing.

## 4. Prediction 2 also fell, and the operating point moved

`notes/18` §5 argued 1.0× was optimal because the under-budget bonus cannot repay lost
recall. The ordering came out **1.2× > 1.0× > 0.8×**: the extra recall at 1.2× (0.889)
outweighs its multiplier penalty (0.9897). That analysis was done on recall proxies; on the
real metric the optimum is at or above 1.2× and has not been bracketed from above.

## 5. What the per-dataset spread says

| arm | min | p25 | median | p75 | max |
|---|---|---|---|---|---|
| champion | 0.487 | 0.787 | 0.924 | 0.963 | 1.000 |
| unet_cap1.2 | **0.655** | **0.830** | 0.925 | 0.969 | 1.000 |

The UNet's **worst** datasets are much better than the champion's — min 0.655 vs 0.487,
p25 0.830 vs 0.787 — while the medians match. The learned detector is more *robust* across
crops; it just cannot convert that into edges.

---

## Where this leaves the plan

The pre-registered stopping rule (`notes/16`, and the approved plan) says: if the score run
fails the gate, stop iterating on the detector and report. **It failed the gate, so this is
the report.**

But the rule was written to prevent sunk-cost iteration on an *uninformative* failure, and
this failure is the opposite of uninformative. It says, with a quantitative fit:

- the learned detector **wins the job it was built for** — more GT cells found, more robust
  across datasets, at equal cost
- it loses on a **different axis nobody was measuring**, temporal coherence, which the
  classical detector got for free
- the fix is named, is standard, and is what the 0.915 pipeline does

That is a decision for the owner, not a decision to make by momentum. Options, honestly
costed:

1. **Stop.** 0.752 is banked and reproducible. Four runs into the detector, no gate pass.
2. **One cheap test (~1.5 h):** skip `refine_centroids` on the learned path. If §3 is a real
   contributor the arm moves; if not, it is ruled out for good.
3. **Temporal input (~3 h train + 1.5 h score):** feed `t−1, t, t+1` as channels so the
   network can be consistent across frames. This is the fix §2 actually points at, it is
   the single largest untested lever, and it is what the leading public pipeline does.
