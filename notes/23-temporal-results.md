# Step 1 — temporal input does not work, and the coherence deficit may not be real

`claude_temporal_train`, Kaggle 2026-08-22, P100, ~1.6 h. `temporal_radius` ∈ {0, 1} ×
both leave-one-embryo-out folds, loss `pu`, checkpoints selected on **paired recall**.

| fold | paired Δ | position Δ | node Δ |
|---|---|---|---|
| train `44b6` → eval `6bba` | **+0.0063** | **+6.0 %** | −0.0029 |
| train `6bba` → eval `44b6` | **−0.0152** | **−6.7 %** | −0.0194 |

**Prediction 1 FALSIFIED. Prediction 2 N/A. Prediction 3 CONFIRMED.**

One fold up, one fold down, near-symmetric. That is noise around zero, not the 58.7-point
coherence recovery `notes/22` sized. **Feeding the network `(t−1, t, t+1)` bought nothing.**

---

## 1. The control did its job, so the negative result is trustworthy

The frame sampling changed from random frames to consecutive runs — a second change landed
alongside the temporal input, and `notes/18` §1 is the standing lesson about that. The
`r=0` control trained on the same tensors:

| | phase 1b node recall | here | Δ |
|---|---|---|---|
| eval `6bba` | 0.9148 | 0.9049 | −0.0099 |
| eval `44b6` | 0.8826 | 0.9197 | +0.0371 |

Both inside the ±0.05 registered in advance. **Prediction 3 confirmed**, so the sampling
change is neutral and the r1-vs-r0 difference really is about temporal input.

Prediction 2 graded **N/A** rather than CONFIRMED. That gate was added after the synthetic
verification run printed "CONFIRMED: the gain arrives through coherence" on a run where
*both* deltas were negative — `mean(d_paired) > mean(d_node)` holds happily when both are
below zero, and node recall clears a `+0.02` ceiling by falling. Without the gate this run
would have reported a coherence-shaped gain that does not exist.

## 2. ⭐ The bigger finding: the deficit this run was built to close may not be there

| | `notes/21` (full corpus, via `predict_dataset`) | here (eval slice, direct) |
|---|---|---|
| DoG temporal position | **+25.2 %** | **+54.0 % / +64.1 %** |
| learned detector | **−41.3 %** | **+32 % … +73 %** |

On this eval the learned detector is **already coherent** — above the independence bound,
comparable to DoG. There was no 58.7-point deficit to close, which is the simplest
explanation for why temporal context changed nothing.

Something is different between the two measurements, and it is worth naming rather than
guessing at:

1. **Distribution.** This eval calls `predict_volume` directly on the stored training
   volumes — per-frame normalised. `notes/21` went through `predict_dataset`, which served
   the whole-movie normalisation. That is the train/serve skew (§3), and it is the one
   candidate that would move the learned arm without moving DoG.
2. **Detection budget.** DoG runs uncapped here — 232 det/frame on `44b6`, 141 on `6bba`.
   `notes/21` used budget-derived caps. More detections raise recall and move `position`.
3. **Horizon.** 5-frame runs here; full movies there. Cells divide, enter and leave over a
   full movie in ways a 5-frame window never sees.

Candidate 1 alone cannot explain DoG's shift, since DoG normalises internally and is
untouched by `prob_input_norm`. So at least two effects are in play, and **the honest
reading is that `notes/21`'s −41.3 % and this run's +64 % are not measuring the same
thing.** `claude_temporal_score` separates candidate 1 directly.

This matters beyond this run: `notes/21` §2's mechanism — "DoG's determinism is a tracking
asset the learned detector threw away" — was the basis for the whole of step 1. It is now
in question. Not refuted: the scoring-path measurement stands on its own terms. But the
"88 % of a 66.5-point deficit" figure that justified this run assumed those two numbers
were commensurable, and they may not be.

## 3. A real train/serve skew, found while building this

Every training notebook stored `dog_response(load_frame(...))[0]` as its input tensor — a
**per-frame** percentile rescale. `predict_dataset` handed `prob_fn` the raw `load_frame`
output, normalised by **whole-movie** quantiles. Different distributions: 5.7× apart in
range on test volumes.

So every learned arm ever scored through the pipeline — `notes/21`, `notes/22`, and the
submission now on the leaderboard — was served an input distribution it had never trained
on. Those runs are internally consistent with one another because they all had it, so
their relative comparisons stand; what is unknown is how much absolute score it cost.

`Config.prob_input_norm` now defaults to what checkpoints expect, with `"movie"` retained
so the earlier runs stay reproducible. `frame_norm` is its own function that `dog_response`
calls, so the mask and the detector cannot drift apart.

## 4. Two guards fired, one of them on me

**`paired_recall` had an indexing inversion.** `match_nodes` returns one entry *per
prediction* holding the GT index it claimed; I indexed it by GT node. It raised here; with
more predictions than ground truth it would have returned a plausible wrong number. Caught
by the test that asserts the metric separates two arms at identical node recall.

**The sub-DoG guard fired on the control.** `r0/44b6` scored paired 0.8529 against DoG's
0.8550 — a −0.0021 tie — and was not saved. That guard exists to stop *unusable* weights
(`notes/20`: 0.2654 against 0.7696) reaching a scorer and masquerading as a clean
refutation. Applying it to a control is a design error: a control is a measurement, not a
shipment, and refusing it destroys the comparison rather than protecting it. Fixed; and the
scorer was restructured around the three surviving checkpoints rather than re-running 1.5 h
of GPU to manufacture a fourth.

That restructure has a rule worth keeping: `r=0` has an out-of-fold model for only one
embryo, so the temporal comparison runs **only** on datasets where both radii are
out-of-fold. Reusing the `6bba` model on `6bba` data would have kept all 60 datasets and
leaked. A smaller honest number beats a larger contaminated one.

---

## What to do next

**Do not iterate further on temporal input.** It was the single largest lever `notes/22`
identified, it is now measured, and it is worth approximately zero. A second variant
(larger radius, temporal loss term) would be sunk cost against a measurement that already
says the premise is shaky.

The scoring run answers one question — what the normalisation skew was worth — and that is
the last thing in flight. After it:

- if the skew is worth real score, the learned path's earlier numbers were all depressed by
  it and `notes/21`'s conclusions need re-reading against corrected arms;
- if it is worth nothing, then the learned detector genuinely plateaus around 0.65 CV
  against the champion's 0.7070, four runs in, and the honest move is to stop and say so.

Either way the classical champion stands at **CV 0.7070 / LB 0.752**, and nothing measured
since has beaten it.
