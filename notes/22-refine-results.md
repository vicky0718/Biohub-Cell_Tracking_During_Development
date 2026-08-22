# Phase 1c — the mechanism is confirmed, and it accounts for 12 % of the gap

`claude_detector_refine`, Kaggle 2026-08-22, 4,834 s. Champion reproduced exactly
(0.7070, drift +0.0000); budget regression reproduced exactly (10.7 %).

**All three pre-registered predictions CONFIRMED.** That has not happened before in this
project, and the one that matters is prediction 2.

| arm | SCORE | edge_J | node recall | **temporal position** | gate |
|---|---|---|---|---|---|
| champion | **0.7070** | 0.7128 | 0.866 | **+25.2 %** | — |
| unet_cap1.2_norefine | 0.6490 | 0.6556 | 0.885 | −33.5 % | reject |
| unet_cap1.0_norefine | 0.6476 | 0.6432 | 0.863 | −11.2 % | reject |
| unet_cap1.2 | 0.6451 | 0.6518 | 0.889 | −41.3 % | reject |

`refine ON → OFF at cap 1.2`: **score +0.0038, temporal position +7.9 %, node recall
−0.0033.**

---

## 1. ⭐ The prediction that matters came true in the exact predicted shape

Prediction 2 said the gain would arrive through **temporal coherence** and not through node
recall. It did, and the numbers separate cleanly:

| | change |
|---|---|
| temporal position | **+7.9 points** |
| node recall | −0.0033 |
| near-miss share | 86.1 % → 85.9 % |
| matched GT nodes | 33,940 → 33,863 |

**Node-level localisation did not improve. Coherence did.** Slightly *fewer* GT nodes were
matched, and the near/far split barely moved — yet the detections chained into edges
better.

That is the signature of **variance, not bias**. The intensity refinement was not moving
peaks systematically off-centre; it was moving them by a *different amount each frame*,
because the intensity noise it keys on differs frame to frame. A per-frame position wobble
is invisible to any single-frame measurement and lethal to edges, which need the same cell
in the same place twice.

`tests/test_detector.py` showed the shift is 1.166 µm on an oracle map. This shows what
that 1.166 µm was actually costing: it was jitter.

## 2. And it is 12 % of the problem

| | |
|---|---|
| temporal position gap, UNet → DoG | −41.3 % → +25.2 % = **66.5 points** |
| closed by removing refinement | **7.8 points (12 %)** |
| **remaining** | **58.7 points (88 %)** |

So the suspect was real and is now eliminated — and it was a small share of the deficit.
The other 88 % is the thing refinement was never going to fix: **the model sees one frame
at a time.** Nothing in its input or its loss asks it to be consistent with the frame
before or after.

That is precisely what the remaining lever addresses, and this run has now sized it.

## 3. What closing the rest is worth, from this baseline

Projecting from `unet_cap1.2_norefine` (0.6556 edge Jaccard, node recall 0.885):

| coherence reached | precision unchanged | + DoG's edge precision |
|---|---|---|
| DoG parity (+25.2 %) | 0.7006 | **0.7377** |
| perfect (100 %) | 0.7567 | **0.8002** |

Champion is 0.7128. So **temporal coherence alone, at DoG parity, still lands just short**
(0.7006). It clears the champion only if edge precision moves too — currently 0.8388
against the champion's 0.8925.

Both should move together if the cause is shared: stable detections produce both more
matched pairs *and* fewer confident mislinks. The projection is not a promise, but it does
say a temporal model has to do more than reach parity on coherence to win.

## 4. Prediction 3 was worth writing down

"It is still not enough to pass the gate" — CONFIRMED, deficit 0.0619 closed by 0.0038,
6 %. It was recorded so that a pass would draw scrutiny rather than celebration. It also
kept the result honest: a +0.0038 gain is real, mechanistically explained, and **nowhere
near** sufficient, and all three of those things are true at once.

## 5. Operational: the run that died first

v1 of this notebook exited at 302 s on its own guard — `Config` had no `refine` field. The
zip contained it and Kaggle was on version 17, so the upload was fine; the fault was
`dataset_status` reporting "ready" for the *already published* version, so my wait loop
returned instantly and the kernel attached the previous one.

`dataset_new_version` now blocks until `currentVersionNumber` actually increments
(measured: 17 → 18 in 11 s, where the old check claimed 0 s). And the guard converted a
silent wrong answer — `refine=False` reading as a no-op, the hypothesis dismissed on
evidence that never tested it — into a loud five-minute failure.

---

## What to do next

**Temporal input**, which is now sized rather than guessed: 88 % of a 66.5-point coherence
deficit, worth roughly +0.045 to +0.10 of edge Jaccard depending on whether precision
follows.

Feed `t−1, t, t+1` as input channels so the network can see persistence. Train from the
same data and the same `pu` loss, with best-checkpoint selection and `refine=False` — this
run establishes `unet_cap1.2_norefine` at **0.6490** as the baseline to beat.
