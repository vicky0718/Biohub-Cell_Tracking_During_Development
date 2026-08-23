# Step 1 — channel-stacked temporal input does not work, and the deficit was an artifact

> **Correction, 2026-08-23, after reading the public pack's source.** This note originally
> concluded "temporal input does not work". That is too broad and I have narrowed it. What
> was tested here is **frame stacking as input channels**. The 0.915 pipeline does
> temporal modelling a completely different way — `_TemporalAttention`, *per-voxel
> multi-head self-attention across the time axis*, applied at every encoder stage. Channel
> stacking lets the first convolution mix frames once and then the information is gone;
> attention compares frames explicitly at every depth. **Temporal modelling as such is
> untested by us. Only the cheap version of it was tested, and the cheap version is what
> failed.** See §6.

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

## 2. ⭐ The 66.5-point coherence deficit was mostly a measurement artifact

`claude_temporal_score` settled this, and the decisive row is the **champion**:

| | `notes/21` | `claude_temporal_score` |
|---|---|---|
| champion (DoG) temporal position | **+25.2 %** | **+58.4 %** |

Same detector, same 60 datasets, same scoring path, +33 points apart. A fixed band-pass
filter did not change its behaviour between two runs — **the two numbers are different
computations.** `notes/21` derived position from corpus-aggregate edge and node recall;
`paired_recall` computes it per dataset against the real GT edge list, weights by edge
count, and restricts node recall to nodes an evaluable edge actually touches.

Measured consistently, the coherence gap is small:

| arm | SCORE | position |
|---|---|---|
| champion | **0.7070** | **+58.4 %** |
| `r1_movie` | 0.6043 | +49.0 % |
| `r1_perframe` | 0.6030 | +49.3 % |

**~9 points, not 66.5.** The learned detector is somewhat less temporally coherent than
DoG, but it was never sitting below the independence bound in the way `notes/21` reported.

This retires `notes/21` §2's headline — "DoG's determinism is a tracking asset the learned
detector threw away", sized at 66.5 points. The *ordering* survives (DoG is more coherent);
the *magnitude* does not, and the magnitude is what justified step 1. `notes/22`'s
projection table, which extrapolated from that figure to 0.7006 at "DoG parity", is
therefore void — parity was worth ~9 points, not 58.7.

**The lesson is procedural, not scientific.** `notes/21`'s number was computed inline in a
notebook; `paired_recall` is a tested function with a test asserting it separates two arms
at identical node recall. When step 1 was planned, the 66.5-point figure was reused as
though it were an instrument reading. It was an intermediate quantity from one run, never
reproduced, and it drove a GPU run and a pipeline change. **A number that decides work
should be reproduced by something with a test before it is spent against.**

## 2b. The scoring run, in full

`claude_temporal_score`, 2026-08-23. Champion reproduced **exactly** — 0.7070, drift
+0.0000.

| arm | SCORE | edge_J | recall | position | vs champion |
|---|---|---|---|---|---|
| champion | **0.7070** | 0.7128 | 0.866 | +58.4 % | — |
| `r1_movie` | 0.6043 | 0.6111 | 0.873 | +49.0 % | −0.1027 |
| `r1_perframe` | 0.6030 | 0.6098 | 0.873 | +49.3 % | −0.1040 |
| `r0_perframe` (21 ds) | 0.5765 | 0.5808 | 0.820 | +52.3 % | −0.1305 |

| decomposed | value | datasets |
|---|---|---|
| normalisation fix | **−0.0013** | 60 |
| temporal input | **+0.0261** | 21 |

**All three predictions resolved: 1 FALSIFIED, 2 FALSIFIED, 3 CONFIRMED.**

**The normalisation fix is worth nothing** (−0.0013, inside noise). §3's defect is real —
the two distributions genuinely differ by 5.7× in range — but the network is insensitive to
it. A real bug that cost nothing is still worth having fixed, and worth recording as
*measured at zero* rather than quietly assumed to have helped.

**The temporal gain is not a coherence gain.** +0.0261 with position **−2.9 %**, so the
notebook's two-clause gate fired and refused to call it confirmation. Node recall explains
it: 0.820 → 0.873. `r=1` finds more cells; it does not chain them better. That gate was
added after the synthetic run produced exactly this shape, and it earned itself here.

## 2c. The leaderboard result, and the number that decides everything

`claude_submit_unet` scored **0.633**.

| | CV | LB | offset |
|---|---|---|---|
| champion | 0.7070 | **0.752** | **+0.045** |
| learned (`pu` ensemble, cap 1.2×) | 0.6490 | **0.633** | **−0.016** |

It came in **below** the pre-registered `<0.674` band, whose reading was written in advance:
*"the learned path loses more to the hidden set than the classical one does."*

That is now measured, and it is the most decision-relevant number in the project. **The
CV→LB offset is 0.061 worse for the learned detector than for the classical one.** DoG is a
fixed filter and transfers to unseen embryos for free; the UNet was fitted to two specific
embryos and pays for that on two it has never seen — the ensemble did not rescue it.

The practical consequence: a learned arm must now clear roughly **CV 0.813** to be expected
to match 0.752 on the leaderboard. Best measured learned CV is 0.6490.

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

## 6. What the 0.915 pipeline actually is — read from the pack, not from a transcription

`claude_cluster_probe` mounted `pilkwang/biohub-tracking-support-pack-50ep-v1` (public,
**CC0: Public Domain**, 355 MB, v10). It ships not just weights but the **entire training
and inference source**: `repo/src/biohub_tracking/models/temporal_unet.py`,
`simple_node_transformer.py`, `repo/scripts/train_unet_transformer.py`,
`predict_unet_transformer.py`, plus offline wheels.

Their checkpoint's own `model_config`:

```json
{"unet_out_channels": 32, "unet_layers": [32, 64, 128],
 "downsample": [1, 4, 4], "window_size": 2, "pool_kernel_um": 5.0}
```

`epoch = 402`, `best_score = 0.9835`, `test_recall = 0.9765`, 2,077,996 parameters.

Set against what we built:

| | ours | theirs |
|---|---|---|
| parameters | 350,809 | **2,077,996** (6×) |
| epochs | 15 | **402** (27×) |
| temporal mechanism | frames as input channels | **per-voxel self-attention across T** |
| linking | Hungarian, hand-written | **learned** (joint `det_loss` + `edge_loss`) |
| `downsample` | (1, 4, 4) | (1, 4, 4) — *identical* |

The geometry matches exactly, which means our data pipeline was never the problem. What
differed is capacity, training length, the temporal mechanism, and — most of all — that
**they learn the linking and we hand-wrote it**.

**And their density control is a single global constant.** `det_threshold` (a fixed sigmoid
probability, run at 0.985) plus a `pool_kernel_um` max-pool NMS — one number for every
dataset in the corpus. We have a per-dataset budget regression at 10.7 % median error,
reproduced exactly in four separate runs. That asymmetry is the clearest differentiator
available, and it is confirmed from their source rather than assumed.

---

## What to do next

**Stop building our own detector from scratch.** Five runs, and every lever we can reach
alone is measured:

| lever | measured worth |
|---|---|
| loss choice (`pu` over `masked`) | +0.0751 held-out recall — the one that worked |
| best-checkpoint selection | recovered −0.5452 of self-inflicted loss |
| dropping `refine_centroids` | +0.0038 |
| normalisation train/serve fix | **−0.0013** |
| temporal input | **+0.0261, and not through coherence** |
| **best learned arm, ever** | **CV 0.6490 / LB 0.633** |
| **champion** | **CV 0.7070 / LB 0.752** |

The stopping rule from `notes/16` was written for exactly this: *"if the score run fails
the gate, stop iterating on the detector and report."* It failed in `notes/21`, and
`notes/22` argued the failure was informative enough to continue — which was right, because
continuing produced §2's correction. It has now failed three more times, and the reasons
have run out.

Two findings make continuing worse than a coin flip rather than merely unpromising:

1. **§2c: the learned path's CV→LB offset is negative.** A learned arm needs ~CV 0.813 to
   project to 0.752. Nothing measured is within 0.16 of that.
2. **§2: the deficit that motivated the last two runs was a measurement artifact.** The gap
   to close is ~9 points of coherence, not 58.7, and temporal input did not close even
   that.

**What is banked:** the classical champion at **CV 0.7070 / LB 0.752**, reproduced exactly
(drift +0.0000) in three separate scoring runs. It is not in doubt.

**On the goal.** Live leaderboard: top **0.962**, gold ~**0.935**, the public cluster
**0.913–0.916** with ~514 teams, us at **0.752**. The gap from us to the cluster is +0.163;
the gap from the cluster to gold is **+0.02**. Those are very different problems, and the
project has been framing the wrong one.

Rules §6.b permits external models ("acceptable unless specifically prohibited by the
Host"); §6.a's "publicly available and equally accessible ... at no cost" is satisfied by a
CC0 public dataset. So the route is:

1. **Reach the cluster** using the published pack — a floor jump of +0.163 that costs days,
   not weeks, and does not depend on out-training 514 teams.
2. **Beat it by +0.02** with the asymmetry §6 identifies: their global `det_threshold`
   against our per-dataset budget regression, aimed at the node-budget multiplier, which
   `notes/16` §2.3 already argued is "the only way the arithmetic reaches 0.935".

The open item is the **Competition-Specific Rules** section, which §7.a references and
which the probe could not read. If it restricts external models, everything above is void
and the honest position reverts to banking 0.752.
