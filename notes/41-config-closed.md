# All three config axes close on the cell we already had, and the last lever goes to the GPU

`claude_config_sweep2` refined every axis `notes/40` left on a boundary — 44 cells, 12
datasets, one run — and **found nothing better than `d0.975_m6_g2`**. Same cell, same
0.9535, to four decimals. The configuration direction is finished.

```
det        m0g2     m6g2     m8g2    m10g2     m0g3     m6g3     m8g3    m10g3    m12g3     m6g1     m0g1
0.98     0.9508   0.9521   0.9509   0.9477   0.9495   0.9504   0.9504   0.9479   0.9430   0.9517   0.9517
0.975    0.9515   0.9535   0.9515   0.9493   0.9499   0.9515   0.9515   0.9498   0.9463   0.9519   0.9514
0.97     0.9505   0.9525   0.9494   0.9482   0.9488   0.9503   0.9492   0.9487   0.9452   0.9506   0.9501
0.965    0.9507   0.9528   0.9501   0.9489   0.9489   0.9505   0.9497   0.9493   0.9457   0.9504   0.9504
```

Four of five predictions passed. The fifth is graded below, because **its failure text is
wrong about what it measured** and a future read would take the wrong lesson from it.

---

## 1. Three interior optima, on grids that do not share an edge

Every axis `notes/40` flagged is now bracketed on both sides:

| axis | tested | best | neighbours |
|---|---|---|---|
| `DET_THRESHOLD` | 0.98, 0.975, 0.97, 0.965 | **0.975** | 0.9521 / 0.9525 |
| `GAP_CLOSE_MAX_GAP` | 1, 2, 3 | **2** | 0.9519 (g1) / 0.9515 (g3) |
| `OUTPUT_MIN_TRACK_LEN` | 0, 6, 8, 10, 12 | **6** | 0.9515 (m0, m8) / 0.9493 (m10) |

The threshold result is the one worth trusting: 0.975 won `notes/40`'s coarse grid
(0.99 / 0.985 / 0.975 / 0.96875) and won again on a **finer grid that shares no other
point with it**. Two independent bracketings agreeing on the same interior value is a
different kind of evidence from one grid's argmax, which is what half this project's
"optima" have actually been.

`max_gap=3` is worse than 2 everywhere — the `m6` row drops 0.9535 → 0.9515. So the
boundary `notes/40` complained about was a boundary in the sampling, not in the response:
bridging holes helps to two frames and hurts past that. Same for `min_len`, where 8 loses
0.0020 and 12 loses 0.0072. Both were the cheap tests, both came back negative, and that
is the point of running them in the same batch as everything else rather than one per
launch.

## 2. Prediction 4 failed against the wrong baseline

The notebook printed:

```
4. the best cell beats the current submission (0.9499) by more than 0.001
   d0.975_m6_g2   0.9535  (+0.0000)   -> FAIL
   "The audited settings do not pay on our pipeline."
```

The deltas it printed are against **`notes/40`'s best cell (0.9535)**, not against the
0.9499 named in the prediction's own text. So the number is right and the conclusion
attached to it is not. What was actually measured: *sweep2's refinement found nothing
beyond sweep1's best.* The audited settings do pay — **+0.0036 over 0.9499**, measured in
`notes/40`, reproduced here to four decimals by prediction 1, and already sitting in the
finished `claude_submit_config`.

Writing this down because the wrong version is the more quotable one, and a future read of
that log without the surrounding context would retract a real gain on the strength of a
mis-worded f-string. The pre-registration discipline works; the baseline inside the
prediction has to be the same object the code subtracts.

## 3. `claude_secondary` — the last identified lever, now running

`notes/33` §1 put ~0.04 in the two model datasets the 0.927 notebooks attach and we do
not. One of them, `biohub-deepcenter-unet3d-center-prior-v1`, measured ~0.002
(`notes/34`). This is the other: `biohub-temporal-unet3d-seed314159-v1`, a second edge
predictor of the same architecture from a different seed. It is **the only remaining
untested thing that adds model capacity** rather than re-reading the model we have.

`pipeline/secondary.py` blends its logits into the primary's inside the pack's own loop,
at two source anchors. Two mixing modes, because the second needs the first as its control:

* **`fixed`** — a constant weight everywhere.
* **`low_margin`** — the weight scales with the primary's own top-2 uncertainty, and is
  **zeroed where the two models disagree** about the best parent. Where the primary is
  confident there is nothing to gain; where the two contradict each other, averaging
  produces a blur worse than either.

The secondary is calibrated onto the primary's per-target mean and std first
(`bidirectional.calibrate`), because the downstream candidate threshold and the ILP were
both tuned against the primary's scale and mixing raw logits would let the secondary's
arbitrary temperature move them. `weight=0` returns the primary **by identity**, so the
control arm is bit-identical rather than approximately equal.

### The build cost one launch instead of three

`claude_bidirectional` burned three GPU launches on anchor mismatches and a missing import,
each one an 8-minute round trip to learn a fact that was written down in a file we could
have read. This time the pack's actual `predict_unet_transformer.py` was downloaded
directly — the Kaggle datasets download endpoint takes `?file_name=repo/scripts/...` — so
`probes/exec_secondary.py` applies the patch to the **real 8,841-character `predict_video`
text** and compiles it here:

```
PASS  both anchors appear exactly once in the real source  — encode 1x, blend 1x
PASS  the patched function still compiles  — 8,841 -> 10,014 chars
PASS  it composes with the bidirectional patch
PASS  and both mechanisms are present after composing
```

Patching against the real source instead of a guess is the cheapest thing in this project
that has ever prevented a failed launch, and it was available the whole time.

### v1 died on two things, and the second one was the dangerous kind

```
official scorer clone rc=128
  pack   /kaggle/input/datasets/pilkwang/biohub-temporal-unet3d-seed314159-v1
```

The first is mundane: the push copied `claude-submit-ratio`'s configuration, which has
**internet off**. That kernel writes a submission file and never scores; this one scores
against ground truth, and the ILP at 0.4/2.0 forks by design so `purescore` is not exact
and the official scorer has to be cloned. `kernel_push_like` was built to stop exactly this
class of mistake and it worked as designed — it copied a *consistent* configuration, just
of the wrong kernel. `claude-config-sweep2` is the right reference: internet on, and it
carries the candidate cache too.

The second did not crash, and would not have. **The secondary model resolved as the pack.**
Both datasets ship the same `repo/ + weights/` layout, the lifted predicate matches both,
and `find_dir` walks `/kaggle/input` alphabetically — `biohub-temporal-…` sorts before
`biohub-tracking-…`. Had the scorer clone succeeded, the primary would have loaded the
secondary's weights, `SEC_PATH` would have found the same file, and every arm would have
blended a model **with itself**. Every weight returns the control. Prediction 4 reports no
gain. The notebook prints *"this closes the last identified lever"* and it would be wrong,
about a mechanism it never once executed.

This is the third time this project has been bitten by **alphabetical order deciding
something that was never meant to be alphabetical** — `notes/34`'s `names[:12]` inverting
the embryo split, `claude_bidirectional` v2 repeating it, and now this. The fix is not a
better predicate; it is refusing to infer identity from a search at all where a cheap
direct check exists. The pack is now identified by something only the pack has, and the
worker then hashes both checkpoints and exits loudly if the bytes match:

```
if _dp == _ds_:
    raise SystemExit("primary and secondary are the SAME WEIGHTS. An ensemble of a model
                      with itself returns the control at every weight and would be
                      reported as 'the mechanism does not pay'.")
```

Comparing the two *paths* would not have been enough — two paths can name identical bytes.
The invariant that matters is that the ensemble contains two different models, so that is
what gets asserted.

The rebuild then failed the dry run's static f-string check: the worker is rendered with
`.format()`, so the braces in the new diagnostic prints were being eaten as template
fields — `unfilled '_dp'`. That check exists because of `NameError: N_DATASETS`, and it
has now caught the same class of bug twice without a GPU minute spent.

### What it varies, and what it holds fixed

Everything located so far is frozen — det 0.975, gap 2, min-track 6, ILP 0.4/2.0,
bidirectional w=0.15 — and only the secondary's weight and mode move:

```
(0.00, fixed)        control, bit-identical to d0.975_m6_g2
(0.15, low_margin)   (0.30, low_margin)
(0.15, fixed)        (0.30, fixed)        (0.50, fixed)
```

Prediction 4 of the five is the sharp one: if no arm clears the control by more than
0.001, this closes **the last identified lever**, and what is left is not a lever we have
named.

## 4. Where that leaves the arithmetic

```
                 train      LB delta          implied ratio
repair chain    +0.0115   (0.0120, 0.0140)   (1.04, 1.22)
ILP asymmetry   +0.0037   (0.0020, 0.0040)   (0.54, 1.08)
ratio0.4_2.0    +0.0221   (0.0130, 0.0150)   (0.59, 0.68)
bidirectional   +0.0036   (0.0010, 0.0030)   (0.28, 0.83)
```

Direction has transferred **5/5**. Magnitude converts by no fixed factor — the first and
third measurements do not even overlap — so the only honest forecast for a train gain is
*"probably the same sign, roughly a third to roughly all of it."*

Every mechanism this session has returned +0.002 to +0.004 on the leaderboard. None
larger. `claude_submit_config` is built and awaiting a click at an expected 0.901–0.903.
Bronze at 0.926 is +0.027 from there, which is another dozen findings of the size we have
been producing, and the pool of identified candidates is down to one — the run that just
started.

Banked floor **0.752**. Best scored **0.899** (rank ~1297/2792).
Bronze **0.926**, gold **0.944**.
