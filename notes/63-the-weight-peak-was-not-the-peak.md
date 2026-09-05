# 0.80 was not the peak — and my smoothing find does not transfer

Four GPU arms, all screened on the fork's own `PROXY_SCORE` (10 held-out datasets), no
submission slots spent.

```
run                  PROXY    vs fork   adj_edge
fork (LB 0.937)     0.9266    +0.0000     0.9203
w = 0.85            0.9283    +0.0017     0.9220    <- best
w = 0.90            0.9270    +0.0004     0.9204
w = 0.95            0.9241    -0.0025     0.9179
polyfit degree 0    0.9261    -0.0005     0.9199    <- notes/62's find, LOSES here
```

---

## 1. `SECONDARY_DETECTION_WEIGHT` 0.80 is not the maximum

`nusrati/0-936`'s header calls 0.80 "our independently-swept dual-seed detection-fusion
peak". Swept past it, the shape is a clean interior maximum at **0.85**:

```
0.80   (their value, PROXY 0.9266)
0.85   +0.0017     <- peak
0.90   +0.0004
0.95   -0.0025
```

Unimodal and monotone on both sides, which is worth more than a single lucky point — the
failure mode `notes/42` and `notes/44` recorded three times was an "optimum" that was one
noisy sample. Here the two neighbours fall away in order.

This is `notes/48`'s boundary-value lesson working as intended: a sweep that stops at a
value reports that value as the peak, and the region past it stays unexamined. Their grid
ended at 0.80; the code allows `[0, 1)`.

## 2. `notes/62`'s degree-0 smoothing does NOT transfer to the fork

I reported +0.0030 for replacing the line fit with a window mean, measured on our cached
graphs, and said it "transfers to the fork as one character". **On the fork it scores
−0.0005.** The character transferred; the gain did not.

The measurement on our chain stands — 4/4 predictions, monotone across five thresholds,
both embryos agreeing. What was wrong was my inference that it would carry.

Two differences that plausibly explain it, neither tested:

* **Ours clamps, theirs does not.** `pipeline/repair.linefit_smooth` bounds the move with
  `max_shift_um=3.2`; `linefit_smooth_output_graph` has no such bound, it just blends
  `(1-w)*orig + w*fitted`. A clamped line fit and an unclamped one are not the same
  estimator, and the mean may be recovering damage that their clamp-free line does not do.
* **Different upstream.** Their nodes come from three models and a different ILP. The
  positions being smoothed are not the positions we measured on.

And the standing one: our +0.0030 is 24 cached datasets at `det 0.985`; their PROXY is 10
datasets on their own held-out split. **Different populations, again** — the error shape
`notes/47`, `notes/50` and `notes/57` all recorded. I should have said "worth testing on
their chain" rather than "transfers".

## 3. What to submit

**`w = 0.85`.** It is the only arm that improves the screen, and it improves it with a
coherent peak rather than a single point.

Sizing it honestly: we have one PROXY→LB calibration point, `0.9266 → 0.937`, an offset of
+0.0104. If that offset held, PROXY 0.9283 would land near **0.939** — short of the 0.940
that rank 100 needs on the live board. `notes/40` measured transfer ratios of 0.28 to 1.22
across four attempts, so a single offset is not a prediction. It is the best available
candidate, not a guaranteed one.

`notes/49` applies and is the reason this is a recommendation rather than a certainty: PROXY
runs on **training** embryos, `notes/24` records that the pack's `split_0` membership is
unknowable so they may be contaminated, and the test set is a third pair. The 0.901 → 0.863
regression came from exactly this kind of train-side confidence.

Keep **0.937 selected** until `w=0.85` scores.

```
0.752 floor    0.901 our chain    0.937 fork (rank ~320)    0.940 = rank 100    0.965 top
```
