# Spotiflow closes, and prediction 4 was a badly designed test

The threshold sweep answers the question one point could not, and it reverses the reading.

```
prob_thresh      nodes    recall    rec/1k     ratio
0.02            19,539    0.5473    0.0280    -0.097
0.05            15,561    0.5361    0.0345    -0.277
0.1             11,672    0.5164    0.0442    -0.448
0.2              6,917    0.4784    0.0692    -0.655
0.3              3,299    0.3850    0.1167    -0.817
pack            24,605    0.9963    0.0709    +0.150
```

---

## 1. Recall saturates at 0.547 — it is not a threshold problem

Dropping `prob_thresh` from 0.3 to 0.02 raises node count six-fold, from 3,299 to 19,539,
and recall only goes **0.385 → 0.547**. The last three steps buy almost nothing: 0.516 →
0.536 → 0.547 while node count nearly doubles.

So Spotiflow does not miss the annotated cells because it is being thresholded too hard.
**It cannot find 45% of them at any threshold.** The per-dataset rows say the same thing
more bluntly — `44b6_0b24845f` goes to 39,290 detections and still recovers **5.9%** of its
51 ground-truth nodes; `6bba_07e24132` reaches 31,318 detections for **25.1%**.

At the pack's own node count the comparison is not close: interpolated to 24,605 nodes
Spotiflow gives **0.547 against 0.996**, and that number is an *extrapolation past the end
of the sweep*, so it is an optimistic upper bound rather than a measurement.

## 2. Prediction 4 passed for two bad reasons, and the curve caught both

`notes/46` set the crux as "recall per 1,000 nodes at least 3× ours", and v5 reported
**3.4× — PASS**. Both halves of that were wrong.

**Wrong aggregate.** The summary averaged per-dataset ratios; the honest figure is the
ratio of the means.

```
ratio of means   0.1166      <- honest
mean of ratios   0.2394      <- what was reported
dominated by a single dataset at n=512, r=0.657  ->  1.2832
```

One small-`n` dataset carried the mean. Averaging ratios across datasets whose denominators
span 149 to 10,174 is not an aggregate of anything.

**Wrong comparison point.** `recall / nodes` mechanically rises as nodes fall — a detector
emitting ten nodes with one hit scores better on it than one emitting ten thousand with
nine thousand hits. Comparing Spotiflow at 3,299 nodes against the pack at 24,605 measured
the difference in node count, not in selectivity. At **matched** node count:

```
spotiflow @ 19,539 nodes   0.0280 rec/1k
pack      @ 24,605 nodes   0.0709 rec/1k     the pack is 2.5x MORE selective
```

The direction reverses completely. This is the same error shape as `notes/34`, `notes/35`
and `notes/38` — *a control compared against something it is not comparable to* — and it is
the fourth time. The previous three were caught by a follow-up run; this one was too, which
is the only good news in it. A ratio metric needs its denominator held fixed, or it is not
a comparison.

## 3. What closes and what survives

**Closed: Spotiflow as a detector for this pipeline.** It is 17× the parameters
(35,489,892 measured at load) and worse than what we have at every matched node count. And
`r35` is a **run index**, not rank 35 — the repo numbers experiments `R1, R3, R6, R7,
R10..R15, R35` beside `Pivot H/I/R` and `Phase-C`. I read a filename as a leaderboard
position twice before checking, which dressed a stranger's experiment artefact as a proven
solution. Their standing is unknown; this may be a run that failed for them too.

**Survives, unchanged:** `notes/45`'s mechanism. `J_adj = J·(1 − 0.1·ratio)` with a ceiling
of 1.1, and predicted structure matching no ground-truth node is excluded from the edge
term rather than penalised. Over-prediction still buys nothing and still costs budget.

**Survives, and is now better supported:** the pack detector is the most selective thing we
have access to — 0.996 recall at 24,605 nodes, beating a 35M-parameter domain fine-tune at
every matched budget.

## 4. The one variant left, and why it is not the same as what already failed

`claude_budget` cut nodes with `pool_kernel_um`, the **non-maximum-suppression radius**.
That suppresses by spatial neighbourhood, and it deleted ground-truth cells as fast as
anything else (`node_recall` 0.983 → 0.537).

**A top-K cut by the detector's own probability is a different selection rule and has never
been run.** `det_threshold` is the nearest thing tried, and `notes/44` showed it moves node
count only 5.6% across 0.965–0.99 — the distribution is too peaked for a threshold to reach
the budget regime, but a rank cut reaches any K by construction.

The evidence is genuinely mixed on whether it would work. In its favour: the pack finds
essentially every annotated cell (0.996) among 24,605 detections, so if its confidence
ranks them highly a top-2,000 cut keeps most of them and lands at ratio ≈ −0.9. Against it:
nothing yet shows that its confidence correlates with *being annotated* rather than with
*being a cell*, and those are different properties — the annotation is a sparse subset of
real cells, not the bright ones.

That is one cheap run, and it is the last thing standing in this direction.

```
0.901 submitted    0.926 bronze    0.944 gold
measurable > 0.0015     worth a slot > 0.01
```
