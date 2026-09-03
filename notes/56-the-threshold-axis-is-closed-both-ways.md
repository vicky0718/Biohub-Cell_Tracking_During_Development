# The threshold axis is closed in both directions, and the misses are capacity

`claude_loosen`, 36 datasets, 5 thresholds × 6 post combos, 11,564 s on a T4. The grid went
**below 0.965 for the first time in this project** — every previous sweep bottomed out at
0.96875.

```
det        score  adj_edge   edge_J  node_rec    ratio    mult      nodes  fn_detect  fn_mislink
0.975     0.9356    0.9264   0.9241    0.9827   -0.025  1.0025    700,216        523         345
0.95      0.9359    0.9259   0.9243    0.9823   -0.017  1.0017    711,997        544         336
0.9       0.9359    0.9254   0.9242    0.9822   -0.013  1.0013    718,166        549         340
0.8       0.9356    0.9249   0.9237    0.9804   -0.013  1.0013    717,741        583         327
0.6       0.9353    0.9238   0.9223    0.9788   -0.016  1.0016    712,634        613         328
```

---

## 1. Correction first: prediction 1's FAIL is my bookkeeping, not a reproduction failure

The grading cell reported the anchor at **0.9356 against a stated 0.9410** and declared
nothing below comparable. **That reference was wrong.** I took 0.9410 from `notes/48`'s
table, which reports *"mean best-cell score"* — the mean over datasets of the **best cell for
each**, an optimistic best-of statistic. The anchor here is a **fixed** cell, `m6_g2`.

Against the right reference, `claude_widecv` in `notes/44`:

```
notes/44   d0.98_m6_g2   0.9356      this run   p0.975_m6_g2   0.9356
notes/44   d0.975_m6_g2  0.9348
```

**The anchor reproduces exactly.** Predictions 2–4 are readable, and I am reading them.

## 2. The sigmoid is saturated below 0.965 too. The axis is closed both ways.

```
node count   700,216 -> 718,166 across det 0.975 -> 0.6      x1.026
```

**2.6%.** `notes/44` measured 5.6% over [0.965, 0.99] and called the axis flat; it is just as
flat, and flatter, going the other way. Dropping the threshold to **0.6** — a value no
sane pipeline would ship — moves node count by less than three percent.

So `notes/52`'s framing was right that we sit under budget and wrong that the threshold
could spend it. It cannot move node count in *either* direction:

```
notes/46     pool_kernel_um (NMS radius)            recall collapses
notes/48/49  det_threshold UP                       0.901 -> 0.863 on the LB
notes/52     track ranking under a per-dataset cap  monotonically worse
notes/56     det_threshold DOWN                     node count moves 2.6%
```

## 3. Loosening makes the undetected endpoints WORSE

The crux, and it fails in the informative direction:

```
det      fn_detect   vs anchor
0.975          523         +0
0.95           544        +21
0.9            549        +26
0.8            583        +60
0.6            613        +90
```

`notes/52` predicted that spending the multiplier would buy back ~29% of the missed
endpoints. It buys back **none** — it loses 90 more. Adding low-confidence candidates does
not recover a missed cell; it gives the ILP worse material and it displaces matches that
were being made. `node_recall` falls too, 0.9827 → 0.9788.

**Those 523 undetected endpoints are detector *capacity* misses, not threshold misses.**
That is exactly `notes/51`'s reading, now tested against the one knob that could have
refuted it. No configuration reaches them.

## 4. The best arm is at the floor and fails the embryo test

```
best  p0.95_m6_g1  0.9371  vs anchor 0.9356   +0.0015    (notes/44's floor is 0.0015)
paired: mean -0.0006  sd 0.0129  t -0.26  not resolved
per-embryo: 44b6 -0.0015   6bba +0.0000   -> NO, pooled t is pseudoreplicated
```

The per-embryo column added to this builder after `notes/49` did its job on its first real
outing: the best-looking arm wins on neither embryo and its pooled mean is *negative*.
Nothing here is submittable.

## 5. What this leaves

Every axis inside our own pipeline is now closed with a measurement rather than by
exhaustion:

```
config      notes/44, 49        divisions   notes/43, 50
node budget notes/46, 48/49, 52, 56 (all four selection rules)
detection ensembling  notes/55  (spotiflow rescues 9 GT nodes of 8,175)
threshold   notes/56  (closed in BOTH directions)
```

The remaining loss is detector capacity, and the only untried lever that addresses it is
`notes/54`'s Track B: **the cluster runs three models and we run one**, integrated at the
stages `notes/55` confirmed load correctly in the fork — dual-seed detection fusion and a
DeepCenter veto on gaps *and* divisions.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
Track A: claude_fork built submission.csv, awaiting submission
Track B: the three-model port is now the only open direction
```
