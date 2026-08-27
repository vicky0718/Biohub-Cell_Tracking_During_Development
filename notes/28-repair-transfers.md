# 0.867 → 0.880. The repair transfers, slightly better than it measured.

`claude_submit_repair` v1, leaderboard **0.880**.

```
0.752  classical champion
0.843  pack, ILP bypassed
0.867  pack, ILP running
0.880  pack + ILP + gap-close(5.75 µm) + linefit-smooth(0.76, win 2)   <- here
0.913–0.916  the cluster, same weights
```

**+0.0130 on the leaderboard against +0.0115 measured on training data.** A transfer ratio
of **1.13×**. The gap to the cluster is now **0.033–0.036**, down from 0.046.

---

## 1. Why this was a controlled measurement and not a hopeful one

The run changed exactly one thing from the 0.867 submission, and the logs prove it rather
than assert it. Per-dataset ILP counts, both runs:

```
24406/24613   20806/22446   6110/6247   64372/65850     <- identical, both runs
nodes   127,790 -> 129,239   (+1,449)
edges   115,694 -> 118,592   (+2,898)
```

The raw prediction is **bit-identical**, and 2,898 = 2 × 1,449 exactly — every node
`close_gaps` inserts brings precisely two edges, which is what it is built to do.
`DET_THRESHOLD` was deliberately held at 0.99 rather than moved to the 0.985 the repair was
tuned at, specifically so the leaderboard delta would be the repair and nothing else.

It also ran on a **T4** this time and reproduced the P100 run's ILP counts exactly, so the
prediction is stable across hardware. That was not guaranteed and is worth knowing before
any future run is compared against an older log.

## 2. The transfer ratio is above 1, and there is a reason to expect that

`notes/24` §2 established that these weights were fitted on some unpublished subset of the
same 199 training datasets, so every local measurement is contaminated. The usual
consequence is that a gain measured on train *overstates* what test will give.

Here it **understated** it, by 13 %. The coherent explanation: contamination makes the
model's graphs on training data **better than they are on unseen data**, so there is less
for a repair to fix. On the hidden set the graphs are worse and the repair has more work.

If that holds, the reachable band `notes/26` measured on train — **+0.047 to +0.079** — is
a *lower* bound on what is available on test, and the remaining 0.033–0.036 to the cluster
sits comfortably inside it.

**Stated limit: this is one data point.** A 1.13× ratio from n=1 is a direction, not a
coefficient, and I will not use it to project the next result.

## 3. What this settles

Three questions were open before this submission and two are now closed:

- **Does graph repair transfer at all?** Yes. This was the gate on whether motion relink
  was worth building, and it has opened.
- **Is the anatomy measured on contaminated data usable?** Yes, at least directionally —
  the bucket it targeted moved on the leaderboard in the predicted direction and size.
- **Is the pack's detector the ceiling?** Already answered no by `notes/26` (`fn_detect`
  1.72 %), and nothing here contradicts it.

## 4. Next, unchanged and now justified

**Motion relink.** `fn_mislink` is **414 edges, 3.0 % of all GT edges** and the largest
remaining bucket. Pure geometry has been swept to exhaustion against it (`notes/27`) and
repaired 12.5 %. The rest needs the model's own `edge_prob` — computed by the pack for
every candidate edge and thrown away by the ILP.

Two constraints carry forward into its design:

- `notes/27` §1: any repair that moves or re-attaches nodes **trades mislinks for detection
  failures**, ~4:1 where it works and inverting when pushed. Relink needs the same kind of
  bound that `max_shift_um` gives smoothing.
- The pack emits ~54 forks per 24 datasets and divisions are worth 0.1 of the metric.
  A relink that re-solves assignments wholesale would destroy them, so forks must be
  **protected**, not re-derived.

It needs a fresh prediction pass caching candidate edges **with probabilities** — the
current cache has none.

Banked floor **0.752**. Current submission **0.880**. Cluster **0.913–0.916**.
