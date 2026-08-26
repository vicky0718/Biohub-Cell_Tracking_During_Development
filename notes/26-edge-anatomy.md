# The edge loss, split — and I was wrong about the biggest lever

`claude_edge_anatomy` v1, 24 budget-stratified datasets, 12 arms, 377 s, no GPU.
Control reproduced **0.8806 exactly**, so everything below is readable.

| prediction | result |
|---|---|
| 1. control reproduces 0.8806 ± 0.0005 | **PASS** (0.8806) |
| 2. buckets account for every GT edge | **PASS** (13,832 = 13,832, asserted per dataset) |
| 3. `prune_isolated` non-negative everywhere | **PASS** (24/24) |
| 4. `linefit_smooth` gains **less** than +0.005 | **FAIL — it gained +0.0086** |

Prediction 4 is the result. The rest is scaffolding.

---

## 1. The anatomy

```
bucket             count    share of GT edges
tp                12,909         93.33%
fn_mislink           473          3.42%     <- largest failure
fn_detect            238          1.72%
fn_gap               212          1.53%
fn_nonconsec           0          0.00%
TOTAL             13,832
                                            plus 669 false-positive edges
of the mislinks: source already linked 336, target already claimed 280
```

`edge_jaccard = 12,909 / (12,909 + 669 + 923) = 0.8902` — the buckets reconstruct the
metric exactly, which is what makes them worth acting on.

**Detection is not the ceiling.** `fn_detect` is **1.72 %**. The branch of `notes/25`'s
plan that said "if detection dominates, bank the wins and stop building repairs" is
closed — it does not dominate, it is the *smallest* recoverable bucket. The cells are
found; the graph joins them wrong.

### What graph repair can reach, in principle

Repairing every gap and every mislink means 685 edges move from FN to TP. Whether the 473
mislinked *predicted* edges also stop being FPs depends on whether the relink lands on the
right target, so the ceiling is a band:

| | edge Jaccard | gain |
|---|---|---|
| now | 0.8902 | — |
| every gap+mislink repaired, FPs unchanged | 0.9375 | **+0.047** |
| …and the mislinked FPs become correct | 0.9691 | **+0.079** |
| **the gap to the cluster on the leaderboard** | | **+0.046** |

**The reachable band contains the whole gap to the cluster.** That is the first time in
this project that a measured ceiling has been shown to cover the deficit rather than fall
short of it. It does not say we will reach it; it says the path is not arithmetically
closed, which is more than could be said for the detector, the budget, or divisions.

## 2. 🚨 I was wrong about position repair, and wrong in a way I have been wrong before

I predicted `linefit_smooth` would gain **less than +0.005**, reasoning that node recall is
0.995 so nearly every annotated cell is already matched and position repair has almost
nothing left to recover.

It gained **+0.0086** — and it is the largest single lever in the run:

```
only_smooth         +0.0086      <- biggest solo arm
all                 +0.0113
all_minus_smooth    +0.0025      <- smoothing is +0.0088 of the +0.0113
```

**78 % of the entire repair gain comes from moving node positions**, with no change to the
node set or the edge set at all.

### Why the reasoning failed

Node recall asks *"is this GT node matched by something?"* Matching is a bipartite
assignment **within a frame**, and the pack predicts 5,000–57,000 nodes against 50–1,950
annotated ones. So a GT node is almost always matched — but frequently to a **nearby wrong
prediction**, one that is not on the true track. Its edges then cannot score. Smoothing
pulls the true track's node closest, the assignment flips to it, and its edges start
counting. **Node recall is identical across that change. Edge Jaccard is not.**

This is the *same class of error* as `notes/21`: a summary statistic that is blind to the
exact failure being fixed. That note measured a 0.074 edge-Jaccard gap across two arms with
**identical 0.866 node recall**, and `pipeline/detector.py::paired_recall` exists because
of it. I then used node recall to bound position repair anyway. Recording that plainly:
**the mechanism is inferred, not measured** — this run computed the anatomy only for
`control` and `all`, so it cannot attribute smoothing's gain to a bucket. Run 2 measures it.

## 3. Two repairs I built were no-ops the ILP had already handled

```
only_prune    +0.0000    nodes 513,025 -> 513,025   (no isolated nodes exist)
only_parent   +0.0000    edges unchanged            (no merges exist)
```

The ILP already emits a graph with no isolated nodes and no merges. Prediction 3 "passed"
on 24/24 datasets by being vacuously true. `notes/25` argued `prune_isolated` was free
points because the multiplier sits at 0.9892 — the reasoning was sound and the situation
simply does not arise. Both stay in `pipeline/repair.py` (tested, cheap) but neither
belongs in the chain.

**And one repair is actively harmful:**

```
only_caplen        -0.0002
all_minus_caplen   +0.0115   <- better than `all` at +0.0113
```

The 14 µm edge cap costs points. Leave-one-out earned its place on its first outing: a solo
arm alone would have shown it as ~zero and it would have stayed in the chain.

## 4. Where the run leaves us

| arm | score | delta |
|---|---|---|
| control | 0.8806 | — |
| **all_minus_caplen** | **0.8921** | **+0.0115** |
| all | 0.8919 | +0.0113 |
| only_smooth | 0.8892 | +0.0086 |
| only_gapclose | 0.8833 | +0.0027 |

Per-dataset the spread is large — `44b6_66f9292d` **+0.0575**, `44b6_18ced818` **+0.0510**,
and eight datasets at +0.0000. Gains concentrate on the datasets that start worst.

Gap closing is under-firing: it recovered **30 of 212 gaps (14 %)** at a 5.75 µm one-frame
radius, for 8,576 added nodes. The radius is probably too tight, and the cost is paid in
budget whether or not the bridge lands.

## 5. Next: sweep the thing that actually worked, before building the thing I planned

`notes/25` named motion relink as Run 2 if mislinks dominated, and they do (3.42 %). That
is still right *eventually* — but it needs a fresh prediction pass caching candidate edges
**with `edge_prob`** (the current cache has none), which is 27 minutes plus implementation.

Smoothing needs none of that. It is already the biggest lever, it runs on the existing
cache in minutes, and **every one of its parameters is currently set to a number I copied
from the public notebook's config rather than measured**: `weight=0.76`, `window=2`,
`max_shift_um=3.2`. Sweeping them is nearly free and strictly better-informed than building
relink on the assumption that 0.76 was optimal for our graphs.

So: **Run 2a is the smoothing sweep** (`claude_smooth_sweep`), on the same cache —
`weight × window`, the shift bound, a second pass, and gap-close radius, **with the anatomy
computed on every arm** so §2's mechanism is measured rather than assumed. **Run 2b is
motion relink**, with a prediction pass that caches `edge_prob`.

Banked floor **0.752**. Current submission **0.867**. Cluster **0.913–0.916**.
