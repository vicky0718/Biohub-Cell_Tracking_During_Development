# The sweep bought +0.0006. The constants I copied were already right.

`claude_smooth_sweep` v1, 25 arms over the same 24 cached graphs, no GPU. Control
reproduced **0.8806 exactly**. **All four pre-registered predictions passed** — and the
run is still, on its own terms, a negative result.

| | |
|---|---|
| best single smoothing arm | `w1.0_win3` **+0.0092** |
| the constant I copied from the public notebook | `w0.76_win2` **+0.0086** |
| **what tuning bought** | **+0.0006** |
| best combination this run | `anchor_smooth_gap` +0.0109 |
| best combination in `notes/26` | `all_minus_caplen` **+0.0115** |

**The sweep found nothing better than what we already had.** That is worth knowing — it
means no more time goes into position repair — but it is not progress toward 0.913.

---

## 1. The mechanism is confirmed, and it is a trade, not a free win

Prediction 2 passed: at the best smoothing arm, **57 mislinks repaired** against
**14 detection failures created**. `notes/26` §2's story survives contact with the buckets.

But the `d_detect` column tells the part I had not anticipated. For **every** pure
smoothing arm it is *positive* — smoothing pushes some nodes out of the 7 µm match radius
while pulling others in:

```
arm                 d_mislink   d_detect   score
w0.76_win2               -54        +9    +0.0086
w1.0_win3                -57       +14    +0.0092
w1.0_win5                -60       +32    +0.0078
w1.0_win8_shift7         -43      +106    -0.0045   <- the trade inverts
```

**Position repair buys mislinks and pays in detections, at roughly 4:1 where it works and
worse as it gets aggressive.** That is why the optimum is sharp, and why a bigger window
or a looser shift bound does not keep helping. `w1.0_win8_shift7` repairs 43 mislinks,
creates 106 detection failures, and goes **negative**.

Gap closing runs the other way — inserting a node can newly match a GT node, so its
`d_detect` is negative. The two compose favourably: the best chain nets `d_detect = −23`.

## 2. Both copied constants were near-optimal, which is itself information

- **Weight/window.** Peaks by window land at 0.76, 0.76, 1.0, 1.0, 0.76 — wobble, not
  structure. The surface is flat near the top and the copied `0.76/2` sits on the plateau.
  Prediction 3 technically passed (`w1.0_win3` wins) but the honest reading is that the
  public notebook's value was already right for our graphs to within +0.0006.
- **Gap radius: wider is strictly worse.**
  ```
  gap5.75   +0.0027    31 gaps repaired    +8,576 nodes
  gap8.0    +0.0007    47 gaps repaired   +10,678 nodes
  gap11.0   -0.0019    68 gaps repaired   +13,085 nodes
  ```
  More gaps close, and the score falls anyway — the node-budget cost and the new mislinks
  (−8 → +4 → +22) overwhelm the recovered edges. 5.75 µm is at or past the optimum.
- **Convergence.** A second pass adds **+0.0003**, a third is worse. One pass is enough.

**Caveat I have to state:** the best *grid* arm `w1.0_win3` sits at the top of the swept
weight range, so the boundary check applies even though it did not print (it inspects the
overall best, which was the anchor chain, not the best grid cell). Given the surface is
flat to ±0.0006, extending weight past 1.0 is unlikely to matter — but that is a judgement,
not a measurement.

## 3. Order matters, and `notes/26` had it right

`anchor_smooth_gap` (+0.0109) is smooth-then-gap-close. `notes/26`'s `all_minus_caplen`
(+0.0115) is gap-close-then-smooth. Same two operations, **+0.0006 apart**, because
smoothing after gap closing also refines the inserted midpoint nodes. Small, but it means
the chain order is a real parameter and the better one is already known.

## 4. Where this leaves the ceiling

```
mislink  473 -> 414   12.5% repaired
gap      212 -> 179   15.6% repaired
detect   238 -> 215    9.7% repaired
```

The best chain is **+0.0115 against a reachable band of +0.047 to +0.079**. We have
claimed **15–24 % of it**. Everything cheap and image-free is now spent.

### One live thread the per-dataset results opened

No single setting wins broadly. Across 24 datasets the best arm was `w1.0_win5`,
`w0.76_win1`, `gap5.75`, `w1.0_win2`, `w0.4_win1`, `anchor_smooth_gap` and plain `control`
— and gains ran from **+0.0781** (`44b6_18ced818`) to **+0.0000** on four datasets.

```
per-dataset ORACLE mean delta   +0.0183   (median +0.0118)
best global arm                 +0.0115
implied headroom                ~+0.0068
```

**This is not a claimable +0.0068.** An oracle that picks the winning arm per dataset using
the labels is not available at test time, and the estimate is an unweighted mean where the
metric is weighted. What it does say is that a *global* constant costs something real, and
that a label-free proxy for per-dataset selection (density, node count, frame count) would
be worth probing — the same per-dataset thesis `notes/25` §2 retired for the node budget,
resurfacing for repair parameters where the spread is 20× larger.

Filed, not pursued. It is smaller than the mislink bucket and less certain.

## 5. Next: motion relink, which is the only thing left that is big

`fn_mislink` is **414 edges, 3.0 % of all GT edges** and the largest remaining bucket. Pure
geometry has now been swept to exhaustion against it and repaired 12.5 %. The rest needs
information geometry does not have: **the model's own `edge_prob`**, which the pack
computes for every candidate edge and which the ILP then throws away.

That is `notes/25`'s Run 2b and it is now unambiguously the next step. It needs a fresh
prediction pass caching candidate edges **with probabilities** — the current cache has
none — so ~27 min of GPU plus the relink implementation, versus the minutes these last two
runs cost.

**Before that**, the +0.0115 chain is worth a submission: it is measured, it is cheap, and
the only honest read of its size is the leaderboard, since training data is contaminated
for these weights (`notes/24` §2). Asking before spending a slot.

Banked floor **0.752**. Current submission **0.867**. Cluster **0.913–0.916**.
