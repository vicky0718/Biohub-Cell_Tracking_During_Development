# We are already 12.9% under the node budget, and that retires the whole thesis

`claude_budget2`, 24 cached instances, 10 arms, 605 s, CPU. 3 of 5 predictions passed and
the crux failed — but the table's third-from-last column is the finding, not the grading.

```
arm             total  adj_edge   edge_J   div_J    nodes    ratio    mult    forks
none           0.9188    0.9072   0.9037  0.1154   18,032   -0.129   1.013   1,443
isolated       0.9188    0.9072   0.9037  0.1154   18,032   -0.129   1.013   1,443
geometry1.0    0.9129    0.9014   0.8940  0.1154   17,490   -0.160   1.016   1,443
geometry0.9    0.9019    0.8904   0.8774  0.1154   16,656   -0.209   1.021   1,443
geometry0.8    0.8737    0.8621   0.8424  0.1154   15,513   -0.280   1.028   1,443
geometry0.7    0.8266    0.8150   0.7893  0.1154   14,062   -0.359   1.036   1,443
```

Reproduction was exact — `none` = 0.9188, `div_J` 0.1154, 1,443 forks, all three matching
`claude_divsweep` to the digit.

---

## 1. `ratio = −0.129`. There was never anything to collect.

**We are already predicting 12.9% FEWER nodes than `estimated_number_of_nodes`**, and the
budget multiplier is already paying us 1.013. `notes/51` said no run had ever measured where
we sit against `N_est`; this is that measurement, and it removes the premise the last three
runs were built on.

Every budget attempt since `notes/45` assumed there was over-prediction to remove. There is
not. So cutting further just deletes real edges to buy a multiplier we already hold:

```
factor    nodes     ratio    mult      edge_J     total
none     18,032    -0.129   1.013      0.9037    0.9188
1.0      17,490    -0.160   1.016      0.8940    0.9129
0.7      14,062    -0.359   1.036      0.7893    0.8266
```

The multiplier climbs 1.013 → 1.036 while `edge_J` falls 0.9037 → 0.7893. **The bonus is
capped in practice by how little it is worth: 0.1 per unit of ratio, against an edge term
that costs far more per node removed.**

This is why all three selection rules failed, and they failed for one reason rather than
three:

```
notes/46   pool_kernel_um, an NMS radius        recall 0.983 -> 0.537
notes/48/49  det_threshold, a confidence cut    0.901 -> 0.863 on the LB
notes/52   track ranking under a per-dataset cap   monotonically worse
```

`notes/51` proposed this run as *"the third selection rule"* on the theory that cutting
after linking would behave differently from cutting at detection. It does behave differently
— it is far gentler, losing 0.0059 at factor 1.0 where confidence thinning lost 0.038 — but
gentler is not positive. **The node budget is closed, and closed at the premise rather than
by exhaustion.**

## 2. `isolated` is exactly zero, and that is informative

`isolated` is identical to `none` on every column: 18,032 nodes, 1,443 forks, the same
score to four decimals. `prune_short_tracks(min_frames=6)` already leaves **no edgeless
node** in the graph — a singleton spans one frame and is removed by the span test. The free
half of the budget lever was already being collected, silently, by a stage added for a
different reason.

Prediction 2 passed on a technicality (0 is non-negative) and should be read as "no such
nodes exist" rather than as evidence about the metric.

## 3. Tightness carries nothing

```
geometry1.0  0.9129   vs  length1.0  0.9126    +0.0003
geometry0.9  0.9019   vs  length0.9  0.9012    +0.0007
geometry0.8  0.8737   vs  length0.8  0.8736    +0.0001
geometry0.7  0.8266   vs  length0.7  0.8277    -0.0011
```

3 of 4 to `geometry`, by +0.0001 to +0.0007 — an order of magnitude under `notes/44`'s
0.0015 floor. r35's `rank_tracks_by_geometry` is, for our graphs, just "keep long tracks."
Prediction 4's PASS is not a result.

## 4. A grading defect, recorded

Prediction 5 printed **FAIL** with the message *"wins on one embryo, loses on the other."*
That message is wrong. The best non-control arm was `isolated`, which is byte-identical to
the control, so every per-embryo delta was exactly `+0.0000` and my `all(x > 0)` test
rejected zeros. The correct verdict is **NOT GRADED — no arm differs from the control**.
The `CLOSED` summary is right; that one diagnostic line is not. Fixed in the builder so the
zero case reports honestly rather than borrowing `notes/49`'s language for a run that never
had an effect to transfer.

## 5. The measurement points somewhere, and it is the opposite direction

Put §1 beside `notes/51`:

```
ratio       -0.129     12.9% UNDER budget, multiplier 1.013
fn_detect      583     4.21% of GT edges, an endpoint never matched -- the largest bucket
```

**We over-corrected.** The chain deletes so many nodes that it misses 4.21% of ground-truth
edges through undetected endpoints, and banks a 1.3% multiplier bonus for it. `notes/35`
recorded the crossing — *"~10% OVER budget → ~2% under it"* — and the ILP weights plus
`prune_short_tracks(6)` have since carried us to 12.9% under.

The arithmetic of going back the other way, at `edge_J = 12,962 / 14,343`:

```
give up the multiplier entirely (ratio -0.129 -> 0)      -0.0118 on adj_edge
break-even needs ~169 of the 583 fn_detect edges recovered
```

So loosening detection pays if roughly **29% of the undetected-endpoint edges come back**
for a 13% rise in node count.

And that region has never been swept. Every threshold grid in this project:

```
notes/40   0.99   0.985   0.975   0.96875
notes/41   0.98   0.975   0.97    0.965
notes/44   0.98   0.975   0.97
notes/48   0.975  0.999   0.9999   0.99999   0.999999
```

**The lowest value ever tried is 0.96875.** `notes/44` called the surface flat over
[0.965, 0.99] and `notes/49` found a cliff above it — but *below* 0.965, where node count
rises and the budget bonus is spent rather than hoarded, is unexplored. That is the one
axis this run argues for, and unlike the last three it spends the budget instead of chasing
it.

It needs a GPU prediction pass — the cached instances are fixed at `det_threshold=0.985` —
so it is not free, and `notes/49`'s rule applies: grade per embryo, both means, or it does
not transfer.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.935 bronze    0.947 gold
node budget: CLOSED at the premise (this note)   config: closed   divisions: closed
open: detection, and the unswept region below 0.965
```
