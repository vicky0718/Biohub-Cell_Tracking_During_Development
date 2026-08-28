# The audit paid: +0.0036 from settings we never tested, and the frozen threshold was real

`claude_config_sweep`, 12 datasets, 4 detection thresholds × 6 post-processing combinations
= 24 cells, one run. **All five predictions passed** — the first clean sweep in this project.

```
det        m0g1     m3g1     m6g1     m0g2     m6g2     m3g2      nodes
0.99      0.9499   0.9499   0.9490   0.9510   0.9519   0.9510   260,477   <- the submission
0.985     0.9500   0.9500   0.9500   0.9508   0.9522   0.9508   266,912
0.975     0.9514   0.9514   0.9519   0.9515   0.9535   0.9515   273,077   <- best
0.96875   0.9503   0.9503   0.9502   0.9506   0.9526   0.9506   274,992
```

**Best cell `d0.975_m6_g2` = 0.9535, +0.0036** over the chain that just scored 0.899.
Anchor cell reproduced 0.9499 exactly, so the grid is anchored to the live submission.

---

## 1. The freeze was not harmless

`notes/28` froze `DET_THRESHOLD` at 0.99 *deliberately*, so one leaderboard delta would
isolate the repair — and then it was never unfrozen. `notes/39` called that the audit's
headline item and made prediction 5 sharp enough to refute it: *"if 0.99 wins, the freeze
was harmless and the audit is wrong."*

```
det=0.99      best cell 0.9519
det=0.985     best cell 0.9522
det=0.975     best cell 0.9535   <- best, and INTERIOR
det=0.96875   best cell 0.9526
```

0.99 is the **worst** threshold in the grid. The optimum is interior at 0.975 — for once
not on a boundary — and worth **+0.0016** on its own. A setting frozen for one controlled
comparison, months of work ago, has been costing that ever since.

## 2. The three levers are roughly additive, and one only works with another

Reading the det=0.99 row against the best cell:

```
m0g1  0.9499   baseline
m0g2  0.9510   +0.0011   two-frame gap closing
m6g2  0.9519   +0.0009   ... plus short-track pruning
d0.975_m6_g2   +0.0016   ... plus the threshold        = +0.0036 total
```

**Short-track pruning alone is negative** (`m6g1` at det=0.99 is −0.0009). It only pays once
two-frame gaps are being closed. That is a real interaction and the mechanism is legible:
bridging wider holes joins fragments into longer tracks, so pruning what is left removes
genuine junk instead of amputating pieces of real tracks. Testing either alone would have
concluded the wrong thing about it — which is an argument for the batched design, not just
for these settings.

## 3. 0.897 → 0.899, and the fourth transfer measurement

`claude_submit_bidir` scored **0.899**. Rank ~1297/2792.

```
                 train      LB delta          implied ratio
repair chain    +0.0115   (0.0120, 0.0140)   (1.04, 1.22)
ILP asymmetry   +0.0037   (0.0020, 0.0040)   (0.54, 1.08)
ratio0.4_2.0    +0.0221   (0.0130, 0.0150)   (0.59, 0.68)
bidirectional   +0.0036   (0.0010, 0.0030)   (0.28, 0.83)
```

**Direction has now transferred five times out of five.** Magnitude still converts by no
fixed factor, and this is the weakest ratio yet. At +0.002 per experiment, the remaining
+0.027 to bronze needs roughly a dozen more successes — which is the arithmetic behind
`notes/39`'s complaint about throughput, and the reason this run tested 24 cells instead of
one mechanism.

## 4. What the sweep says to do next

Two axes are not exhausted:

- **`max_gap` is on the boundary.** Only 1 and 2 were tried and 2 won everywhere. 3 is
  untested and free.
- **`min_len` is coarse.** 0, 3, 6 — and 6 beat 3 everywhere it mattered. 8 and 10 untested.
- **The threshold grid is coarse** around an interior optimum: 0.98 and 0.97 are unsampled.
- **`GAP_DENSITY_ADAPTIVE`** (their gain 0.040) is still untested entirely.

That is a second batched sweep, same 4 GPU passes, roughly twice the cells. It runs
alongside the submission of the current best rather than after it.

**And the model thesis is not dead, it is just demoted.** `notes/33` put ~0.04 in the two
missing models; measured so far, model-level mechanisms have returned ~0.002 (deepcenter)
and +0.0036 (bidirectional, no second model needed), while *configuration* has returned
+0.0036 in a single run. The temporal linker remains the only untested item that adds
capacity rather than re-reading what we have, but it is no longer the obvious next thing.

Banked floor **0.752**. Best scored **0.899** (rank ~1297/2792).
Bronze **0.926**, gold **0.944**.
