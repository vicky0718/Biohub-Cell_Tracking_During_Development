# We have been optimising noise, and the headroom is somewhere we stopped looking

Two findings, one method and one target. The method finding invalidates a chunk of
`notes/40` and `notes/41`. The target finding is the first thing in this project with
enough headroom to matter.

---

## 1. At n=12 the smallest effect we can resolve is 0.0036

Taking `claude_config_sweep2`'s own per-dataset numbers and running a **paired** test —
pairing cancels dataset difficulty, which is enormous here (scores span 0.85–1.06, sd
0.064) next to the effects being tested:

```
comparison              mean Δ      sd       SE       t     verdict
det 0.975 vs 0.98      +0.0021   0.0058   0.0017   1.24    not resolved
det 0.975 vs 0.97      +0.0018   0.0032   0.0009   1.94    not resolved
det 0.975 vs 0.965     +0.0014   0.0021   0.0006   2.32    resolved, barely
```

```
effect     datasets needed
0.0036          11        <- what we have
0.0020          34
0.0015          61
0.0010         136
```

Everything chased since `notes/34` is +0.001 to +0.002. **So the three "located interior
optima" in `notes/41` are not located.** The threshold optimum was claimed twice, on two
grids, and neither grid could distinguish 0.975 from 0.98. That claim is withdrawn; the
setting may still be right, but we do not know it.

It also explains the transfer ratio nobody could pin down — (1.04, 1.22), (0.54, 1.08),
(0.59, 0.68), (0.28, 0.83), ranges that do not overlap. Dividing an unresolved train delta
by a rounded leaderboard delta is noise over noise, and four attempts at it produced
exactly the spread you would predict.

### The leaderboard cannot resolve them either

The test set is **4 datasets**. Even paired against our own previous submission, a
0.002 difference in the mean over 4 datasets is inside the sampling error. So
0.897 → 0.899 → 0.901 is one flat line with error bars, not a trend. We have been reading
a ranking off a ruler with no marks on it.

## 2. `claude_secondary` passed all five predictions and the result is not real

The second-model ensemble ran cleanly. Every pre-registered prediction passed, including
the sharp one:

```
BEST ARM: w0.3_low_margin at 0.9564  (+0.0029 vs control)
```

Paired, on the same 12 datasets:

```
arm                    mean Δ       sd       SE       t     verdict
w0.15_low_margin      +0.0004   0.0124   0.0036   0.12    not resolved
w0.3_low_margin       +0.0026   0.0143   0.0041   0.63    not resolved
w0.3_fixed            -0.0019   0.0151   0.0044  -0.43    not resolved
w0.5_fixed            -0.0178   0.0361   0.0104  -1.70    not resolved
```

**t = 0.63.** The ensemble's per-dataset swings are large and two-sided — one dataset goes
0.8552 → 0.8235, another 0.9746 → 0.9917 — so its paired sd is 0.0143, two and a half
times the config sweep's. Resolving +0.0026 against that needs **n ≈ 147**.

The mechanism is not refuted. It is **unmeasured**, and the five green PASS lines say
nothing about it. A pre-registered prediction with a threshold below the measurement's
resolution is not a test; it is a coin flip with a paper trail. Every prediction in this
project that read "beats the control by more than 0.001" was one.

## 3. ~~The headroom is in divisions, and we traded it away~~

> **❌ WITHDRAWN by `notes/50`.** Nothing was traded away. `claude_divsweep` measured
> `div_J` on one dataset sample across eight post-processing chains and it is **0.1154 on
> every one** (1,443 forks on every one) — the config makes no difference to the division
> term at all. The 0.1154-vs-0.0645 gap below compares n=24 against n=12, and `notes/44`
> measured that n=12 sample as easy by +0.0116. `notes/43` had already cut this section's
> +0.0935 to +0.016 on the 151-event count; `notes/50` removes the remaining premise.
> The reasoning about what `div_J` *would* be worth is still arithmetically correct — it is
> the claim that we once had it and lost it that does not survive.

```
score = adjusted_edge_jaccard + 0.1 · division_jaccard        (max 1.1)
```

Current best configuration, measured this run:

```
edge_J 0.9449    div_J 0.0645
divisions contribute   0.1 × 0.0645 = 0.0065
divisions could contribute            0.1000
                            headroom  0.0935
```

**We recover about 6% of divisions.** The gap to bronze is 0.027 — divisions alone hold
three and a half times that. And `div_J` across this project's history reads:

```
0.0000 → 0.0263 → 0.1023 → 0.1047 → 0.1154 → 0.0645  (current best config)
```

`notes/35` measured **0.1154** at the same ILP weights. The config audit then optimised the
*total* and gave half of it back, which nobody noticed because the total went up. A term
worth up to 0.1 was being spent to buy a few thousandths of edge jaccard.

Unlike everything above, this is a **large** effect. Going from 0.065 to 0.35 is +0.028 —
eight times the n=12 resolution limit, and by itself the whole bronze gap. It does not need
more datasets to see; it needs to be worked on.

`division_jaccard = TP / (FP + D)` with D fixed by ground truth, so it is a recall-shaped
quantity with a false-positive penalty: predicting no divisions scores 0, and predicting
many scores badly too. `notes/39` established the mechanism — **appearance** cost creates
forks (53 → 2,477 as it goes 0.13 → 2.0), disappearance suppresses them. That knob was
swept for total score, never for `div_J`.

## What follows

1. `claude_widecv` — n=60, the 12 prior datasets as a strict **superset** so old numbers
   stay comparable, paired-test grading, a time guard. Running. It settles whether the
   config axes are real or were noise all along.
2. Divisions become the direction. The target is `div_J`, not the total, and the first
   question is what the appearance/disappearance surface looks like when `div_J` is what
   is being read off it.
3. Stop pre-registering thresholds below the resolution limit. Any prediction of the form
   "beats the control by more than 0.001" at n=12 is unanswerable, and five of them just
   passed on a result with t=0.63.

Banked floor **0.752**. Best scored **0.899**. `claude_submit_config` pending.
Bronze **0.926**, gold **0.944**.
