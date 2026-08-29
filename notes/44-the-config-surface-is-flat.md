# n=60 settles it: the config surface is flat, and 12 datasets were an easy draw

`claude_widecv`, 60 datasets, 3 thresholds × 6 post combos, paired grading. Two of five
predictions passed. The three failures are the result.

```
det             m6g2      m0g1      m6g1      m0g2      m8g2      m6g3
0.98          0.9356    0.9365    0.9368    0.9350    0.9330    0.9336
0.975         0.9348    0.9352    0.9357    0.9339    0.9322    0.9326
0.97          0.9348    0.9352    0.9356    0.9340    0.9317    0.9322
```

---

## 1. The superset held. The old sample was lucky.

All 12 previously-measured datasets are among the 60 — checked directly against
`cfg2.log`, dataset by dataset. (Prediction 1 graded FAIL for a reason that is my error,
not the data's: the candidate cache holds **24** datasets, not 12, so it compared the
incumbent on 24 against a number measured on 12. The per-dataset comparison is the right
check and it passes.)

```
mean on the 12 seed datasets   0.9450
mean on the other 48           0.9333
mean on all 60                 0.9352
```

**The 12 were an easy subset by +0.0116.** Every absolute train figure this project has
quoted — 0.9499, 0.9535, 0.9564 — is inflated by roughly that. The honest anchor is
**0.9352 at n=60**, against a leaderboard of 0.901.

## 2. The threshold optimum does not exist

```
d0.975_m6_g2 − d0.98_m6_g2  =  +0.0001   SE 0.0008   t 0.12   n=60
```

At n=12 this read +0.0021. At n=60 it is **one ten-thousandth**. `notes/40` located it,
`notes/41` re-confirmed it on "a second, finer grid that shares no other point", and
`notes/42` withdrew the claim on power grounds. This is the direct measurement: the effect
is zero. Both notes were reading noise, and the second grid agreeing with the first was
two samples of the same noise, not independent confirmation.

## 3. The best cell moved, and the repair chain does not replicate

Best at n=60 is **`d0.98_m6_g1`** — `gap=1`, reversing `notes/40`'s "two-frame gap closing
wins everywhere". It beats the incumbent by +0.0010 at **t = 0.85**, so the new ranking is
no more real than the old one. A ranking that reorders when you add datasets and cannot
clear t=2 either way is not a ranking.

Worse for the record, at det 0.975:

```
m6g2  0.9348      the repair chain
m0g1  0.9352      no gap closing, no pruning
                  −0.0004
```

`notes/40` measured that same pair at **+0.0020** and called the levers "roughly additive".
At n=60 it is negative and unresolved. The audit's headline **+0.0036 does not replicate**;
only its 0.99 → 0.975 component is untested here, since 0.99 was not in this grid.

## 4. What *is* real, and it is all negative

Five arms resolve, every one of them worse than the incumbent:

```
d0.975_m6_g3   −0.0023   t −3.76      gap=3 is genuinely worse than gap=2
d0.98_m6_g3    −0.0023   t −2.66
d0.97_m6_g3    −0.0031   t −2.50
d0.97_m8_g2    −0.0057   t −2.10      min_len=8 is genuinely worse than 6
d0.98_m8_g2    −0.0055   t −2.07
```

The only durable config knowledge this project has produced is **which settings to avoid**.
Nothing it ever claimed to have *found* survives measurement.

## 5. The floor, and why it closes the direction

The paired sd at n=60 is **0.0105**, not the 0.0058 `notes/42` estimated — because that
estimate was taken from the easy 12. So:

```
n= 12   resolves 0.0061
n= 60   resolves 0.0027
n=199   resolves 0.0015     <- every dataset the competition gives us
```

**Even using all 199 training datasets, the smallest measurable effect is 0.0015.** Every
config result this project has produced is at or below that line. The test set is 4
datasets, so the leaderboard cannot see them either.

This is not "we have not found the optimum yet". It is: **the config surface is flat inside
the resolution of every measurement available to us.** There is no experiment left in that
space worth running, because no outcome of one could be believed.

## 6. Where that leaves the work

The bar is now explicit: **a change must be worth more than ~0.0015 to be measurable at
all, and more than ~0.01 to be worth a submission slot.** Nothing in configuration,
ensembling, or graph repair has cleared it. Ranked by whether they *could*:

- **Divisions by fork suppression** (`notes/43` §1) — `div_J = TP/(FP+D)` with D ≈ 9 per
  twelve movies; we emit tens of false forks against them. Driving FP toward zero is worth
  ~+0.016 arithmetically. An ILP-weight question, cheap, and never once asked with `div_J`
  as the quantity being read.
- **A detector trained on dense labels** (`notes/43` §3) — 1.36M labelled nodes in one
  external embryo against 133k across all 199 competition datasets, and a mechanism against
  the recorded failure rather than a hope. High variance, high ceiling.

Both are worth more than another sweep. Neither is certain. What is certain is that the
thing we have been doing for the last ten runs cannot produce a measurable result.

Banked floor **0.752**. Best scored **0.901**. Bronze **0.926**, gold **0.944**.
Honest train anchor **0.9352** at n=60, not 0.9535.
