# +0.0221 on train — and the asymmetry story I told twice was wrong

`claude_ilp_sweep2`, 24 budget-stratified datasets (10 × `44b6`, 14 × `6bba`), 34 arms.
Both control checks reproduced **exactly**: `control` 0.8806, `asym0.1_0.5+repair` 0.8958.

```
asym0.1_0.5+repair    0.8958    <- what scored 0.883 on the LB
ratio0.4_2.0+repair   0.9179    +0.0221    <- new best, appear 0.4 / disappear 2.0
sym1.0+repair         0.9132    +0.0174
asym0.1_2.0+repair    0.9085    +0.0127
```

That is the largest train-measured gain in this project by a factor of five. Two of the
five predictions failed, and both failures are more useful than the passes.

---

## 1. 🚨 Correction: the gain was MAGNITUDE, not the lab's asymmetry ratio

`notes/31` §2 and `notes/32` §1 both claimed the win came from the *asymmetry* — the same
lab's 5× appear/disappear ratio transferring to this metric. `notes/32` called it "the
first time in this project a primary-source constant from a different acquisition
transferred." **That claim does not survive a properly matched control.**

```
asym0.1_0.5   0.8849   vs   sym0.5   0.8980     -0.0132   symmetric WINS
asym0.1_1.0   0.8909   vs   sym1.0   0.9036     -0.0127   symmetric WINS
```

`notes/31` §2 asserted the v1 grid "paired every asymmetric arm with a symmetric arm at
matched magnitude." **It did not, for the winning arm.** v1's symmetric arms were 0.02 and
0.25; the winning arm was `asym(0.1, 0.5)`, whose disappearance is **2× the largest
symmetric control in the grid**. The +0.0017 I attributed to asymmetry was a magnitude
difference wearing an asymmetry label. My own design note said the pairing existed
specifically so a gain "could not be attributed to the wrong cause," and then I did not
check that the pairing covered the arm that won.

The 0.880 → 0.883 leaderboard gain is real. Its explanation was not: raising
`disappearance` from 0.1 to 0.5 helped because it was **larger**, not because it was
**asymmetric**.

## 2. Divisions come from the termination penalty, not from `division_weight`

The mechanism finding, and it is a good one. `division_weight` saturates and prices to a
wash (`notes/31` §1, replicated externally in `notes/33` §3). But the appear/disappear
*magnitude* creates forks that actually score:

```
weights                        forks    div_J
control        (0.1 / 0.1)        54    0.0000
sym0.5         (0.5 / 0.5)     4,617    0.0787
sym1.0         (1.0 / 1.0)     3,661    0.0849
ratio0.4_2.0   (0.4 / 2.0)     1,443    0.1154   <- best, and the FEWEST forks of the three
```

Coherent mechanism: a high disappearance penalty makes the solver unwilling to *end* a
track. When a cell divides, terminating one branch becomes expensive, so it forks instead.
And `ratio0.4_2.0` reaches the best `div_J` with **a third of `sym0.5`'s forks** — these are
precision-selected forks, not volume.

`div_J = 0.1154` lands squarely in the range `notes/33` found competitors reporting (kevin
park 0.12, mikelou1 0.3) on a term we have scored **0.000** on all project. The division
term is open.

Separately, prediction 3 passed: `division_weight` **is** humped in the gap v1 never
sampled — `div0.7` (0.8857) beats both endpoints (0.8806 / 0.8839), with fork counts
544 / 1,779 / 3,889 confirming the knob is a dial there, not a step.

## 3. Where the +0.0221 actually comes from — and half of it is the node budget

```
                        raw edge_J    adj_edge    0.1 x div_J    total
asym0.1_0.5+repair        0.9050       0.8958       0.0000       0.8958
ratio0.4_2.0+repair       0.9047       0.9064       0.0115       0.9179
                          -0.0003      +0.0106      +0.0115      +0.0221
```

**Raw linking quality is unchanged (−0.0003).** The gain is two things: the division term
(+0.0115) and the **node-budget multiplier** (+0.0106).

Backing out the multiplier `1 − 0.1·(N_pred − N_total)/N_total`: the current submission runs
~10 % **over** budget, and `ratio0.4_2.0` runs ~2 % **under** it. Fewer surviving nodes,
which the metric rewards directly. The cost is visible and large — 363 more unmatched GT
nodes than `control+repair` — and on train the trade pays anyway.

**This is a legitimate part of the metric, but it is not better tracking, and it is the
half of the gain most at risk on test.** The budget comes from per-dataset GT metadata that
exists on test too, so the mechanism transfers; what may not transfer is the *calibration* —
2 % under on train could be well past the optimum on a hidden set where the model does worse
and drops more nodes anyway. Stated now so a smaller LB gain is not a surprise.

## 4. All three axes are still climbing at the grid boundary

Prediction 4 failed for the second sweep running:

```
disappear   0.5      0.75     1.0      1.5      2.0
score      0.8849   0.8900   0.8909   0.8956   0.8975   <- still rising
det tax      +46     +120     +141     +244     +330    <- also still rising
```

`asym0.1_2.0`, `sym1.0` and `ratio0.4_2.0` are each the largest arm on their axis and each
won it. The `fn_detect` tax rises monotonically, so a turn must exist — it is simply not
inside this grid either. **Do not treat `ratio0.4_2.0` as near-optimal.**

## 5. Next

1. **`claude_ilp_sweep3`** — push all three axes past the boundary, on the same cache, no
   GPU. The interesting region is now appear/disappear magnitudes of 2–8 and the
   appear:disappear *ratio* as its own axis, since `ratio0.4_2.0` (1:5) beat `sym1.0` (1:1)
   at comparable magnitude while `asym0.1_0.5` (1:5) lost to `sym0.5` — the ratio's sign
   flips with scale, which nothing here explains yet.
2. **Submit `ratio0.4_2.0 + repair`.** +0.0221 against the +0.0037 that bought the last
   +0.003, and `notes/24` §2 says train is contaminated so the leaderboard is the only
   honest measure. **Asking before spending the slot.**
3. Gate `insert_divisions` on deepcenter (`notes/34` §3) — now clearly more attractive,
   since §2 shows the division term is live and reachable.

**Method note.** §1 is the second sampling/design error in two runs (`notes/34` was the
first). Both were controls that did not cover the case they were cited for. Any future
"attributable to X" claim gets its control checked against the *winning* arm specifically,
not against the grid in general.

Banked floor **0.752**. Best scored **0.883**. Bronze **0.926**, gold **0.944**.
