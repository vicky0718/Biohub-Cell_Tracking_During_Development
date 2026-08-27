# The solve's own weights were worth more than either relink — and divisions are not dead

`claude_ilp_sweep` v1, 24 budget-stratified datasets, 26 arms, 6,915 s. Control reproduced
**0.8806 exactly** by re-solving cached instances, so the cache is a faithful ILP input and
the run is readable.

| prediction | result |
|---|---|
| 1. control reproduces 0.8806 ± 0.0005 | **PASS** (0.8806) |
| 2. lowering `division_weight` raises fork count | **PASS** (54 → 7,149) |
| 3. more forks does **not** lift `division_jaccard` above 0.01 | **FAIL — it reached 0.0562** |
| 4. asymmetric appear/disappear beats the symmetric default | **PASS**, and genuinely |

```
control              0.8806
control+repair       0.8921   (+0.0115)   <- the chain that scored 0.880 on the LB
asym0.1_0.5+repair   0.8958   (+0.0152)   <- best; the ILP change adds +0.0037 on top
```

Two relink attempts (`notes/29`, `notes/30`) tried to *override* this solve and both failed.
Changing one of its weights beat both, which is the shape the evidence had been pointing at.

---

## 1. 🚨 Divisions are reachable after all — and cost exactly what they earn

`notes/25` measured geometric fork insertion at **1 TP per 2,223 guesses** and closed the
division term. `notes/26` recorded the ILP's own forks as *unevaluable* (37 emitted, 0 TP,
0 FP). Prediction 3 was written expecting a third confirmation.

It failed. Lowering `division_weight` takes `division_jaccard` from **0.0000 to 0.0562**.
**Model-driven forks score where geometric ones did not.** That distinction is real and I
had been treating "divisions are dead" as settled on evidence that only covered geometry.

But the trade is a near-exact wash:

```
                 control+repair -> div0.5+repair
edge_jaccard        0.9029    ->    0.8971     = -0.0058
0.1 x division_J    0.0000    ->    0.0056     = +0.0056
                                    net          -0.0002
forks                   54    ->     7,149     (132x)
```

132× more forks buys 0.0056 of score on the division term and gives back 0.0058 on the
edge term. The two cancel to within 0.0002. That is not "dead" — it is **live and priced
at zero**, which is a different and more useful fact, because a price can move.

**The unexplored region is `division_weight` between 0.5 and 1.0.** Every swept value below
1.0 — 0.5, 0.2, 0.0, −0.5 — gives *byte-identical* results (7,149 forks, `div_J` 0.0559).
The knob saturates immediately. So the sweep measured two points, not five: the default
(54 forks, `div_J` 0.0000) and saturation (7,149 forks, `div_J` 0.0562). Anything
interesting lives strictly between them and was never sampled.

Note also that `edge-2.0` and `edge-4.0` reproduce the saturated solution exactly. That is
consistent rather than coincidental: scaling `edge_weight` up shrinks `division_weight`
*relative* to the objective, so both knobs approach the same corner. They are not
independent axes near saturation.

## 2. The asymmetry is real, not a magnitude effect in disguise

`notes/03` §3 recorded the same lab's own zebrafish Ultrack config using
`appear_weight = -0.002`, `disappear_weight = -0.01` — a deliberate **5×** asymmetry,
"discouraging track termination more than initiation". The pack ships symmetric `0.1 / 0.1`.

The grid paired every asymmetric arm with a **symmetric arm at matched magnitude**
specifically so a gain could not be attributed to the wrong cause:

```
asym0.1_0.5   +0.0043     (appear 0.1, disappear 0.5 — the lab's 5x ratio)
sym0.25       +0.0026     (best symmetric)
                +0.0017   attributable to asymmetry itself
```

The winning arm keeps appearance at the pack's 0.1 and raises disappearance 5× — the lab's
own ratio, at the pack's scale. **A primary-source constant from a different acquisition
transferred to this metric**, which is the first time in this project that has happened.

## 3. What the anatomy says at the best arm

```
                control+repair -> asym0.1_0.5+repair
fn_mislink            411     ->      382
fn_gap                177     ->      159
fn_detect             218     ->      254
edges added        +17,152    ->   +11,678
```

The gain comes from **mislinks and gaps together**, not one bucket — and it pays the same
detection tax `notes/27` §1 identified for position repair (+36 undetected). Consistent
mechanism: raising the disappearance penalty makes the solver reluctant to end a track, so
it links through ambiguity it previously abandoned. Some of those links are right (−29
mislink, −18 gap) and some strand nodes that no longer match (+36 detect). Net positive
here, and that ratio is what would invert if pushed further — `asym0.1_0.5` is the largest
asymmetry swept, so **the optimum may lie beyond the grid**.

## 4. Next

1. **Submit `asym0.1_0.5 + repair`.** +0.0152 against the +0.0115 that scored 0.880. It is
   a one-line change to `claude_submit_repair`'s ILP call, and the leaderboard is the only
   honest measure (`notes/24` §2 — training data is contaminated for these weights).
   **Asking before spending the slot.**
2. **Sweep `division_weight` in (0.5, 1.0)** — 0.9/0.8/0.7/0.6 — where the knob is not
   saturated. §1 shows the two endpoints price out to zero; the question is whether an
   intermediate fork count buys the division term without the full edge cost. Free: it
   re-solves the same cached instances.
3. **Extend the asymmetry grid past 0.1/0.5**, since the best arm sits on the grid boundary.

Banked floor **0.752**. Best scored submission **0.880**. Cluster **0.913–0.916**.
