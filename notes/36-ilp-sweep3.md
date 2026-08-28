# `ratio0.4_2.0` is a real optimum. The ILP-weight direction is closed.

`claude_ilp_sweep3`, 24 datasets, 18 weight settings, 36 arms. Both controls reproduced
**exactly** (`control` 0.8806, `ratio0.4_2.0+repair` 0.9179).

**Nothing beat it.** Closest was `ap0.67_dis2.0+repair` at 0.9170 — **−0.0009**, which the
grading cell correctly refused to call a result:

```
INSIDE NOISE (<0.001). sweep2's ratio0.4_2.0 stands; do not re-submit for this.
```

That threshold is the fix `notes/34` forced, after a +0.0000 arm got called "submittable"
on a 1e-6 test. It did its job on its first outing.

The submission being evaluated right now is therefore the best arm in everything searched
across three sweeps and 69 weight settings.

---

## 1. The axes have turned, and hard

`notes/35` §4 flagged that every axis was still climbing at sweep2's boundary. Extending
them answers it: past `ratio0.4_2.0` the score **collapses**.

```
magnitude at 1:5     0.9093 -> 0.9037 -> 0.8838 -> 0.8615 -> 0.8113 -> 0.7518
fn_detect tax          +370    +760    +1,261   +1,742   +2,647   +3,560
symmetric magnitude  (sweep2) 0.8980 -> 0.9036 -> 0.9032 -> 0.8941 -> 0.8810 -> 0.8464
                       sym0.5   sym1.0   sym2.0   sym3.0   sym4.0   sym6.0
```

The symmetric axis peaks at **sym1.0**, visible only when the two sweeps are read together.
The `fn_detect` tax is what does the killing: ten times the magnitude means ten times the
GT nodes we no longer match, and the edge term cannot survive it.

**`div_J` keeps rising the whole way** — 0.1154 at the optimum, 0.1500 at `r5_1.6` where the
score has already fallen to 0.8615. So divisions are not the binding constraint. The edge
term is, and it is what pays for every extra fork.

## 2. The clean measurement sweep2 could not make

Prediction 4 passed, and it is the number `notes/35` §1's retraction was written to obtain —
the ratio's own optimum with **magnitude held constant**, so it cannot be confounded:

```
disappear fixed at 2.0
appear    0.13    0.25    0.40    0.67    1.0     1.4     2.0
score    0.8969  0.9007  0.9093  0.9090  0.9086  0.9071  0.9032
div_J    0.0000  0.0333  0.1154  0.1098  0.1098  0.1098  0.1084
forks        53     421   1,443   2,477   2,477   2,477   2,468
```

**Interior peak at appear 0.4 — a 1:5 ratio.** So the lab's 5× ratio *is* right, but only
at the correct magnitude, and only once magnitude is controlled for. That is a narrower and
better-supported claim than the one `notes/32` made and `notes/35` retracted.

Note `appear 0.13` produces 53 forks and `div_J` 0.0000 — the same dead division term as the
pack's defaults. Cheap appearance lets the solver start a new track instead of forking, so
**appearance cost is what converts a track-start into a division.**

## 3. The two knobs are substitutes, as predicted

```
ratio0.4_2.0    0.9093              1,443 forks   div_J 0.1154
div0.85+r5      0.9074   -0.0019    2,590 forks   div_J 0.1047
div0.7+r5       0.9072   -0.0020    2,725 forks   div_J 0.1023
```

Cheapening `division_weight` on top of a high termination penalty makes **more** forks and a
**worse** `div_J` — it is adding low-precision forks to a set the termination penalty had
already selected well. `division_weight` is now closed for the third time and from a new
direction: not "it does nothing," but "the termination penalty already did its job better."

## 4. Two of the three FAILs are my grading, not the data

Stated plainly because a reader scanning for PASS/FAIL would draw the wrong conclusion.

- **Prediction 2 ("each axis turns inside this grid") — FAIL is wrong.** The check was
  `best not in (first, last)`, which treats both ends alike. On axes 1 and 2 the peak is at
  the *smallest* magnitude in *this* grid — because the previous grid already bracketed it
  from below. "Peak at the bottom of this grid, top of the last one" is a turn; my test
  called it a boundary.
- **Prediction 3 ("the fn_detect tax is monotone in magnitude") — FAIL is wrong.** I sorted
  by `max(appear, disappear)`, which lumps all seven `dis=2.0` arms at magnitude 2 in
  arbitrary order and reads the jumble as non-monotone. Within each axis the tax *is*
  monotone: 370 → 760 → 1,261 → 1,742 → 2,647 → 3,560.

Both are fixed in the builder. The finding they were testing holds; the tests did not.

## 5. What this closes, and what is left

**Closed: the ILP's weights.** Three sweeps, 69 settings, a located interior optimum on the
one axis that can be measured cleanly, and catastrophic falloff on the other two. There is
no more score in this direction.

**The remaining gap is the edge term, and it is the whole gap.** At the best arm:

```
raw edge_J   0.9047        <- ours
div_J        0.1154        <- competitive with the field (notes/33: 0.12 reported)
public notebooks  0.923–0.927 total
```

The division term is no longer where we are losing. `notes/33` §1 identified the lever that
acts on edges and it is still untouched: **`biohub-temporal-unet3d-seed314159-v1`**, the
second missing model, a secondary linker whose edge logits are calibrated onto the primary's
scale and blended — plus **bidirectional harmonic linking** (`notes/33` §2), which needs no
new weights at all.

That is the next build, and unlike the last three it needs a real prediction pass rather
than the cached ILP instances.

Banked floor **0.752**. Best scored **0.883**; `claude_submit_ratio` in evaluation.
Bronze **0.926**, gold **0.944**.
