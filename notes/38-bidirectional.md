# Bidirectional linking works — but it buys edges by giving up divisions

`claude_bidirectional`, 12 datasets (**5 × `44b6`, 7 × `6bba`** — stratified this time),
4 weights × 2 repair states, 2,926 s. The source patch landed: `anchor matches 1 x in 8678
chars`, and candidate counts move 2.1–2.6 % away from the control, so the reverse pass is
genuinely running.

```
w0.0+repair    0.9462   (control)
w0.15+repair   0.9499   +0.0036   <- best
w0.4+repair    0.9492   +0.0030
w0.25+repair   0.9485   +0.0022
```

Predictions 2, 3, 4 and 5 all passed. **Prediction 1 failed, and that failure is mine, not
the run's** — see §3.

---

## 1. The mechanism is not the one I predicted, and the anatomy says so plainly

I predicted the blend would cut `fn_mislink`, because mutual support should kill
wrong-parent links. It does — but barely. The large effect is somewhere else entirely:

```
                  mislink        gap        detect      candidates
w0.0+repair          86          33          158         264,336
w0.15+repair         83 (-3)     25 (-8)     124 (-34)   269,955  (+2.1%)
w0.25+repair         80 (-6)     23 (-10)    116 (-42)   270,887  (+2.5%)
w0.4+repair          79 (-7)     21 (-12)    117 (-41)   271,277  (+2.6%)
```

**`fn_detect` falls 5–6× more than `fn_mislink` does.** The blend admits ~2 % more
candidate edges, the ILP keeps more of them (239,432 → 246,255), and GT nodes that were
previously stranded on a dead track end now sit on a live one and match. The gain is
*retention*, not *correction*.

That is worth stating loudly because it inverts the reasoning that motivated the build.
`notes/26` named `fn_mislink` as the target and `pipeline/bidirectional.py`'s own docstring
argues for the harmonic mean on those grounds. The argument was right about the mean being
the correct combiner and wrong about which bucket it would empty.

## 2. 🚨 The blend suppresses divisions, and that is what caps it

```
              div_J     0.1 x div_J     adj_edge      total
w0.0+repair   0.0690      0.0069         0.9393      0.9462
w0.15+repair  0.0571      0.0057         0.9442      0.9499
w0.25+repair  0.0270      0.0027         0.9458      0.9485
w0.4+repair   0.0263      0.0026         0.9466      0.9492
```

**The edge term improves monotonically with the weight — +0.0049, +0.0065, +0.0073 — while
`division_jaccard` collapses from 0.069 to 0.026.** w=0.15 wins not because it is the best
linker but because it is the point where the edge gain still outruns the division loss.

The mechanism is coherent. `notes/35` established that divisions here come from the ILP's
*termination penalty*: a high disappearance cost makes the solver fork rather than end a
track. The blend admits more candidate edges, which gives the solver cheaper ways to
continue a track without forking, so it forks less. **The two levers are pulling against
each other**, and the current settings were tuned with one of them absent.

This is a live, cheap follow-up rather than a dead end: the ILP weights were located
(`notes/36`) on *unblended* candidates. Re-locating them with the blend active could
plausibly keep the +0.0073 edge gain at w=0.4 *and* restore the division term — the two
together would be worth ~+0.011 rather than +0.0036.

## 3. Prediction 1's failure is a miscalibrated prediction, not a bad control

```
1. w=0 reproduces notes/35's ratio0.4_2.0+repair (0.9179 +- 0.002)
   w0.0+repair = 0.9462  ->  FAIL
```

The grading cell then blamed the cache. **It is wrong, and so was the prediction.**
`notes/35`'s 0.9179 was pooled over **24** datasets; this run uses **12 of them**. A
different subset gives a different pooled score, and the per-dataset spread here is
0.777–1.052 — entirely consistent with `notes/33`'s report of ±0.14 movie-to-movie
variance. Comparing a 12-dataset pool to a 24-dataset one was never a valid test.

What that costs: the cross-run comparison to `notes/35` is unavailable, so I cannot confirm
from this run that a fresh prediction pass reproduces the cached candidates. What it does
**not** cost: the arm-to-arm comparison, which is what the run exists for. All four arms
went through the same fresh pipeline, and the control uses the *original unpatched function
object*, so it is the unblended pipeline by construction.

The fix for next time is to grade the control against a number measured on the *same*
datasets, or to run all 24. This is the third measurement-design error in five runs
(`notes/34` had two, `notes/35` §1 had one) and they share a shape: **a control compared
against something it is not actually comparable to.**

## 4. Next

1. **`claude_bidir_ilp`** — predict once per dataset with the blend at w=0.4 (the best
   *edge* term), cache the candidates, then sweep the ILP's appearance/disappearance on
   that cache exactly as `notes/31`–`36` did. One prediction pass, many solves. §2 says the
   division term should be recoverable at the new operating point, and if it is the total
   is ~+0.011 rather than +0.0036.
2. **Only then decide on a submission.** +0.0036 alone is above the 0.001 noise floor but
   translates to +0.002 to +0.004 at the observed transfer range — real, and not worth a
   slot on its own while a 3× larger version of the same thing is one cheap run away.
3. Still untouched: the **temporal linker blend** (`notes/33` §1), the second missing model.
   This run used only the model we already had.

Banked floor **0.752**. Best scored **0.897** (rank ~1336/2792).
Bronze **0.926**, gold **0.944**.
