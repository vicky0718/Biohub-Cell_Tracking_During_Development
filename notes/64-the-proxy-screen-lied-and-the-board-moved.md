# The PROXY screen lied, and while we swept, the board moved

Two things happened at once. The first is a loss; the second changes the plan.

```
submitted 2026-09-04   claude_fork (nusrati/0-938, unmodified)   PROXY 0.9266   LB 0.937
submitted 2026-09-05   claude_forkw085 (w 0.80 -> 0.85)          PROXY 0.9283   LB 0.932
```

**PROXY moved +0.0017. The leaderboard moved −0.005, in the opposite direction.**

---

## 1. `notes/63` §3 is withdrawn; `SECONDARY_DETECTION_WEIGHT` 0.80 was the peak

I recommended 0.85 because the fork's own validator showed a clean interior maximum with
both neighbours falling away in order (0.85 +0.0017, 0.90 +0.0004, 0.95 −0.0025). That
shape is real on their 10 held-out training datasets and it is wrong on the test embryos.

The mistake was not the sweep. It was **filing the parameter on the wrong side of
`notes/60`'s rule**, which I had written six days earlier:

> *before checking a constant against geometry, ask whether score has ever been allowed to
> move it. If yes, expect the geometry to lose.*

I classified `SECONDARY_DETECTION_WEIGHT` as untuned because *their grid stopped at 0.80*.
But `nusrati/0-936`'s header says 0.80 is "our independently-swept dual-seed detection-fusion
peak (**+0.003 on our lineage**)" — a leaderboard number. 0.80 is a **score-tuned constant**,
tuned on the same metric and the same hidden test set we are graded on. `notes/60`'s rule
covers it exactly, and the rule was right again: a train-side screen lost to a
metric-tuned value. A sweep ending at a value is not evidence the value is a boundary; it is
usually evidence the author stopped where the score did.

It also says the parameter is **sharp**, not flat: 0.05 of weight is worth −0.005 of score.

## 2. PROXY_SCORE is not a submission gate

This is the second train-side screen to point the wrong way:

```
notes/49   topk p=0.0006 on 36 pooled crops    0.901 -> 0.863   (-0.038)
notes/64   PROXY +0.0017 on 10 held-out sets   0.937 -> 0.932   (-0.005)
```

Both had the same defect and it is structural, not fixable by more careful statistics.
`notes/49`: the 36 training datasets are crops of **two** embryos and the test set is a
**third pair**. `notes/24`: the pack's `split_0` membership is unknowable, so the fork's ten
"held-out" datasets may be in its own training data. A screen run on training embryos
measures fit to those two embryos, and the quantity we are paid for is transfer to a pair we
have never seen.

**Rule: PROXY may be reported, never used to decide.** With ~23 days left and 5 slots a day
we have ~115 submissions and roughly 140 GPU-hours. **Slots are not the scarce resource —
GPU time is.** So stop screening changes on a proxy that has now misled twice, and spend
slots directly on the only instrument that measures the thing we are scored on.

## 3. The board moved under us, and the public frontier is at 0.941

Full leaderboard download, 2026-09-06T01:43Z, **3,166 teams** (up from 3,038 on 2026-09-04):

```
                 2026-09-04     2026-09-06
top                  0.965          0.970
rank  50                            0.944
rank 100             0.940          0.942
rank 300                            0.940
our 0.937          rank ~330      rank 466
```

Our score did not fall; the field rose past it. **0.937 was rank ~330 two days ago and is
rank 466 today**, and rank 100 now costs **0.942**. Anything that takes days to build is
being priced against a target that is climbing about +0.001/day.

And the public notebooks moved with it. Sorting public kernels by score turns up a whole
band above where we forked:

```
votes  kernel                                                    claimed LB
   65  nusrati/0-940                                                  0.940
   76  analyticaobscura/biohub-lb-941                                 0.941
   26  rishabhr0y/941-biohub-fresh-adaptive-assoc                     0.941
    2  reyhanksatria/redistributing-the-repair-budget-0-941-lb        0.941
   27  qrz1201/biohub-v19c-sister14-repro                             0.939
```

**We forked the lineage two versions before its current head.** `claude_fork` reproduced
`nusrati/0-938` at 0.937 on 2026-09-04; `nusrati/0-940` already existed and its successors
were landing while we swept one of its constants.

## 4. What separates our 0.937 from their 0.941: four environment variables

Every notebook above is the same code with a different first cell. Diffing the `BIOHUB_*`
assignments against `claude_fork_source.json` (all keys not listed are identical):

```
key                                    ours     0-940    lb-941   rishabh   reyhank
BIOHUB_DET_THRESHOLD                   0.97     0.965     0.965     0.965     0.965
BIOHUB_SAFE_DIV_DIVERGE_UM              4.5      2.25      2.25         -      2.25
BIOHUB_GAP_CLOSE_UM                     5.8      5.8       5.0        5.8      5.0
BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD      -        -       0.25         -      0.25
BIOHUB_SECONDARY_LINK_MODE          low_marg  low_marg  low_marg  adaptive  low_marg
```

Two of them say so in their own metadata. `biohub-lb-941` carries
`BIOHUB_SCORE_AXIS = 'public 0.940 base + {"BIOHUB_DEEPCENTER_SAFE_DIV_THRESHOLD": "0.25",
"BIOHUB_GAP_CLOSE_UM": "5.0"}'`, and `reyhanksatria` publishes the whole progression as a
table — 0.934 → 0.939 → 0.941, with the last step being exactly *DeepCenter safe-div
threshold 0.12 → 0.25* and *gap-close 5.8 → 5.0*. **Two independent 0.941 notebooks, same
four values.** That is the strongest corroboration available without spending a run.

Note what `GAP_CLOSE_UM` 5.8 → 5.0 means for us: `notes/60` swept our own `close_gaps`
radius and found 5.75 optimal with every *widening* monotonically worse. It never tried
*narrower*. Their 0.941 step is a narrowing. Same direction of error, opposite side —
`notes/48`'s boundary-value lesson, and this time it is our grid that stopped short.

## 5. Plan

1. **`claude_fork` (0.937) stays selected.** It is our best submitted score and nothing
   pending beats it.
2. **Fork the frontier unmodified, two arms in parallel** (2 concurrent GPU sessions):
   `analyticaobscura/biohub-lb-941` (most votes at 0.941, config corroborated by
   `reyhanksatria`) and `rishabhr0y/941-biohub-fresh-adaptive-assoc` (a *different* route to
   0.941 via `SECONDARY_LINK_MODE=adaptive`). Unmodified is the point — `notes/61` reproduced
   at −0.001 and every modification since has lost.
3. **Only then look for +0.001.** Rank 100 costs 0.942 and a clean 0.941 is rank ~115, so
   the fork is a floor and not the finish. The honest place to look is where these four
   knobs disagree with each other, because that is where nobody has run the combination:
   `adaptive` link mode has never been paired with deepcenter 0.25 + gap 5.0.

```
0.752 floor   0.901 our chain   0.937 fork (rank 466)   0.941 = rank 115   0.942 = rank 100
```
