# Motion relink was destroying the divisions *and* costing edges — `norelinkdiv` is +0.0069

`claude-eval-norelinkdiv`: motion relink **off**, ILP division penalty 1.2 → 0.4. Twelve
held-out movies, official scorer.

```
arm            cands  added  forks |   edge_J      adj    tp/fp/fn    div_J     score
ttasec           975    194    194 |  0.93281  0.93623       2/2/9  0.15385  0.95161
divloose         644    155      - |  0.93281  0.93623       2/2/9  0.15385  0.95161
dcloose          975    245      - |  0.93262  0.93603       2/4/9  0.13333  0.94936
ilpdiv04         984    200      - |  0.93242  0.93577       2/3/9  0.14286  0.95006
ilpdiv00        1016    209      - |  0.93253  0.93571       2/3/9  0.14286  0.95000
norelinkdiv      670     70  1,376 |  0.94505  0.94870      6/50/5  0.09836  0.95854
```

**+0.0069 on score.** Five previous arms moved it by at most 0.0022, in the wrong direction.

## 1. The gain is on the edge term, and it is a count, not an aggregate

```
             edge_tp   edge_fp   edge_fn
ttasec          8997       324       324
norelinkdiv     9047       252       274
delta            +50       -72       -50
```

**All three counts move the right way**: 50 more true edges, 22% fewer false positives, 15%
fewer false negatives, out of 9,321 GT edges. For scale, the difference between `ttasec` and
`ttaz16` — the comparison that defeated this harness in `notes/77` — was **three** edges.
This is twenty times that.

That distinction matters, because `notes/77` closed offline *ranking* of 0.003-scale arms and
I am not reopening it. The claim here is not "0.95854 > 0.95161 therefore better". It is that
`edge_fp` fell by 72 and `edge_fn` by 50 against a shared baseline — a direct count on the
same 9,321 edges, of the kind `notes/77` §8 said this harness can still answer.

Per movie: **9 of 12 improved**, and of the three that did not, two moved by −0.001. The single
real decline is `44b6_0c582fdc` at −0.023, a 70-GT-edge movie. Paired mean +0.0172, t = +1.85
— which is the *weak* half of the evidence, and it is weak because it averages over movies
whose GT density varies 37-fold. The counts are the strong half.

## 2. The ILP knew about the divisions all along

```
             division_tp   division_fn   division_fp   forks
ttasec                 2             9             2     194
norelinkdiv            6             5            50   1,376
```

`division_tp` moves for the first time in six evals: **2 → 6 of 11**. `notes/79` predicted this
— the ILP's divisions were real and were being discarded wholesale by a one-to-one assignment.
With the relink off they survive, and four more GT divisions come with them.

**But `div_J` got worse**, 0.1538 → 0.0984, because the ILP emits **1,376** forks where 194
existed before and 50 of them land on annotated, still-continuing tracks. The division term
went from 0.0154 to 0.0098.

So the +0.0069 decomposes as **+0.0125 edge, −0.0056 division**. The headline gain is
*despite* the division term, not because of it.

## 3. Which makes the next arm obvious

Keep the relink off, keep the 6 recovered divisions, prune the 1,376 forks. The tool for that
is already in the notebook and is **dead code**:

```python
if OUTPUT_DIVISION_GEOMETRY_FILTER and edges:      # defaults to '0', never set
    ...
    valid_division = max(d1, d2) <= DIV_PARENT_MAX_UM and sister <= DIV_SISTER_MAX_UM and ...
    if valid_division:      filtered.extend([top1, top2])
    elif DIV_DROP_TO_SINGLE_IF_BAD:  filtered.append(top1)
```

It takes the top two edges per source and drops to a single child unless the geometry looks
like a division. With 194 forks it was irrelevant. With 1,376 it is exactly the right
instrument, and `notes/57` measured the distribution to set it against: parent→daughter median
7.13 / p90 10.05, sister median 10.57 / p90 14.36, against defaults of
`DIV_PARENT_MAX_UM` 10.5 (≈p92) and `DIV_SISTER_MAX_UM` **8.0** (≈p25 — far too tight).

Arithmetic worth having: at `tp` 6, `fn` 5, if the filter cut `fp` 50 → 5 then
`div_J = 6/16 = 0.375` and the division term becomes 0.0375 — **+0.028** on top of the edge gain.

## 4. Status and what this costs

`claude-eval-norelink` (relink off, penalty left at 1.2) is running. It decomposes the +50
edge TPs: how much is the relink itself and how much the extra ILP divisions.

This is the first result of the session worth a full arm run (~11 GPU-h) and a submission slot.
It is also the first that clears the noise floor by a wide margin. It should still go to the
board before anything is built on it, because this harness is contaminated (`notes/72` §3) and
its absolute level is not a leaderboard estimate — `notes/77` §2 is not repealed by one large
effect.
