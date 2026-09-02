# ~~det_threshold 0.999: the largest measured effect this project has produced~~

> **❌ REFUTED by `notes/49`. `claude_submit_topk` scored 0.863 against the 0.901 incumbent
> — a −0.038 miss on a predicted +0.006 to +0.010.** The train numbers below are correct;
> the inference from them is not. `p = 0.0006` pools 36 crops of **two** embryos, and the
> test set shares no embryo_id with train (`notes/07` §3), so the honest sample is §2's two
> embryo means (+0.0028, +0.0085) — n=2, with a spread as large as the effect. Section 3's
> claim that 0.999 is "genuinely interior" is also wrong: its neighbour at 0.9999 costs
> −0.18, so it is a shoulder, and detector logits are not calibrated across embryos.
> **`DET_THRESHOLD` stays at 0.975.** Read this note as the record of a well-powered test of
> the wrong population.

`claude_topk` swept `det_threshold` past 0.99 for the first time. **0.999 beats the
submitted 0.975 on 28 of 36 datasets, sign test p = 0.0006.**

```
det           mean best-cell score   datasets > 0.9
0.975                       0.9410       26
0.999                       0.9474       28
0.9999                      0.7646       14
0.99999                     0.2991        3
0.999999                    0.0835        1
```

---

## 1. The plain t-test is wrong here, and three robust tests agree it is

```
n = 36        positive 28        negative 8
sign test                     p = 0.0006      SIGNIFICANT
mean               +0.0063    sd 0.0291   t  1.31    not resolved
median             +0.0065
10% trimmed mean   +0.0076    sd 0.0106   t  3.92    RESOLVED
minus worst point  +0.0099                t  2.95    RESOLVED
```

One dataset, `44b6_0c582fdc`, moves **−0.1189** — four times larger than any other
difference in either direction — and it alone holds the plain `t` to 1.31. Every estimator
that is not dominated by a single tail point resolves the effect.

`notes/42` set the rule that a threshold below the measurement's resolution is a coin flip
with a paper trail. The corollary shows up here: **a paired t-test is not the right
instrument when the differences are heavy-tailed.** Every earlier "not resolved" verdict in
this project used it, and none of them had a tail like this one — but this is the first
time it would have thrown away a real result.

## 2. It is concentrated where the leaderboard actually is

```
44b6   n=14   mean +0.0028   t 0.25    11/14 better
6bba   n=22   mean +0.0085   t 2.60    17/22 better
```

`harness/purescore.py`'s own docstring: *`adj_edge_jaccard` is a weighted mean of
per-dataset values, weight `TP + FP + FN` … that makes the two `6bba_` test datasets ~95%
of the leaderboard.* The effect is real on the embryo that carries almost all the weight,
and near-zero on the one that barely counts. That is the favourable arrangement, not the
suspicious one — but it also means the leaderboard delta will track the 6bba figure, not
the pooled +0.0063.

## 3. Why it was never tested

`notes/28` froze `DET_THRESHOLD` at 0.99 so that one leaderboard delta would isolate the
repair chain. `notes/40` unfroze it and swept 0.99 / 0.985 / 0.975 / 0.96875. `notes/41`
refined to 0.98 / 0.975 / 0.97 / 0.965. `notes/44` concluded the axis was flat and closed
it — correctly, **for the interval it had looked at**.

Every one of those sweeps stayed inside `[0.965, 0.99]`. **0.999 sits just past the edge of
the region anyone examined**, and `notes/44`'s "the config surface is flat" was a statement
about that box. The boundary was never probed because the frozen value had been at its
edge, so the sweeps grew *inward* from it.

Past 0.999 the detector falls off a cliff — 0.9999 is −0.176, 0.999999 is −0.858 — so the
useful region is narrow and 0.999 is genuinely interior to it, bracketed by 0.975 below and
0.9999 above.

## 4. The run also nearly threw itself away

The worker completed all 36 datasets in 7,771 s and the **analysis cell** then died:

```
KeyError: 'pool_grid'
```

I renamed the JSON key `pool_grid` → `det_grid` in the writer and left the reader alone.
Every number above was recovered from the per-dataset log lines instead. The reader now
accepts either name, so a rename cannot discard two hours of GPU again.

## 5. Submitted

`claude_submit_topk` is the chain that scored **0.901** with exactly one line changed —
`DET_THRESHOLD` 0.975 → 0.999 — verified by diffing the built notebooks: one functional
line, the rest comments. Gap 2, min-track 6, ILP 0.4/2.0 and the bidirectional blend are
untouched.

Expected: the train effect is +0.006 to +0.010, and `notes/40`'s transfer record is five
directions out of five correct in sign with a magnitude ratio between roughly 0.3 and 1.2.
So somewhere around **0.903–0.911**, most likely near the middle. That is not bronze. It is
the largest single change since the ILP weights, and the first thing since `notes/40` to
clear the 0.0015 measurement floor by a comfortable margin.

```
0.752 floor    0.901 submitted    0.926 bronze    0.944 gold
measurable > 0.0015     worth a slot > 0.01
```
