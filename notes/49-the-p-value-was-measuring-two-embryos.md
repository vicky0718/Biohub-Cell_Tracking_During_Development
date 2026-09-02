# 0.901 → 0.863: the p-value was measuring two embryos, not thirty-six datasets

`claude_submit_topk` scored **0.863**. Predicted 0.903–0.911. First transfer failure in six,
and it missed by −0.038 against a forecast of +0.006 to +0.010.

The train measurement was not wrong. **The inference from it was**, and the error is one
this project diagnosed at the harness level in `notes/07` and then reintroduced in the
grading cell.

---

## 1. n was never 36

`notes/48` reported `sign test p = 0.0006` over 36 datasets. The host, quoted in
`notes/07` §3:

> *"there are two unique embryo_ids in the training set … you can assume the test set is
> roughly similar in size, with **no overlap in embryo_ids** between train and test sets"*

So the 36 datasets are 36 crops of **two embryos**, and the leaderboard is **two embryos
never seen**. A test that pools crops answers *"does 0.999 beat 0.975 on these two
embryos"*. That is not the question the leaderboard asks.

`notes/48` §2 already contained the honest sample and read it as reassurance:

```
44b6   n=14   mean +0.0028   t 0.25    11/14 better
6bba   n=22   mean +0.0085   t 2.60    17/22 better
```

**Two numbers. n = 2.** The between-embryo spread (0.0057) is as large as the effect
(0.0063). Written that way there was never a resolvable result — and I read the table as
"concentrated where the leaderboard is" rather than as the whole sample.

`Harness` has defaulted to `fold_by="embryo"` since `notes/07` for exactly this reason. The
topk grading cell computed its own paired statistics over `PER` — every crop, pooled — and
bypassed it:

```python
def paired(a, b):
    d = []
    for nm, r in PER.items():        # 36 crops, 2 embryos, treated as 36 draws
        ...
    n = len(d)                       # 36
```

## 2. Why the miss was −0.038 and not −0.006

Pseudoreplication explains why the effect was unproven. It does not by itself explain a
loss eight times the predicted gain. The second mechanism is **curvature**, and it is
visible in `notes/48`'s own table:

```
det        train score      step
0.975         0.9410
0.999         0.9474      +0.0064
0.9999        0.7646      -0.1828      <- the next grid point
```

0.999 is not interior to anything. It sits on the **shoulder of a cliff**, one order of
magnitude in `(1 − p)` from a −0.18 collapse, while 0.975 sits mid-plateau — `notes/44`
measured the entire 0.965–0.99 range moving the score by **0.0001**.

A detector's logits are not calibrated across embryos (`notes/04` §9: the two scored
datasets differ **11× in cells per frame**). On a plateau that does not matter. On a
shoulder, a modest calibration shift on unseen embryos slides the effective operating point
up the cliff, and the move is one-directional: there is nothing to gain above 0.999 and
0.18 to lose.

The budget term cannot be the culprit and that is worth stating, because it was the
motivation for the whole direction. `J_adj = J·(1 − 0.1·ratio)` is **capped at 1.1**, and
raising the threshold lowers `N_pred`, so it drives `ratio` negative — toward the cap, never
toward the `max(0, ·)` floor. Under-prediction cannot lose points through the multiplier.
**The −0.038 is pure edge loss from deleting true nodes**, which is precisely what the
0.9999 arm does catastrophically on train.

## 3. What this invalidates beyond one value

Every paired test since `notes/40` pooled crops across two embryos. `notes/42` caught the
*power* problem and `notes/44` caught the *flatness*, but both quoted an `n` inflated by
pseudoreplication — n=60 in `notes/44` is still two embryos.

The five-for-five transfer record survives, with a caveat that now looks load-bearing: the
things that transferred (repair chain +0.0115, ILP weights +0.0221, bidirectional +0.0036)
were **structural** — changes to how the graph is built, which behave the same way on any
embryo. This is the first *calibration* parameter pushed off its plateau, and it is the
first to fail. That is a distinction the record did not previously have to make.

## 4. The rule

1. **Grade config by leave-one-embryo-out.** Two folds, as `Harness` already defaults.
   Report both embryo-level means. If they disagree in sign or differ by more than the
   effect, nothing is established for a third embryo — whatever the pooled p-value says.
2. **Prefer plateau interiors to shoulder optima.** When a grid point's neighbour costs
   −0.18, its measured +0.006 is not worth a slot. Quote the local gradient next to every
   proposed config change.
3. **A pooled p-value across crops is not evidence about a new embryo.** `notes/42` said
   pre-registering below the resolution limit is a coin flip with a paper trail; this adds
   that a well-powered test of the wrong population is worse, because it produces
   confidence instead of ambiguity.

## 5. Standing

**Keep 0.901 selected.** Nothing in the repo needs reverting — `claude_submit_topk` is a
separate notebook and the 0.901 chain is untouched at `DET_THRESHOLD = 0.975`.

The threshold axis is now closed for a second and better reason than `notes/44` gave: not
merely flat, but flat with a cliff at one end, and the plateau value is the robust choice
rather than an arbitrary one.

```
0.752 floor    0.901 best    0.863 topk (rejected)    0.926 bronze    0.944 gold
config axis: closed        remaining headroom: structural, not calibration
```
