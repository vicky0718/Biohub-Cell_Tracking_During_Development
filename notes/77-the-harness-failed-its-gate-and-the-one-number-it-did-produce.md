# The harness failed its acceptance gate — and the one number it did produce

`claude-score` ran. Scorer wiring verified by a self-test that controls its own inputs
(4/4 cases). polars installed correctly once `polars-runtime-32` was named. Official metric,
official formula, real ground truth, 12 held-out training movies.

```
FINAL ttasec    score=0.95161   adj_edge=0.93623   div_J=0.15385
FINAL ttaz16    score=0.95590   adj_edge=0.93715   div_J=0.18750
```

Leaderboard: **ttasec 0.945, ttaz16 0.942.** The harness puts them the wrong way round.

## 1. The gate, condition by condition

| # | condition | result |
|---|---|---|
| 1 | ranks `ttasec` above `ttaz16` | **FAILED** — ranks `ttaz16` higher by +0.0043 |
| 2 | `ttasec` lands near 0.945 | passed — 0.9516, +0.0066 optimism, consistent with `notes/72` §3 |
| 3 | ~5–15 GT divisions over 12 movies | passed — **11** (`tp` 2 + `fn` 9), against 151/199×12 ≈ 9.1 |

Condition 1 was declared blocking before the run. It failed. **No arm is ranked on this
harness**, and the plan's step 3 does not begin on its authority.

## 2. Why it failed — it is underpowered, not miswired

The per-movie paired difference, `ttaz16 − ttasec` on `adj_edge_jaccard`:

```
n = 12 movies
paired mean   +0.00282
per-movie sd   0.01332
standard error 0.00385
t             +0.73
```

**t = 0.73 is not a verdict, it is silence.** The harness did not conclude that `ttaz16` is
better; it concluded nothing, and noise fell on the wrong side. Leave-one-out confirms the
whole result is one movie: drop `44b6_0c582fdc` (where `ttaz16` leads by 0.0378, against
±0.013 everywhere else) and the sign flips. Eleven of twelve leave-one-out fits still favour
`ttaz16`, so this is not a single outlier being removed to taste — it is that the aggregate
never had enough signal to survive one movie either way.

To resolve the true 0.003 leaderboard gap at a standard error of half the gap:

```
n = (sd / 0.0015)^2 = 79 movies
```

**79 movies is ~3.4 GPU-hours per arm**, against ~8.5 for a full submission run. So an offline
comparison strong enough to rank two arms 0.003 apart costs the same order as simply
submitting them — and does not yield a leaderboard number. For edge-level differences of the
size this project produces, **offline ranking is closed on economics, not on correctness.**

That is the third instrument to fail at ranking 0.003-scale arms (`purescore` on 4 clips,
0-for-4; the `ttaz16` proxy call; now this). The common cause is not the scorers. It is that
our arms differ by about a quarter of the between-movie standard deviation.

## 3. What it *did* measure, and this part is solid

```
ttasec, 12 movies:   division_tp 2    division_fn 9    division_fp 2    div_J 0.1538
```

**We recover 2 of 11 ground-truth divisions. We miss 82% of them.** The division term
contributes `0.1 × 0.154 = 0.0154` to our 0.945.

This is a different kind of claim from §2 and the evidence is different in kind. §2 needed to
resolve a difference between two noisy aggregates. This is a **direct count**: 2 matched, 9
missed. Even ±3 divisions leaves "we miss most of them" intact. Condition 3 passing — 11 GT
divisions where `notes/57` predicts 9.1 — is what licenses reading it.

And it lands on the branch pre-registered in `notes/76` **before the run**:

> | `div_J` 0.10–0.25, `fn` > `fp` | recall-limited: real divisions are being vetoed | loosen the DeepCenter veto |

`fn` 9 against `fp` 2. Recall-limited, by a factor of 4.5.

Corroboration from the board, which is not this harness: `dc40` tightened that veto
(0.25 → 0.40) and scored **0.933**; `div15` tightened the divergence gate and scored **0.938**.
Both moved divisions *down* and both lost far more than edge arithmetic can explain.

## 4. The power calculation that decides what to do next

The harness is underpowered for the effects that defeated it. It is **not** underpowered for
the effect now in view:

```
effect that defeated it   0.003   (ttasec vs ttaz16)      t = 0.73
division headroom         0.020   (div_J 0.154 -> ~0.40)  t ~ 5
```

Same instrument, same n, effect seven times larger. A division arm that materially moves
`div_J` would be visible at n = 12; `ttasec` vs `ttaz16` never could be.

**This does not un-fail the gate.** The gate was about ranking arms, and arms stay unranked
here. It says which question this instrument may still be asked: *did the division count
move, and in which direction* — a mechanism check on direct counts — not *which arm scores
higher*.

## 5. What follows

* `claude-eval-dcloose` and `claude-eval-divloose` are running (12 movies each). They are read
  for **division funnel and tp/fn counts only** — is the knob live, does it admit more real
  divisions — never for a score ranking.
* The leaderboard is the only instrument that resolves 0.003, and submissions are **not** the
  scarce resource: 28 of ~115 used, 16 days left. GPU is (~70 h, ~8 full arm runs).
* So a division arm that the funnel shows is live goes to the **board**, not to a bigger
  offline run. Paying 3.4 GPU-hours for a weak offline t-test, when 8.5 buys a real number,
  is the trade this note exists to refuse.

## 6. Why divisions are cheap to emit — the metric says so in its own docstrings

194 predicted forks over 12 movies produced **`fp` = 2**. That is a 1% charge rate, and it is
not luck. `division_metrics.count_matched_pred_divisions`:

> "A matched GT node with no children marks the end of the annotation — we can't tell whether
> the cell actually divided there, so such predicted divisions are **excluded from the count
> (and therefore from the FP tally)**."

A predicted division is charged only when its parent matches a GT node **that still has a
child**. 190 of our 194 forks sit on GT-terminal or unmatched nodes and are invisible.

The edge term is gated the same way. `metrics._evaluate_matched_graph` builds `pred_valid`:

```python
edge_attrs.with_columns((pl.col("out_valid") | pl.col("in_valid")).alias("pred_valid"))
```

where `out_valid` means the edge's source matched a GT node with `out_degree > 0` and
`in_valid` that its target matched one with `in_degree > 0`. `_compute_score` then uses
`n_valid_pred_edges`, **not** the predicted edge count. A predicted edge whose endpoints fall
outside the annotated region is neither TP nor FP — it does not exist to the metric.

So the asymmetry that governs this whole direction:

```
missing a real division    costs a division FN, permanently  (we have 9)
emitting a wrong division  costs ~0.01 division FP on average (we have 2 from 194)
```

**Emitting divisions is close to free; missing them is not.** That is the mechanism behind the
board results — `dc40` and `div15` both cut divisions and lost 0.008 and 0.003 — and it is why
`fn` 9 / `fp` 2 points one way only.

One caveat kept honest: the extra edge each fork adds is *not* uniformly free. A daughter that
matches a GT node with an incoming edge makes that edge `in_valid`, so it is charged. The 1%
figure is the measured division-FP rate, not a proof that the edge cost is zero — which is
exactly what `claude-eval-dcloose` measures, since it reports both terms.

## 7. A second consequence, noted and not yet acted on

`adj_edge_jaccard = max(0, J × (1 − 0.1 × ratio))` has **no upper clamp**, and `ratio` is
signed. `44b6_66f9292d` came back at `adj = 1.0390` with `ratio = −0.5482`: under-predicting
node count by 55% multiplied its edge Jaccard by 1.055. Our ratios are negative on 10 of 12
movies, so we are already collecting a small bonus.

This is a real property of the shipped metric, not an artifact of this harness — the formula
is quoted verbatim from `metrics.per_sample_metrics`. It is recorded here rather than pursued:
raising the detection threshold to chase the bonus also drops edge TPs, and `det955`/`det960`
already probed that axis and landed at 0.941. Worth revisiting only with the division work
settled.

## 8. Correction to §4 — the training ground truth is 5% dense, and §4 overstated the power

`score_summary.json` carries the counts the log did not print, and they change the diagnosis:

```
ttasec, 12 movies:   edge tp/fp/fn  8997 / 324 / 324      GT edges  9,321
                     division       2 / 2 / 9             GT divisions  11
                     predicted nodes 196,723
ttaz16, 12 movies:   edge tp/fp/fn  9000 / 320 / 321      GT edges  9,321
                     division       3 / 5 / 8
```

**The training GT annotates ~4.7% of a movie** — 9,321 edges against 196,723 predicted nodes —
and its density varies 37-fold between movies:

```
44b6_0113de3b     50 GT edges     25,622 predicted nodes     0 GT divisions
6bba_09961292  1,871 GT edges     29,754 predicted nodes     4 GT divisions
```

So this harness was never scoring dense movies against dense truth. It is the same sparse
placeholder annotation `notes/66` found on the four visible clips — it is simply *more* of it.
That is why `harness/purescore.py` and this both fail at the same task.

### What actually decided the ranking

```
edge term      ttasec 8997 TP   ttaz16 9000 TP    difference: 3 edges in 9,321
division term  ttasec 2 TP      ttaz16 3 TP       difference: 1 event in 11
```

`0.1 × (3/16 − 2/13) = 0.0034` of the 0.0043 gap. **The harness ranked `ttaz16` above `ttasec`
because of one division event out of eleven.** The edge term — the thing that genuinely
separates these arms on the board — came out +0.0007 for `ttaz16` against a true −0.003, i.e.
three edges out of nine thousand. Both terms are noise; the division term is just noisier per
event, and carries a 0.1 multiplier.

### §4 was wrong

`notes/77` §4 argued the harness is "adequately powered for the effect now in view" because
division headroom is ~0.020 against the 0.003 that defeated it. That reasoning used the
between-movie sd from §2 and ignored that `div_J` rests on **11 events**. One division is
±0.03 of `div_J` and ±0.003 of `score` — the same size as the gap that defeated it. The
harness is **not** powered to rank arms on `score`, division arms included. §4 stands only as
the observation that division effects are larger, not as a licence to rank.

### What it can still do, stated tightly

Loosening a gate is **nested**: every candidate `ttasec` admits, `dcloose` also admits. So

* **"how many of the 11 does it recover"** is monotone under loosening and directly observed —
  a count, with no aggregation and no pairing noise. If `dcloose` recovers 2, the knob is inert.
* **"what does it cost"** — added division FPs, added charged edges — is also a direct count.

Both are mechanism questions. Neither is a score. **No arm gets ranked here**, which is where
the gate left things and where they stay; the board decides, and 28 of ~115 submissions are
used with 16 days left.
