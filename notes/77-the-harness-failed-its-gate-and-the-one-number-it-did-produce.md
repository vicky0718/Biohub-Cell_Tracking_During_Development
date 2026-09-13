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
