# Experiment #6 — the embryo split breaks, and most of the win turns out to be unbuildable

`07_adaptive.ipynb`, run on Kaggle 2026-08-19, 11,128 s (3.1 h) on the fixed 60-dataset
subset. Leave-one-embryo-out folds verified in-run: `{0: 21 = 44b6, 1: 39 = 6bba}`.
Raw output in `adaptive_results.json`.

**Reproduction check passed exactly**: `champion_04` 0.6760 against `04`'s 0.6760, drift
−0.0000.

> **The whole notebook was then re-run in a fresh Kaggle session (2026-08-21, 11,110 s)
> and every arm came back bit-identical to four decimals** — 0.6760 / 0.7072 / 0.6849 /
> 0.7048 / 0.7098 / 0.6962 / 0.7115 — with the same fold deltas (+0.0449 / +0.0320 for
> `adaptive_1.2`, −0.0417 / +0.0097 for `adaptive_wide`). Per-arm wall times moved by up
> to 40 %, so the machine differed; the numbers did not. The pipeline is deterministic
> across sessions, which means CV differences of the size we have been gating on are real
> signal and not run-to-run noise.

| arm | SCORE | micro edge_J | budget mult | recall | nodes | ratio | gate vs `champion_04` |
|---|---|---|---|---|---|---|---|
| adaptive_wide | **0.7115** | 0.7075 | 1.0056 | 0.832 | 996,972 | −0.118 | PROMOTE +0.0356 |
| **adaptive_1.2** | **0.7098** | **0.7169** | 0.9901 | 0.867 | 1,215,318 | +0.049 | **PROMOTE +0.0338** |
| prune_plus_cap | 0.7072 | 0.7013 | 1.0084 | 0.870 | 1,126,596 | −0.098 | PROMOTE +0.0312 |
| adaptive_1.0 | 0.7048 | 0.7035 | 1.0019 | 0.839 | 1,112,166 | −0.063 | PROMOTE +0.0289 |
| adaptive_noprune | 0.6962 | 0.7111 | 0.9790 | 0.879 | 1,284,257 | +0.145 | PROMOTE +0.0202 |
| adaptive_0.8 | 0.6849 | 0.6730 | 1.0177 | 0.789 | 956,259 | −0.211 | reject |
| champion_04 | 0.6760 | 0.7053 | 0.9584 | 0.857 | 1,160,180 | +0.239 | — |

All three pre-registered predictions **CONFIRMED**. And then §3 undercuts most of it.

---

## 1. ⭐ Prediction 2 — the embryo split broke

`adaptive_1.2` over `champion_04`: **PROMOTE +0.0338**, fold deltas **`44b6` +0.0449 /
`6bba` +0.0320**. `embryos still split: False`.

That is the first time in four experiments that a node-count change has moved both embryos
the same way. `05`, `06` × 2 and every fold table in `notes/12` §3 said the two embryos want
opposite densities and no global setting serves both. They still do — but **that
disagreement was about a *global* setting, and there is no longer a global setting.** Each
dataset picks its own `min_separation_um` to land on its own budget, so there is nothing
left for the embryos to disagree about. The obstacle was never the embryos; it was the
single knob.

This was the condition I set for continuing the classical line at all (`notes/12` §5:
stop tuning if `07` does not break the split). It was met.

Prediction 1 **CONFIRMED** too: `prune_plus_cap` gates at +0.0312, folds +0.0256 / +0.0323 —
so `06`'s ungated 0.7072 headline was legitimate after all, just unproven at the time.

Prediction 3 **CONFIRMED**: pruning adds +0.0136 to the adaptive arm and lifts micro edge
Jaccard 0.7111 → 0.7169. Three for three; pruning stays on everywhere.

## 2. The notebook named the wrong champion

Final line: *"new gated champion: adaptive_wide"*. That is wrong, and the notebook had
already produced the evidence against it two cells earlier.

`adaptive_wide` was gated against `adaptive_1.2` and **REJECTED**:

| | pooled | `44b6` | `6bba` |
|---|---|---|---|
| adaptive_wide − adaptive_1.2 | **+0.0017** | **−0.0417** | +0.0097 |

It buys +0.0017 pooled — inside noise — for a 0.042 loss on `44b6`. The summary cell then
declared it champion anyway, because `promoted` is computed as *"passes the gate versus
`champion_04`"* and `adaptive_wide` has the highest score among those. That is a **weaker
test than the incremental gate it had just failed**: measuring every arm against a
two-experiments-old baseline lets an arm win on ground it already lost.

**`adaptive_1.2` is the champion**: 0.7098, folds +0.0449 / +0.0320, both solidly positive.
`adaptive_wide` is the score-maximising but fold-fragile one. Fix for `08`: chain the gate
against the *current* champion, not a fixed constant.

## 3. 🚨 Two thirds of the gain is the node-budget term — which we cannot read at test time

Because `n_adj == n` and no dataset is dropped, `adj_edge_jaccard` reduces to the micro
edge Jaccard times the weighted budget multiplier. So the arms decompose cleanly:

| `adaptive_1.2` − `champion_04` | |
|---|---|
| edge quality (micro edge_J 0.7053 → 0.7169) | **+0.0111** |
| node-budget multiplier (0.9584 → 0.9901) | **+0.0227** |
| total | +0.0338 |

The multiplier is two-sided (`notes/02` §2): over budget is penalised, **under budget is
rewarded**, up to ×1.1. `champion_04` runs +23.9 % over budget and pays ×0.958;
`adaptive_1.2` lands +4.9 % over and pays only ×0.990. That recovered penalty is 67 % of
the headline. The tracking itself improved by +0.0111.

And now the problem. **`test/` ships images only — 4 `.zarr`, zero `.geff`** (`notes/05`
§0). `estimated_number_of_nodes` lives in GEFF metadata, so at submission time it is
**unreadable**. The scorer still applies the multiplier using the hidden ground truth
(`discussions/01-scouting-report.md` §, confirmed against the host's own scoring code) —
we are graded on it, we just cannot see the target.

Both mechanisms behind every arm above read that number:

- `budget_fill` → cap per frame = `budget_fill * N_total / T`
- `adaptive_separation` → target per frame = `adaptive_target * N_total / T`

With no budget, `predict_dataset` prints `!! no budget for this dataset` and **falls back to
the fixed `min_separation_um`** — i.e. to `champion_04`. So as it stands:

> **`adaptive_1.2` scores 0.7098 on CV and 0.6760 on the leaderboard**, because on the test
> set it silently degrades into the arm it beat.

Nothing since `04` is submittable except pruning, which needs no metadata at all. That is
the real state of the project, and it was hidden by scoring every arm on train, where the
`.geff` files happen to be sitting next to the images.

*(This is not a scoring bug and not a leak — the host has confirmed scores above 1.0 are
expected precisely because sparse annotation puts most honest predictions under budget.)*

## 4. The wide-scale axis is still embryo-split — four for four

`adaptive_wide` vs `adaptive_1.2` splits `44b6` −0.0417 / `6bba` +0.0097. Same sign pattern
as `06`'s wide-vs-default (`44b6` −0.0597 / `6bba` +0.0468). Larger DoG sigmas suit the
sparse embryo and hurt the dense one, and **adaptive separation does not fix it** —
separation controls *how far apart* peaks must be, sigma controls *what size of blob is a
peak at all*, and only the first is being adapted. Note `adaptive_wide` undershoots badly
(996,972 nodes, ratio −0.118) despite targeting the same 1.2×: at wide sigmas the detector
cannot emit enough peaks even at minimum separation, so the calibration loop bottoms out.

If sigma is worth adapting, it is the same trick applied one axis over. Not yet — §3 first.

---

## What to do next

1. **⭐ Predict the node budget from the image.** This is now the highest-value experiment
   in the project, because it converts a CV-only +0.0338 into a real one. 199 train
   datasets with known budgets and images is a supervised problem with a trivial feature:
   count DoG peaks on a few frames at a fixed reference separation, regress
   `N_total` on that count. Validate **leave-one-embryo-out** — that is exactly the
   generalisation being asked for, since the hidden test is two different embryos.
2. **Measure the best *submittable* configuration.** We have never scored
   `champion_04 + prune` — DoG sep 6.0, no cap, pruning on. Every arm we have either needs
   the budget or predates pruning. It is one arm and it is the number we would submit today.
3. **Chain the gate.** `08` gates against `adaptive_1.2` (0.7098), not `04`'s constant.
4. **The submission notebook still does not exist.** Notebook-only competition, internet
   off (so `pip install` is out — `harness/purescore.py` is why anything runs), `test/`
   swapped at rerun so names must be globbed, `id` column required, 12 h cap. A competitor
   also reports **scoring timeouts on graphs of ~62k nodes with fragmented connectivity**
   (`discussions/threads/724917-*`), so submission is its own engineering risk and needs
   lead time. It should not be left to the last week.
5. Standing: still **no leaderboard score**. Best gated CV 0.7098, best gated *submittable*
   CV 0.6760 pending item 2. Public evidence puts the tuned classical ceiling near 0.85
   against a leaderboard median of 0.890.
