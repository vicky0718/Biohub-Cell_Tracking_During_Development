# Experiment #7 — the budget is legible in the pixels, and the CV number survives

`08_budget.ipynb`, run on Kaggle 2026-08-20, 10,241 s (2.8 h) on the fixed 60-dataset
subset. Folds verified `{0: 21 = 44b6, 1: 39 = 6bba}`. Raw output in `budget_results.json`.

Both reproductions exact: `champion_04` 0.6760 (drift −0.0000), `adaptive_oracle` 0.7098
(drift +0.0000).

| arm | SCORE | micro edge_J | budget mult | recall | nodes | ratio | gate |
|---|---|---|---|---|---|---|---|
| adaptive_oracle | 0.7098 | 0.7169 | 0.9901 | 0.867 | 1,215,318 | +0.049 | *reads the budget — not submittable* |
| **adaptive_predicted** | **0.7070** | 0.7128 | 0.9919 | 0.866 | 1,205,332 | +0.040 | **PROMOTE** +0.0174 over champion+prune |
| champion_plus_prune | 0.6896 | 0.7098 | 0.9715 | 0.844 | 1,087,402 | +0.132 | **PROMOTE** +0.0136 over champion_04 |
| champion_04 | 0.6760 | 0.7053 | 0.9585 | 0.857 | 1,160,180 | +0.239 | chain start |
| adaptive_null | 0.6014 | 0.6515 | 0.9231 | 0.738 | 859,704 | +0.291 | *reference — a constant budget* |

**New submittable champion: `adaptive_predicted`, 0.7070.** All three predictions confirmed.
`notes/13` §3 said the 0.7098 would collapse to 0.6760 at submission time. It does not —
**0.7070 of it survives**, and the chain is clean: 0.6760 → 0.6896 → 0.7070, each step
gated against the standing champion with no fold regression.

---

## 1. Prediction 2 — the budget is not just predictable, it is easy

Leave-one-embryo-out: fit on one embryo, predict the other. Never on itself.

| | median \|rel err\| | mean | `44b6` | `6bba` |
|---|---|---|---|---|
| **regression** | **10.7 %** | 12.1 % | 10.4 % | 11.0 % |
| null (training embryo's median) | 86.6 % | 172.5 % | 69.8 % | 231.2 % |

Ten per cent, and **symmetric across the embryos** — 10.4 vs 11.0. That is the property
that matters: the model is not learning "this is the dense embryo", it is reading density
off the image. Which is exactly what the hidden set requires, since it is two embryos this
model will never have seen.

Feature correlations with `log(budget/frame)`:

| feature | pooled | `44b6` | `6bba` |
|---|---|---|---|
| **nstrong_sep4** | **+0.987** | +0.98 | +0.98 |
| nstrong_sep8 | +0.985 | +0.98 | +0.98 |
| n_sep8 | +0.930 | +0.96 | +0.90 |
| n_sep4 | +0.911 | +0.96 | +0.87 |
| frac_fg | +0.900 | +0.82 | +0.89 |
| mean_int | +0.876 | +0.77 | +0.87 |
| nstrong_sep16 | +0.816 | +0.92 | +0.75 |
| n_sep16 | +0.624 | +0.80 | +0.49 |

**A correction.** `notes/13` and the `budget_features` docstring both warn that the raw
peak count is *anti*-correlated with density — measured on synthetic movies of 12/8/6/3
nuclei, where it ran 20/36/50/104. **On real data it is +0.91.** The anti-correlation is
an artefact of volumes far emptier than anything in this corpus: with three nuclei in a
128³ box the percentile floor sits in pure background and noise peaks fill in. Real
datasets never get that sparse. The synthetic finding did not transfer, and the warning as
written is too strong.

What *did* transfer is the fix it motivated. `nstrong_*` — the same peaks, filtered by an
absolute intensity cut instead of a percentile — is the best feature in the table by a
clear margin, and the only one that is equally strong on both embryos. So the test caught a
real weakness in the percentile floor even though it exaggerated its size. The docstring
and the test comment have been amended to say both halves.

## 2. Prediction 3 — a 10 % budget error costs 0.0028 of score

| | score | vs champion+prune |
|---|---|---|
| oracle budget (the true `estimated_number_of_nodes`) | 0.7098 | +0.0202 |
| **predicted budget** | **0.7070** | **+0.0174** |
| null budget (a constant) | 0.6014 | **−0.0882** |

**Retention 86 %.** The gap between a perfect budget and a predicted one is **0.0028** —
the calibration loop only uses the budget to choose a separation, and separation is a blunt
enough knob to absorb a 10 % error.

The null is the load-bearing result here. Applying one constant budget to every dataset
scores **0.0882 below doing nothing at all**, and 0.1057 below the regression. Per-dataset
density is not a refinement; getting it wrong is actively destructive. That also retires
any temptation to hardcode a global budget in the submission.

## 3. Where the score is actually coming from

`adj_edge_jaccard` = micro edge Jaccard × the weighted node-budget multiplier, so:

| step | total | edge quality | budget multiplier |
|---|---|---|---|
| champion_04 → +prune | +0.0136 | +0.0043 (32 %) | +0.0093 (68 %) |
| +prune → adaptive_predicted | +0.0174 | +0.0029 (17 %) | +0.0145 (83 %) |
| **champion_04 → adaptive_predicted** | **+0.0310** | **+0.0072 (23 %)** | **+0.0238 (77 %)** |

Three quarters of everything gained since `04` is the budget term, and that term is now
nearly exhausted: the multiplier stands at **0.9919** against a ceiling of 1.0 at exact
budget, worth **+0.0058** more at the current edge Jaccard. (Going *under* budget pays a
bonus up to ×1.1, but `07`'s `adaptive_0.8` and `adaptive_wide` both showed the quality
loss arriving faster than the bonus.)

**So the density axis is closed.** Everything from here has to come from edge quality,
which has moved 0.7053 → 0.7128 across three experiments — +0.0075 in total.

## 4. Two more things this run settles

**Runtime is not a risk.** 30–46 s per dataset for the full pipeline, 12 s for features.
A ~200-dataset hidden test is 2.5–3.5 h against a 12 h cap. The forum's "scoring timeout"
threads are a red herring — a GRANDMASTER answers thread 724917 directly: *"this is not
caused by scoring; local scoring on all train data runs just over 1 minute. All the other
time goes into constructing your prediction."*

**`geff` is no longer needed at test time.** It was only ever read for
`estimated_number_of_nodes`, and that number is now predicted from the image. The
submission path needs `zarr` and nothing else — which matters because internet is off
during a scored rerun and `pip install` does not run.

---

## What to do next

1. **⭐ Build the submission notebook and get a leaderboard number.** This is now the
   binding constraint. We have a defensible configuration at CV 0.7070 and **no LB score
   at all**, five and a half weeks from the deadline. Until a submission scores, every CV
   number is an untested assumption about the metric, the rerun mount, and the CSV format.
   Constraints: notebook-only, no internet (so nothing may `pip install`), `test/` swapped
   at rerun so names must be globbed, `id` column required, no `max_frames`.
2. **The budget regression must be fit inside the submission**, on `train/`, at runtime —
   not hardcoded. §2 shows a stale or constant budget is worth −0.0882. Reading train
   budgets needs `estimated_number_of_nodes` out of a `.geff`, which is a zarr group, so a
   zarr-only reader avoids depending on the `geff` package.
3. **After that, edge quality is the only axis left.** Division Jaccard is **0.000 in every
   arm we have ever run** — the term is worth 0.1 of the 1.1 maximum and we collect none of
   it. `allow_divisions=False` has never been turned on.
4. Standing: best gated CV **0.7070**, best gated CV that reads privileged metadata 0.7098,
   still **no leaderboard score**. Leaderboard median 0.890.
