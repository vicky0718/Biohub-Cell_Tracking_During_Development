# The division funnel, measured — and what I will conclude before I see the score

Written **before** `claude-score` returns, deliberately. This project's two worst calls —
ranking arms on a proxy that was 0-for-4, and four rounds of pip theories — share a shape:
a number arrived, and the reading of it was chosen afterwards. So the reading goes first.

## 1. What is already measured, at zero cost

Both eval kernels completed, and the fork prints a division funnel per movie that nobody had
read. Summed over the 12 held-out movies (`claude-eval-ttasec`):

```
divergence_rejected            5,130      <- SAFE_DIV_DIVERGE_UM 2.25
geometric_candidates             975      <- survive SAFE_DIV_MAX_UM 9.0 / SISTER 14.0
deepcenter_rejected              752      77% of candidates, at threshold 0.25
post_veto_candidates             223
safe_divisions_added             194
```

And the submissions carry exactly that many forks — 194 for `ttasec`, 190 for `ttaz16` —
confirmed by counting out-degree≥2 source nodes in the CSVs. Nothing downstream prunes them.

**The DeepCenter veto is the dominant filter at the candidate stage, and the divergence gate
is dominant overall.** Neither has ever been evaluated against `div_J`.

## 2. The comparison that looks alarming and is not

`notes/57`: **151 binary divisions across 199 training movies** ≈ 0.76 per movie, so ~9 in
twelve. Against 194 predicted, that reads as 21× over-prediction — and would say the whole
division direction is inverted, that we should be cutting divisions, not adding them.

Two things say otherwise, and both were already in the notes:

1. **`notes/57` records `div_J` = 0.1154 on the old pipeline**, with those same 151 GT
   divisions. A 21× over-prediction cannot produce 0.115; it caps out near 0.046.
2. The official scorer charges **FP = `matched_pred_divisions - tp`**
   (`division_metrics.py:449`) — a predicted division is only chargeable if its dividing node
   **matches a GT node within 7 µm**. The GT is sparse: the node budget the metric divides by
   is `estimated_number_of_nodes` from the geff metadata, an *estimate* of a total the GT
   graph itself does not contain. Predicted forks on unannotated cells are free.

So 194-vs-9 is apples to oranges, and "we over-predict divisions 21×" is **not** established.
Writing it down because it was my first reading and it was wrong.

## 3. Where the fork's gates actually sit against ground truth

`notes/57`'s GT distribution, against the 0.946 fork's constants:

```
                        median     p90     p95     max     fork gate      percentile
parent->daughter          7.13   10.05   11.78   13.53    9.0  (MAX_UM)      ~p85
sister<->sister          10.57   14.36   15.34   20.30   14.0  (SISTER)      ~p88
```

This is the correction to `notes/57` §1 that matters: those 88%-rejection figures are for
**our retired `pipeline/divisions.py`** at 4.5/6.8, not for the fork. The fork's gates reject
roughly **12–15%** of real divisions, not seven of eight. The geometry is close to right
already, which makes the geometry the *small* lever and the two learned/heuristic vetoes the
large ones.

## 4. Pre-registered readings

`claude-score` reports `division_tp/fp/fn` per movie. `tp + fn` is the GT division count.

**Gate first — nothing below is read until all three hold** (the acceptance bar chosen for
this harness):

* `ttasec` ranks **above** `ttaz16`. LB says 0.945 vs 0.942.
* `ttasec`'s level lands near 0.945, allowing contamination optimism (`notes/72` §3).
* `tp + fn` over 12 movies is roughly **5–15**, matching `notes/57`'s 151/199.

If the ordering fails, the offline direction is closed and I report that — not a workaround.
If `tp + fn` comes back at, say, 200, then `notes/57`'s count is measuring something narrower
than the scorer does, and every division number in these notes needs redoing before use.

**Then, and only then:**

| measured | reading | next arm |
|---|---|---|
| `div_J` ≥ 0.25 | near mikelou1's 0.3; ≤ +0.005 left | **close the direction**, go back to edges |
| `div_J` 0.10–0.25, `fn` > `fp` | recall-limited: real divisions are being vetoed | loosen **DeepCenter veto** (0.25 → 0.15), which kills 77% of candidates |
| `div_J` 0.10–0.25, `fp` ≥ `fn` | precision-limited | tighten, and `dc40`'s board loss gets a cause |
| `div_J` < 0.10, `fn` dominant | the 5,130 divergence rejections are the story | loosen `SAFE_DIV_DIVERGE_UM` 2.25 |

Geometry (`SAFE_DIV_MAX_UM` 9.0 → 11.8, `SISTER` 14.0 → 15.4, both p95) is worth one arm in
any recall-limited branch, but §3 says it is the smaller lever and it should not go first.

**`fn` vs `fp` is the whole decision.** Both come back from the same run, and the plan's
step 3 — "loosen toward the GT distribution" — is only right in two of four branches. It was
written before the funnel was read.

## 5. Build mechanics, settled while waiting

The 0.946 base quotes env assignments with **single** quotes; `env()`/`guard()` in
`_mk_claude_arm.py` emit double and match only the lb941 family. Added `env1()`/`guard1()`,
and verified all six division knobs plus both guard lines resolve to **exactly one** anchor in
`claude_arm_tta946_source.json`. Guarded by `_EXPECTED_NUMERIC` on this base, so needing a
paired guard edit: `SAFE_DIV_MAX_UM`, `DEEPCENTER_SAFE_DIV_THRESHOLD`. Free to edit alone:
`SAFE_DIV_SISTER_MAX_UM`, `SAFE_DIV_DIVERGE_UM`, `SAFE_DIV_SISTER_SYMMETRY_TAU`,
`ILP_DIVISION_WEIGHT`. `OUTPUT_SAFE_DIVISIONS` is never assigned in the env block, so
switching it off is an insertion, not a replacement.
