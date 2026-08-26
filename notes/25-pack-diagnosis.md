# The 0.046 is in the division term, and my first read of question 2 was wrong

`claude_pack_diag` v1, 24 datasets stratified, ILP on, `det_threshold=0.985`, 1,442 s.

| | |
|---|---|
| our LB with the pack + ILP | **0.867** |
| the cluster, same weights | **0.913–0.916** |
| this run, on (contaminated) training data | 0.9304 |

Two questions were pre-registered. **Question 1 has a decisive answer. Question 2's answer
in the notebook's own output is confounded and has the wrong sign**; §2 corrects it, and
the correction is the more useful result.

---

## 1. Question 1 — which term is short? Divisions, by a mile

```
score              0.9304          node_recall        0.9950
adj_edge_jaccard   0.9304          budget multiplier  1.0012   (ceiling 1.1)
edge_jaccard       0.9293
division_jaccard   0.0000          divisions: TP=0  FP=0  FN=27   (37 forks predicted)
```

**The division term contributes 0.0000 of a possible 0.1000.** Not "a little short" — zero.
That is more than twice the entire 0.046 gap to the cluster sitting in one term.

The other two terms are close to done:

- **Node budget is neutral.** 1.0012 against a 1.1 ceiling sounds like 0.099 of headroom,
  but the ceiling is reached at `total_node_ratio = −1.0`, i.e. predicting no nodes at all.
  §2 shows the practically reachable part of it is small.
- **Edge Jaccard is 0.9293** with node recall 0.9950. `notes/04` §7 measured the entire
  linking problem as worth ≤ 0.015 once nodes are given, so most of what is left here is
  detection, not linking.

### Why zero, and what that implies — read from the scorer, not inferred

`division_metrics.py` (official repo, read directly). A predicted fork is charged as a
false positive only if it is in one of three sets:

| set | rule |
|---|---|
| `evaluable_forks` | the fork's node matched a GT node **with out-degree ≥ 1** |
| `considered` | the fork sat inside a GT division's local window and failed the topology test |
| `invalid_forks` | the fork's two child branches resolve to distinct GT components, or a child has another parent (`malformed`) |

**None of the pack's 37 forks hit any of those.** They are unevaluable — placed on nodes
the sparse ground truth never annotated.

Two consequences follow from the metric's arithmetic, and both matter:

**(a) The division denominator is `FP + D`, not `k + D`.** Per dataset
`FN = D − TP` by construction (`evaluate_divisions`: `fn = len(scores) - tp`), and
`summarise` micro-averages, so pooled

```
    division_jaccard = TP / (TP + FP + FN) = TP / (FP + D)
```

with `D` = the GT's own division count, fixed. **Adding a true positive never raises the
denominator.** Adding an *unevaluable* wrong fork changes nothing at all. Only a
*chargeable* wrong fork dilutes. From the current state (TP = 0), the marginal change from
one speculative fork is `p / (FP + q + D) > 0` for any non-zero hit rate `p` — it cannot
make the division term worse than the 0.0000 it already is.

**(b) `notes/04` §10's ">49.6 % or it doesn't pay" is the EDGE break-even, and it still
holds — for the edge it adds.** Re-derived: adding one speculative edge to a graph at edge
Jaccard `J` pays iff `p > J/(1+J)`. At `J = 0.93` that is 48.2 %; at the 0.9915 linking
ceiling, 49.8 % — which is exactly where §10's figure came from. So §10 is not overturned;
it was answering a different question. A speculative sister link is **one edge and one
possible division at once**, and the two terms price it differently:

- as an edge it is worth `≈ [p − (1−p)·J] / D_edge`, with `D_edge ≈ 5,000` per dataset —
  order `10⁻⁴`, and negative below 48 % precision;
- as a division it is worth `0.1 · p / (FP + D)`, with `D ≈ 27` across all 24 datasets —
  three orders of magnitude more leverage per event.

**The division term is where a speculative link is cheap and the edge term is where it is
expensive, and the division term dominates by ~1000× per event.** That is the finding.

### What is not yet known, and must be measured rather than assumed

How large is the chargeable FP surface once forks are placed *deliberately* rather than
incidentally? The exposure is roughly the fraction of predicted nodes that are matched to
annotated GT, which in this run ranges from **1.3 %** (`44b6_e28840c6`, 311 annotated
against 23,105 predicted) to **6.5 %** (`6bba_09961292`, 1,950 against 30,135). At ~3 %,
emitting 200 forks per dataset would produce ~6 chargeable FPs per dataset — ~150 across
this subset against `D = 27`, which would put `J_div = TP/177` and cost most of the term.
Emitting 20 well-chosen forks per dataset would produce ~0.6.

So the shape of the answer is **precision-per-fork against forks-per-dataset**, and it is
not derivable from here. It needs a sweep. §4.

One implementation constraint falls straight out of the FP rules and should be built in
from the start rather than discovered: **the second child must be a detection with
in-degree 0.** Attaching a sister that already has a parent makes the fork `malformed`,
which is an automatic FP under rule three.

---

## 2. 🚨 Question 2 — my first reading was confounded, and the real correlation is negative

The notebook printed `corr = +0.569` and `** THESIS SUPPORTED **`. **Do not use that line.**
It stratified and correlated against the **annotated GT node count**, and annotation rate
varies **20×** between the two embryos (`notes/04` §9: `6bba` labelled 1-in-8, `44b6`
1-in-167). So that count sorts datasets by *labelling protocol*, not by size: every
small-count dataset in the subset is `44b6` and every large-count one is `6bba`. The
"size" axis was an embryo-identity axis wearing a hat.

Recomputed against the scorer's actual node budget `estimated_number_of_nodes` — which is
both the true dataset size and the quantity the multiplier is computed against:

```
corr(log10 BUDGET,    node ratio) = -0.643    <- the size test
corr(log10 annotated, node ratio) = +0.569    <- what the notebook printed; confounded

smallest third by budget  +0.0199
largest third by budget   -0.2994      drift  -0.3193

  44b6   n=9    mean ratio -0.3143  sd 0.2856   budget 22,110..74,773   corr -0.810
  6bba   n=15   mean ratio +0.0380  sd 0.0873   budget  5,033..38,714   corr +0.042
```

**The corrected correlation is stronger and runs the other way.** Sorted by true budget:

| dataset | budget | annotated | pred | ratio | edge_J |
|---|---|---|---|---|---|
| `6bba_aeee7805` | 5,033 | 795 | 5,305 | +0.054 | 0.9793 |
| `6bba_062c8d37` | 6,033 | 930 | 6,268 | +0.039 | 0.9978 |
| `6bba_969618f6` | 13,883 | 679 | 16,812 | **+0.211** | 0.9508 |
| `44b6_aaf8b0ea` | 22,110 | 209 | 15,278 | −0.309 | 0.9571 |
| `6bba_32db13fc` | 38,714 | 862 | 48,818 | **+0.261** | 0.7807 |
| `44b6_cf2536e8` | 57,621 | 76 | 23,740 | −0.588 | 0.8554 |
| `44b6_9bfa6a0a` | 64,291 | 264 | 34,203 | −0.468 | 0.8536 |
| `44b6_71a4179f` | 67,661 | 121 | 28,147 | −0.584 | **0.5269** |
| `44b6_e28840c6` | 74,773 | 311 | 23,105 | **−0.691** | 0.8455 |

*(full 24 rows in the run log; these are the ends and the outliers)*

The within-embryo split is the part that keeps this honest. `44b6` alone gives **−0.810**,
so the effect is not merely "the two embryos differ". `6bba` alone gives **+0.042** — flat —
but `6bba`'s budgets stop at 38,714 while `44b6` runs to 74,773, so `6bba` never reaches
the range where the drift appears. A flat correlation over half the range is not a
refutation, and I am not going to read it as one.

### But the sign means the drift is mostly *not* a loss

This is the part the confounded reading would have got backwards. On the largest datasets
the pack **under**-predicts, and under-prediction is what the multiplier **rewards**:
`max(0, J·(1 − 0.1·ratio))` pays `1.069` at `ratio = −0.691`. `44b6_e28840c6` is already
collecting a 6.9 % bonus.

Measured across the 24:

```
per-dataset multiplier   mean 1.0094   min 0.9739   max 1.0691
mean edge_J 0.9090   mean adj 0.9165   (pooled multiplier 1.0012)
```

So a per-dataset budget calibration has two genuinely available pieces and one mirage:

- **Real, small: the over-predicting tail.** `6bba_32db13fc` (+0.261, mult 0.974) and
  `6bba_969618f6` (+0.211, mult 0.979) are paying a penalty. Pulling them to `ratio = 0`
  is worth ~2 % of their edge Jaccard — **2 datasets in 24, ≈ 0.002 pooled**.
- **Real, unquantified: `44b6_71a4179f` at edge_J 0.5269** is a genuine outlier failure and
  has nothing to do with the budget term. Worth its own look, not a calibration.
- **Mirage: the 0.099 of "headroom" to the 1.1 ceiling.** Reaching it means predicting
  nothing. The reachable part is the ~0.009 already being collected plus the ~0.002 above.

**The per-dataset budget regression is therefore worth roughly 0.002, not 0.02.**
`notes/24` §4 and `notes/16` §2.3 both argued the multiplier was the term whose arithmetic
reaches 0.935. On this measurement that is wrong: the multiplier is nearly saturated
already, and the arithmetic that reaches 0.935 runs through **divisions**.

---

## 3. What this reprioritises

The differentiator I have been planning around since `notes/24` — per-dataset budget
calibration against their single global `det_threshold` — survives as a real effect and
dies as a *priority*. It is worth ~0.002. The division term is worth up to 0.1 and is
currently returning 0.000.

| lever | measured / bounded at | status |
|---|---|---|
| **division term** | 0.000 of 0.100 | **next** |
| per-dataset budget calibration | ~0.002 | park |
| linking / edge repair | ≤ 0.015 total (`notes/04` §7) | after |
| `44b6_71a4179f`-class outliers | unquantified, 1 in 24 | investigate cheaply |

This also explains the pack manifest's own framing. It ships the *"ILP candidate graph
before notebook-level graph repair"*, and the public notebook's repair contains
`add_safe_divisions_postlink` with constants `SAFE_DIV_MAX_UM = 4.5`,
`SAFE_DIV_SISTER_MAX_UM = 6.8`, `SAFE_DIV_FRAME_FRAC_CAP = 0.008`. That last one caps
inserted divisions at 0.8 % of a frame's nodes — at ~500 nodes/frame over ~50 frames, of
order 200 per dataset, against the **1.5 per dataset** the raw ILP emits. The cluster is
running a division inserter at roughly 100× our fork rate and we are not.

---

## 4. Next: `claude_div_probe`, one prediction pass, a swept insertion

Designed as a measurement, not a bet, because §1 ends in a number I cannot derive.

- **Predict once, sweep many.** The 24 post-ILP graphs cost 1,442 s to produce. Cache them
  (coords + edges) and run every insertion variant against the cache in numpy. The sweep is
  then nearly free and the probe is one short run rather than one run per setting.
- **Reimplemented from the algorithm.** The public notebook's `licenseName` is `None`, so
  its code is not usable. The geometric idea — a second child within a radius, sisters
  within a wider radius, capped per frame — is the approach; the implementation is ours.
- **Hard constraints from §1's FP rules, built in:** the sister must have **in-degree 0**
  (else the fork is `malformed` → automatic FP), and it must be at exactly `t+1`.
- **Swept:** `frame_frac_cap ∈ {0, 0.002, 0.008, 0.02, 0.05}` crossed with
  `max_um ∈ {4.5, 6.0}`. `cap = 0` reproduces the current 0.9304 and is the drift control.
- **Reported per cell of the sweep:** division TP / FP / FN, `division_jaccard`,
  `edge_jaccard`, `adj_edge_jaccard`, total score, and forks emitted.

**Pre-registered predictions**, to be graded honestly whichever way they land:

1. `cap = 0` reproduces 0.9304 ± 0.0005. If it does not, the cache or the scoring path is
   wrong and nothing else in the run is readable.
2. Chargeable FPs grow **sub-linearly** in forks emitted, because most forks land on
   unmatched nodes. If FP tracks forks 1:1 instead, §1(a) is wrong about the FP surface and
   the whole division angle is much weaker than this note claims.
3. The score curve is **single-peaked** in the cap, and the peak is at a cap far above the
   pack's current ~1.5 forks/dataset.
4. Edge Jaccard falls monotonically as forks are added, and by less than the division term
   gains at the peak. If edge losses dominate, §1(b)'s 1000× leverage estimate is wrong.

Failing 2 or 4 kills the division angle and sends the effort to edge repair. That is the
point of running it before building the inserter properly.

Contamination caveat from `notes/24` §2 applies unchanged: these are training datasets the
pack may have been fitted on, so **absolute levels are inflated**. Every number this run is
for is a *delta* across the sweep on fixed datasets, which contamination shifts equally.

---

## 5. Fixed in the code, so this cannot recur

- `claude_pack_diag` now stratifies and correlates on `estimated_number_of_nodes`, prints
  the annotated-count correlation **beside** it explicitly labelled as confounded, reports
  within-embryo correlations with each embryo's budget range, and warns that a flat
  within-embryo correlation is only evidence of absence if that embryo spans the range.
  The `** THESIS SUPPORTED **` wording is gone; it now names the sign and says to confirm
  it against the table.
- `Harness.score_graph` now puts **`n_total`** and `n_gt_annotated` on every scored row, so
  no future analysis has to reach for whichever node count happens to be at hand. The
  comment says why the annotated count is not a size.

The banked floor is unchanged at **0.752**; the pack submission stands at **0.867**.
