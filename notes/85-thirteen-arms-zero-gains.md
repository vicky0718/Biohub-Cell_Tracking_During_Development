# Thirteen arms, three days, zero gains — the knob surface is exhausted

```
ttaret85      0.945  (base 0.944)  +0.001   retention 0.90 -> 0.85
ttasecw20     0.945   0.000   mrr12      0.945   0.000   gapdc15  0.945  0.000
ttadom        0.945   0.000   ttasecret85 0.945  0.000   mtl8     0.945  0.000
sdw90         0.944  -0.001   gap65      0.944  -0.001
mtl4          0.943  -0.002   ttadse44   0.943  -0.002   bew10    0.943  -0.002
dse52t        0.943  -0.002
```

**Best remains `ttasec`, 0.945.** Every knob tested is at a local optimum or worse.

## 1. Additivity failed a third time, and this one had a reason

`ttasecret85` was the batch's one real hypothesis: `ttasec` banks +0.001 from secondary-TTA,
`ttaret85` measured +0.001 from the retention floor on the `tta946` base, and the two looked
like different stages. Predicted 0.946. **Got 0.945.**

The prediction was wrong, and the flaw is visible in hindsight. `ttasec` averages the
**secondary model's** edge features over eight D4 views; the retention floor governs
**dual-seed** candidate retention — and the dual-seed pathway *is* the secondary detector's.
They are not disjoint stages, they are two corrections to overlapping errors. Once the
features are eight-view averaged, the candidates the floor would have rescued are already
being kept. I called them disjoint on the basis that they live in different parts of the file,
which is not the same thing.

Prior art I under-weighted: `sewdet` — SEW +0.001, DET +0.001, together **0.000**, with node
counts exactly additive. Same shape. Two-for-two before this; three-for-three now.

## 2. Knobs bracketed on both sides are closed

```
DUAL_SEED_EDGE_THRESHOLD   0.44 -> -0.002    0.52 -> -0.002    0.48 is optimal
OUTPUT_MIN_TRACK_LEN       4    -> -0.002    8    ->  0.000    6 is optimal (asymmetric)
BIDIRECTIONAL_EDGE_WEIGHT  0.10 -> -0.002    (author pinned it twice; board agrees)
```

`OUTPUT_MIN_TRACK_LEN` is the interesting shape: cutting it hurts, raising it does nothing.
The filter is removing fragments that cost nothing to keep and nothing to drop — it is not
where the score lives.

## 3. What the whole exercise established

Thirteen arms across every stage of the post-processing chain — detection blending, edge
fusion, gap closure, track filtering, division repair, dual-seed thresholds, motion relink —
and **not one moves the leaderboard up**. Combined with `notes/78`–`notes/81`, which closed
divisions and the relink on mechanism, the finding is:

> **The post-processing chain of this pipeline is at a local optimum.** Its authors tuned it
> well. There is no thousandth left in the knobs, and the one knob that ever paid (+0.001,
> twice) is already banked in `ttasec`.

That is worth knowing with confidence, and it cost 13 slots out of ~115 to learn honestly.
The alternative — inferring it from an offline harness — is exactly what produced `norelink`.

## 4. What is left

One thing, and its closing argument has a hole (`notes/81` §4): the fine-tune. `notes/75`
recorded 0/30 epochs beating the baseline, but the selection ran on movies the 400-epoch
checkpoint had trained on, and the builder seeds

```python
best_score = _b_acc * _b_recall     # from the untouched checkpoint
```

so the bar is the memorising baseline's inflated holdout score and **no epoch could clear it**.
The run's artifact is the baseline. Seeding `best_score = -1.0` instead saves the best
*fine-tuned* epoch — still selected on contaminated data, but the bias mostly cancels between
two fine-tunes of the same base, where it does not between a memoriser and its replacement.

~4 GPU-hours, one slot, blind submission. It is the only untested direction with a plausible
+0.003 rather than +0.001, and it can equally return 0.93. With 11 days left it is this or
nothing.

## 5. The fine-tune ran, and this time it produced a model

`claude-train-elastic` v16, Tesla T4, 6 h 22 m, 30 epochs on 169 movies.

```
BASELINE (public checkpoint, untouched)   acc=0.9998  recall=0.9692  score=0.9690
best fine-tuned epoch                     acc=0.9999  recall=0.9616  score=0.9682
FULL restore: 136/136 tensors loaded
```

**The `-1.0` seeding worked.** `notes/75`'s run saved the baseline because the bar was seeded
from it; this one saved a genuinely fine-tuned checkpoint to `/kaggle/working/claude_finetuned`.
That is the fix doing what it was meant to do, independent of whether the model is any good.

Training was healthy — edge loss 0.0001, detection loss 0.0069 at epoch 29, no divergence.

### The shape of the change, stated before the board rules on it

The fine-tune **raised accuracy** (0.9998 → 0.9999) and **lowered recall** (0.9692 → 0.9616).
It moved along the precision/recall tradeoff toward conservatism: fewer detections, fewer
nodes. Every measurement in this project says lost edge true-positives dominate the small
`adj` bonus that a more negative node ratio buys (`notes/77` §7, and `det955`'s neutral
result). **So the expectation on record is that `ftune` scores at or below 0.945.**

The one argument on the other side, and the reason it is still worth a slot: that recall gap
is measured on movies the 400-epoch baseline **trained on**. Its 0.9692 there is inflated; the
fine-tune's 0.9616 is less so. The true gap may be smaller, or reversed. That is precisely the
bias `notes/81` identified, and precisely why this has to be settled blind on the leaderboard
rather than on the holdout — the holdout cannot answer it, in either direction.

`claude-arm-ftune` is running the verification pass now.

## 6. GPU quota exhausted — and the tool was hiding Kaggle's own explanation

`claude-arm-ftune` was pushed six times and discarded six times. `busy_slots()` reported
**none** running, so the "no free GPU session" diagnosis `run_arm.py` printed was wrong. The
real reason was in every push response, in a field the tool never read:

```json
{"versionNumber": 0, "error": "Maximum weekly GPU quota of 30.00 hours reached."}
```

**Kaggle said exactly what was wrong, six times, and we printed a guess instead.** This is the
third instance of the same bug class in one week — "6 consecutive P100 draws" for a
concurrency discard, a 429 reported as a failed run, and now a quota wall reported as a busy
slot. Each time a transport- or policy-level signal was rendered as a claim about the work.
`run_arm.py` now reads `error`/`errorNullable` and, on a quota message, **stops instead of
retrying**: a busy slot clears in minutes, an exhausted quota does not clear for days, and
five more pushes are pure noise.

I also told the user the first discard was "transient and self-healing" before checking. It
was neither.

### Where the 30 hours went

```
claude-train-elastic   6 h 22 m   the fine-tune
~7 eval kernels        ~5 h       12-movie offline runs (notes/77-80)
~16 arm verifications  ~6 h 30 m  25 min each
graded reruns          the rest   one per submission, under the arm's own slug
```

The fine-tune alone took a fifth of the week's budget. Nothing is broken; the resource is
simply spent until the weekly reset.

### What still works

Submissions of **already-built** arms are unaffected — those notebooks exist on Kaggle and
their graded reruns are Kaggle's business, not a fresh push from us. `ftune` is the one thing
blocked, because its verification pass has never run.
