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
