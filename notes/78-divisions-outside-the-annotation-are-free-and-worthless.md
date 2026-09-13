# `divloose` is a clean null — and it tells us exactly where the division work has to aim

`claude-eval-divloose`, 12 held-out movies, self-scored with the official metric.

```
                 forks emitted   edge_jaccard          division tp/fp/fn   div_J
ttasec                     194   0.932814930015552             2 / 2 / 9  0.15385
divloose                   155   0.932814930015552             2 / 2 / 9  0.15385
```

**39 fewer divisions and `edge_jaccard` is identical to sixteen significant figures.** Not one
of the 39 removed forks changed a single edge TP, FP or FN, and none of them was a division
TP or FP. They were invisible to the metric in both terms simultaneously.

That is `notes/77` §6 confirmed the hard way: the GT annotates ~4.7% of a movie, and both the
edge term (`pred_valid`) and the division term (`count_matched_pred_divisions`) ignore
everything outside it. Divisions in unannotated regions are free to emit and worthless to have.

## 1. I had the gate's sign backwards, and it inverts a board reading

The arm is misnamed. The gate is

```python
if grandchild_dist - sister_dist < SAFE_DIV_DIVERGE_UM:
    stats['safe_division_divergence_rejected'] += 1
    continue
```

It **requires** the daughters to diverge: the grandchildren must separate by at least
`SAFE_DIV_DIVERGE_UM` more than the sisters do. So **raising** the threshold demands more
divergence and rejects more, which the funnel confirms:

```
                geometric_candidates   divergence_rejected   added
ttasec   2.25                    975                 5,130     194
divloose 3.00                    644                 5,461     155
```

`divloose` is **stricter**, not looser. Which means `div15` — `2.25 → 1.5`, scoring **0.938**
against a 0.941 base — was the *loosening*, and it **lost 0.003 on the board**.

`notes/76` and `notes/77` both described `div15` as tightening. That was wrong, and it matters:
it was cited as evidence that cutting divisions loses. On this gate the opposite is measured.
`dc40` remains correctly read — its threshold genuinely tightens the DeepCenter veto.

## 2. What is actually left of the division thesis

Standing, still measured:

* We recover **2 of 11** GT divisions. The 9 FNs are real and permanent.
* Emitting a division costs ~0.01 FP on average (194 forks → 2 FPs).
* `dc40` tightened the DeepCenter veto and lost 0.008 on the board.

Retracted or weakened:

* "`div15` tightened and lost" — **backwards**; it loosened and lost.
* "Loosening gates raises `div_J`" — `divloose` moved 39 divisions and moved `div_J` by
  **zero**. Gate changes that add or remove divisions *uniformly across the movie* do nothing,
  because ~95% of the movie is unscored.

## 3. The target is narrower than "more divisions"

`div_J` can only move if a fork lands on a GT node that is annotated **and still has a child**.
There are 11 such sites in 12 movies. Loosening a geometric gate scatters new forks over the
whole movie and ~5% of them land where anything is scored — which is why 39 divisions bought
exactly nothing.

So the question for `dcloose` is not "did the count go up" but **"did `division_tp` go above
2"**. That is a nested comparison on the same 11 sites, and it is the only division number
this harness can answer. If `dcloose` also returns 2/2/9, then the 9 misses are not gate
rejections at all — the pipeline never proposes a fork at those sites — and the whole
division direction closes on the post-processing knobs, leaving only the detector/linker that
`notes/75` already measured as saturated.
