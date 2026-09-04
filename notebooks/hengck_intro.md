# Implementing hengck23's two constructive suggestions

```
0.901 submitted (rank ~1388/3038)    0.938 = rank 100    0.947 gold
```

`notes/58` read all 43 of hengck23's posts. Most of what he proposes needs training (peak
detection loss, dense pseudo-labels, elastic augmentation) or an external model. **Two are
implementable on cached graphs, on CPU, today**, and both act on parameters that
`notes/60`'s rule says are worth checking: neither has ever been moved by score.

## Suggestion 1 — "long links are almost wrong (easy to filter such results)"

Posted 2026-07-17 with visualisations of the host baseline's own output:

> *"Visualisation results from host repo baseline code (temporal unet + link transformer).
> i show max prob link. **long links are almost wrong (easy to filter such results)**"*

`pipeline/repair.py::cap_edge_length` exists and does exactly this — and **it is not in the
shipped chain.** `notes/59` measured its 14.0 µm default as dropping 0.10% of real links and
concluded "correctly set", but that only checked the **recall** side. His claim is about
**precision**: long *predicted* links are mostly false. That has never been measured.

`notes/59`'s distribution says where to cut:

```
real single-frame GT links (n=128,883)   median 1.82   p95 5.34   p99 8.38   max 60.76
```

So 8.4 µm keeps ~99% of real links, 5.0 µm keeps ~95%. If his claim holds, cutting there
removes far more false edges than true ones.

## Suggestion 2 — divisions as a post-processing stage, with the gates fixed

Posted 2026-08-06:

> *"my suggestion is to learn track without cell division first (i.e. all tracks have only
> one BIRTH and DEATH). cell division is then handled at **post-processing or stage 2**…
> it is difficult even for humans to decide if there is cell division just based on two
> frames."*

`pipeline/divisions.py::insert_divisions` is precisely that stage, and it is **also not in
the shipped chain**. `notes/25` planned a probe for it and no note ever recorded a result.

**`notes/57` is why this is worth running now.** Its gates default to `max_um=4.5`,
`sister_max_um=6.8`, and measured against all 151 ground-truth divisions those reject
**88.7%** and **86.8%** of real events. Any earlier evaluation was testing a function that
discards seven of every eight targets by construction. With the gates set from the data:

```
GT divisions (n=151)   parent->daughter  median 7.13   p90 10.05   p95 11.78   max 13.53
                       sister<->sister   median 10.57  p90 14.36   p95 15.34   max 20.30
```

## What is deliberately NOT here

`notes/60` bought a rule and it applies: **before checking a constant, ask whether score has
ever been allowed to move it.** `close_gaps`' radius had been swept four times, geometry said
widen it, and the metric said no. Both parameters here are in the other category — adopted
from a source, never tuned, never shipped.

Also excluded: things hengck23 reports as *problems* rather than proposals — sparse-label
overfitting past epoch 10, ultrack/byotrack disagreeing on large movements, "co-tracking
alone will not win". Those are diagnoses, not experiments.

And his duplication metric property (`notes/58` §1) stays out on `notes/54`'s reasoning: the
organisers patched one exploit already and are actively closing that surface.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `base` equals `claude_divsweep`'s `inc/g2sp6` — 0.9188, `div_J` 0.1154,
   1,443 forks. Otherwise nothing below is comparable.
2. **Long links are worth filtering.** Some `cap_*` arm beats `base` by more than 0.0015
   (`notes/44`'s floor). This is suggestion 1, and failing it says his observation about the
   host baseline does not transfer to our chain.
3. **The broken gates were the reason insertion looked dead.** `div_old` (4.5 / 6.8) inserts
   fewer than half the forks that `div_p90` does. A mechanism check on `notes/57`, not a
   score claim.
4. **Corrected gates raise `div_J`** above `base`. Suggestion 2's actual test.
5. **The best arm holds in sign on BOTH embryos.**

*Prediction 3 is the one I expect to pass regardless of whether 2 or 4 do — it tests
`notes/57`'s measurement rather than hengck23's advice, and separating them is the point.*
