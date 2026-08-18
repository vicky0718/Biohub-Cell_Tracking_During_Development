# Experiment #5 — pruning is free quality, and the two embryos want opposite things

`06_prune_scales.ipynb`, run on Kaggle 2026-08-18, 2.2 h on the fixed 60-dataset subset.
Leave-one-embryo-out folds: 0 = `44b6` (21), 1 = `6bba` (39).

**Reproduction check passed exactly**: `base_nocap` 0.6671 against `05`'s 0.6671, drift
+0.0000. The harness is deterministic across runs.

| arm | SCORE | edge_J | recall | nodes | ratio |
|---|---|---|---|---|---|
| **prune + cap** | **0.7072** | 0.7013 | 0.870 | 1,126,596 | −0.098 |
| scales wide | 0.6975 | 0.7024 | 0.837 | 994,642 | −0.037 |
| cap | 0.6957 | 0.6931 | 0.885 | 1,176,957 | −0.051 |
| scales verywide | 0.6879 | 0.6787 | 0.779 | 758,687 | −0.244 |
| prune | 0.6838 | **0.7260** | 0.894 | 1,366,573 | +0.408 |
| scales spread | 0.6715 | 0.7191 | 0.886 | 1,376,197 | +0.459 |
| base (no cap, no prune) | 0.6671 | 0.7195 | 0.904 | 1,444,570 | +0.528 |
| scales tight | 0.3065 | 0.6551 | 0.938 | 3,963,447 | +6.230 |

**New best: 0.7072.** But read §3 before treating that as progress.

---

## 1. ⭐ Pruning does not just save budget — it *improves* edge quality

| vs base | nodes removed | edge Jaccard |
|---|---|---|
| prune | −5.4 % | **+0.0065** |
| cap | −18.5 % | −0.0264 |
| prune + cap | −22.0 % | −0.0182 |

`prune` produced the **highest edge Jaccard we have ever measured: 0.7260.**

That is stronger than prediction 2 asked for. The hypothesis was that isolated nodes are
*inert* — pure budget cost, no edge contribution — so removing them should be roughly free.
Instead removing them made the edge metric **better**. The reason is in the metric: node
matching is a bipartite assignment, and an isolated predicted node sitting near an
annotated GT node can **win that match** and thereby deny it to a predicted node that
actually has edges. Delete the freeloader and the linked node matches instead, converting
lost edges into TPs.

So pruning is not a budget tactic. It is a **matching-quality** tactic that also happens to
save budget, and it should be on in every configuration from here.

## 2. Prediction 1 was the wrong question

Framed as "pruning beats capping", it came back **FALSIFIED** — prune 0.6838 vs cap 0.6957,
−0.0119. But that framing treated them as alternatives, and they are not:

- pruning removes nodes that contributed nothing → **quality up**, budget down a little
- capping removes the weakest peaks blindly → **quality down**, budget down a lot

Together, **0.7072** — better than either. The right question was "do they compose", and
the answer is yes. My prediction set that up as a contest and so scored a real result as a
failure.

## 3. 🚨 The two embryos want opposite detector settings — and that is now three for three

Every fold breakdown we have points the same way once you line the baselines up correctly:

| comparison | `44b6` | `6bba` | reading |
|---|---|---|---|
| `05`: cap vs no cap | **−0.0286** | +0.0389 | fewer nodes helps `6bba`, hurts `44b6` |
| `06`: prune vs cap | **+0.0378** | −0.0209 | *more* nodes (prune keeps 18 % more than cap) helps `44b6` |
| `06`: wide vs default scales | **−0.0597** | +0.0468 | fewer nodes helps `6bba`, hurts `44b6` |

All three agree: **`44b6` wants a denser detection field, `6bba` wants a sparser one.**
Consistent with recon — `44b6` has a median budget of 32,681 nodes (~327 cells/frame) and
`6bba` 9,691 (~97/frame). More than a 3× difference in true cell density.

*(An earlier draft of this analysis called the `06` signs a "flip". They are not — `06`
gates `prune` against `cap`, so a positive `44b6` delta there means the same thing as the
negative `44b6` delta in `05`. Three consistent observations, not two contradictory ones.)*

This is why every node-reducing change keeps failing the gate. It is not noise, and it will
not be fixed by finding a better global setting — **there isn't one.** And the hidden test
set is two *different* embryos whose densities we cannot know.

## 4. What has actually passed the gate

| run | comparison | verdict | score |
|---|---|---|---|
| `04` | DoG sep 6.0 no-cap **over intensity** | **PROMOTE**, all folds | **0.6760** |
| `05` | sep4.5 + cap over sep4.5 no-cap | REJECT (`44b6` −0.0286) | 0.6957 |
| `06` | prune over cap | REJECT (`6bba` −0.0209) | 0.6838 |
| `06` | wide scales over default | REJECT (`44b6` −0.0597) | 0.6975 |
| `06` | **prune + cap** | **never gated** | **0.7072** |

**The best gate-passing configuration is still `04`'s 0.6760.** Everything since is either
rejected or untested against the gate — including the 0.7072 headline, which `06` never
compared to anything. That is a gap in the notebook: it printed a "BEST THIS RUN" line for
an arm it never subjected to the promotion rule.

## 5. Scale tuning: the wide direction is real but embryo-split

`tight` `[(1.0,2.5),(1.5,4.0)]` is catastrophic — 3.96M nodes, ratio +6.23, score 0.3065.
Fine scales in a dense field is the `notes/09` failure mode again.

`wide` `[(2.0,5.0),(3.0,8.0)]` gives +0.0304 pooled and lands naturally near budget
(ratio −0.037) without any cap. It fails the gate only on the embryo split. The direction —
**larger DoG sigmas** — is worth keeping; the specific value is not settled.

Note these arms ran with `KEEP="neither"` (the gate rejected both prune and cap), so **none
of the scale arms had pruning on.** Since pruning is +0.0065 of free quality, every scale
number here is understated.

---

## What to do next

1. **Turn pruning on permanently.** It raises edge Jaccard. There is no configuration where
   you would want to keep a detection that contributed no edge.
2. **⭐ Adapt density per dataset, not globally.** This is the principled answer to §3.
   Instead of truncating peaks to hit the budget (blind, quality-destroying), choose
   `min_separation_um` **per dataset** so the detector naturally emits about
   `estimated_number_of_nodes / T` per frame. Dense crops get tight separation, sparse crops
   get wide, and no embryo-level assumption is needed. It targets exactly the axis all three
   fold splits disagree on.
3. **Two bugs found while implementing (2), both of which would have cost a Kaggle run.**
   - `maximum_filter` with an explicit ball footprint costs O(N x |footprint|). On the
     isotropic 1.625 um grid a 12 um ball is 1,743 voxels, but on a finer grid it is
     27,067 — enough to wedge the process for minutes per frame. `BALL_MAX_VOXELS` now
     falls back to the separable box above 4,000 elements. The box is a superset of the
     ball, so `vol == box_max` implies `vol == ball_max`: the fallback yields a *subset*
     of true maxima, never spurious ones.
   - The first `adaptive_bounds` upper limit of 12 um only reaches **786 detections/frame**
     — the densest dataset in the corpus. Recon's median budget needs ~18.5 um and its
     minimum needs ~31 um, so the clamp would have bound on essentially every dataset and
     made the whole arm a no-op that looked like a null result. Now (2.5, 32.0).
4. **Gate the composite.** `prune + cap`, and any adaptive variant, must be compared against
   `04`'s 0.6760 champion under the promotion rule before any of it counts.
5. Standing: no leaderboard score, no submission notebook, best *gated* CV 0.6760. Public
   evidence puts the tuned classical ceiling near 0.85 against a leaderboard median of
   0.890, so this remains baseline work, not a route to the pack.
