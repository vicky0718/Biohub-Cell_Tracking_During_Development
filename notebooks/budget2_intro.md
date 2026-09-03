# The third selection rule: rank tracks under a per-dataset budget

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
score = adj_edge_jaccard + 0.1*div_J,   adj = max(0, edge_J * (1 - 0.1*ratio))
ratio = (N_pred - N_est) / N_est        floor at 0, NO ceiling
```

`notes/51` established two things that make this run the obvious next one.

**Detection is now the ceiling.** At the shipped chain `fn_detect` is 583 (4.21% of GT
edges) against `fn_mislink` 226 (1.63%) — the exact inverse of `notes/26`. Perfect linking
is worth +0.020 to +0.035 against a 0.034 gap, so the graph side cannot get there alone.

**We have never used the per-dataset budget.** `notes/04` §9 said it plainly — *"the two
datasets that are the leaderboard have node budgets 11× apart, 64 vs 698 cells per frame. A
detector with one global threshold cannot serve both"* — and we still ship one global
`DET_THRESHOLD`. r35's `linker.py`, read for the first time in `notes/51`, carries
`max_pred_nodes` beside `rank_tracks_by_geometry`: *"Pivot H — drop short false tracks to cut
|V̂|/φ penalty"*, *"R11 — rank tracks by link geometry (tight long tracks) under budget"*.

## Why this is not the two thinning runs that already failed

Both previous attempts cut at the **detection** stage, before anything knew which detections
would end up in a good track:

```
claude_budget    pool_kernel_um, an NMS radius     node_recall 0.983 -> 0.537   notes/46
claude_topk      det_threshold, a confidence cut   0.901 -> 0.863 on the LB     notes/48,49
```

**Cutting after linking is different in kind.** Dropping a junk track removes its nodes — a
budget gain — *and* its false-positive edges — a Jaccard gain. A detector-stage cut removes
nodes that a surviving track needed, which is precisely why both runs above destroyed recall.

The mechanism is confirmed from two directions: `notes/45` derived it, and forum thread
739018 (Michael Hernandez, then TWEAK) independently read it off the released evaluator.
TWEAK's synthetic case is the sharp statement — *"with edge TP/FP/FN fixed at 90/5/5,
reducing predicted nodes from 100 to 50 changed adjusted-edge Jaccard from 0.900 to 0.945."*
A node that is not an endpoint of a kept edge is pure cost.

## The grid

`pipeline/repair.rank_budget_prune`, applied as a final stage after the shipped chain
(`inc/g2sp6`). Every arm re-solves nothing — one ILP solve per dataset, then post-processing.

```
isolated          drop nodes in no edge at all. Ignores the budget; should be free.
geometry @ f      rank by span, tie-break on median step length. Keep until N_est * f.
length   @ f      span alone -- the ablation that says whether tightness carries anything.
f in {1.0, 0.9, 0.8, 0.7}
```

`N_est` is each dataset's own `estimated_number_of_nodes`, read from its `.geff`.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** The `none` arm equals `claude_divsweep`'s `inc/g2sp6` — total 0.9188,
   `div_J` 0.1154, 1,443 forks. Otherwise the chain has moved and nothing else is readable.
2. **`isolated` is non-negative on both embryos.** An edgeless node cannot contribute a TP
   edge, so dropping it should be free. If this *loses*, my reading of the metric is wrong
   and predictions 3-5 mean nothing.
3. **Some budget arm beats `none` by more than 0.0015** (`notes/44`'s floor). This is the
   crux. Failing it closes the last cheap direction and leaves only the detector.
4. **`geometry` beats `length` at the same factor.** If tightness carries nothing, r35's
   ranking is just "keep long tracks" and the mechanism is simpler than advertised.
5. **The best arm holds in sign on BOTH embryos.** `notes/49`: the test set is a third pair
   of embryos, and a pooled win across crops of two is not evidence about a third. This is
   the prediction that 0.901 -> 0.863 was missing.

*Node counts and `total_node_ratio` are reported per arm regardless, so even a clean failure
tells us where the budget actually sits — which no run so far has measured directly.*
