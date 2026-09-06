# The queue is one experiment, and it is about node count

Written before the scores land, so it cannot be fitted to them.

## The hypothesis

`notes/66` §3b priced pruning on graded data for the first time: `claude_submit_topk` wrote
**14.9% fewer rows** than the configuration it came from and scored **0.863 against 0.901**.
The metric's node-budget factor can only account for about +0.014 of that, so `edge_J`
itself must have fallen by roughly 0.05. **Node count and `edge_J` are steeply coupled, and
the coupling is worth far more than the budget term.**

The budget term also turns out to be nearly exhausted in the other direction. On the
placeholder clips the base sits 11.9% under `N_est` (`notes/66` §2), collecting a bonus of
about +0.011 at `edge_J` 0.92. There is not much more of it to collect.

Put together: **if the gradient is that steep going down, the interesting direction is up.**
Adding nodes costs a little of an almost-exhausted bonus and buys recall on a term that
carries ten times the weight.

That is not a certainty — `topk`'s prune was aggressive and selective, and the local slope
at −15% need not be the slope at +3% — but it is the reading that makes the queue coherent
rather than a list of knobs.

## What each running arm does to node count

```
arm       change                             nodes vs lb941    direction
union     LINK_MODE -> adaptive                     +3,333     UP   (+2.8%)
det960    DET_THRESHOLD 0.965 -> 0.960              expect +   UP   (lower bar to detect)
dse44     DUAL_SEED_EDGE 0.48 -> 0.44               expect +   UP   (edges, not nodes)
sew20     SEW 0.15 -> 0.20                            +101     UP   (+0.08%)
div15     DIVERGE_UM 2.25 -> 1.5                       +30     UP   (via +48 divisions)
dc40      DC_SAFE_DIV 0.25 -> 0.40                       -4     DOWN (-62 divisions)
dse52     DUAL_SEED_EDGE 0.48 -> 0.52               expect -   DOWN
gap44     GAP_CLOSE_UM 5.0 -> 4.4                   expect -   DOWN
dcgap40   DC_GAP 0.25 -> 0.40                       expect -   DOWN
```

Four up, four down, one negligible. **So the queue is already a signed experiment**, and the
results should be read as a set: if the hypothesis holds, the UP arms cluster above `lb941`
and the DOWN arms below it, and `union` — the largest move in either direction — moves most.

If instead the signs are mixed with no relation to direction, the hypothesis is wrong and
each arm is only worth its own parameter's story. Recording that in advance so the answer
cannot be read as confirmation either way.

## The one that would falsify it cleanly

`dse44` and `dse52` move the same constant in opposite directions by the same amount, from a
value (`DUAL_SEED_EDGE_THRESHOLD` 0.48) that is identical in all 56 public kernels mined and
has been moved by nobody. If node count is what matters, they should straddle `lb941` with
opposite signs. If they both lose, 0.48 is a real optimum somebody found and never wrote
down, and `notes/65` §3's "never moved" category is worth less than it looks.

## What this does not license

Choosing a submission by predicted node count. `notes/64` and `notes/66` between them killed
three screens, and a fourth built out of an unfalsified hypothesis would be the same mistake
in a new costume. The order handed over stays evidence-first — `union` above `sew20` because
two configurations that each score 0.941 is stronger than one parameter three kernels share,
not because `union` adds more nodes.
