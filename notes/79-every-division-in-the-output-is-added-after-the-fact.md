# Every division in the output is added after the fact — the ILP's are discarded wholesale

`ilpdiv04` (ILP division penalty 1.2 → 0.4) came back **2/3/9**. `division_tp` is still 2.

```
arm          cands  added |   edge_J      adj   tp/fp/fn    div_J     score
ttasec         975    194 |  0.93281  0.93623      2/2/9  0.15385  0.95161
divloose       644    155 |  0.93281  0.93623      2/2/9  0.15385  0.95161
dcloose        975    245 |  0.93262  0.93603      2/4/9  0.13333  0.94936
ilpdiv04       984    200 |  0.93242  0.93577      2/3/9  0.14286  0.95006
```

Cutting the ILP's division penalty by two-thirds moved the geometric candidate count by nine
(975 → 984). That is not a lever being pulled; that is a lever that is not connected.

## 1. Why — and it is structural, measured two ways

`filter_output_graph` runs this before anything else touches the graph:

```python
motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
if motion_edges:
    stats['motion_relink_replaced_raw_edges'] = len(edges)
    edges = motion_edges          # the entire ILP edge set, discarded
```

and `motion_relink_edges` assigns with

```python
row_ind, col_ind = linear_sum_assignment(cost)
```

A linear sum assignment is **one-to-one**. It cannot emit a second child for any source. So
the relink stage replaces the ILP's graph with one that has, by construction, **no divisions
at all** — and it is on by default (`BIOHUB_OUTPUT_MOTION_RELINK` defaults to `'1'`).

The arithmetic confirms it exactly:

```
forks in ttasec's submission.csv                      194
safe divisions added by add_safe_divisions_postlink   194
divisions surviving from the ILP stage                  0
```

**Every division in the output is inserted after the fact by the repair step.** The ILP's
division decisions never reach the submission, which is why `ILP_DIVISION_WEIGHT` is inert —
`notes/68`'s failure mode, found this time by measurement before a submission was spent.

(`OUTPUT_SINGLE_CHILD_REPAIR`, which *would* also flatten every fork, is defaulted **off**
(`'0'`) and never set. I read it as the culprit first and was wrong; the relink is the culprit.)

`ilpdiv00` (penalty → 0.0) is still running. Its prediction is now fixed in advance: **no change
in `division_tp` and no change in fork count**. If it moves either, this section is wrong.

## 2. Three stages, three closed doors

```
stage                         lever                      status
ILP solve                     ILP_DIVISION_WEIGHT        INERT  -- output discarded by relink
motion relink                 (1-to-1 assignment)        emits zero divisions by construction
post-link repair gates        veto / divergence / geom   CANNOT REACH already-linked daughters
```

The repair step is the only source of divisions, and `notes/78` measured its ceiling: moved
ninety divisions apart in both directions, `division_tp` stayed at 2 every time, because
its candidates must have **no incoming edge**.

## 3. The one door left

`OUTPUT_MOTION_RELINK = 0` keeps the ILP's edges instead of replacing them — the only
configuration in which an ILP division can reach the output. It is unguarded by
`_EXPECTED_NUMERIC`, but it is not a small change: the relink is part of what the lineage
credits for 0.941, so the edge term may pay for it.

That makes the eval's two-sided read essential, and it is the same read as before:

* does `division_tp` rise above 2 — the first time any lever could move it at all, and
* does `edge_jaccard` survive.

Two arms: `norelink` (relink off alone, to price it) and `norelinkdiv` (relink off **and**
division penalty 1.2 → 0.4, so the ILP both makes divisions and keeps them). If
`division_tp` does not move with the relink off and the penalty down, then nothing reachable
in this pipeline produces the nine missing divisions, and the direction closes on measurement.
