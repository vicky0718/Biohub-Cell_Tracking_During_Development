# Arm log — running record, updated as each run lands

## SCORES (2026-09-06/07)

```
arm      LB       vs lb941   what it says
lb941    0.941     baseline  reproduced its public claim EXACTLY -- no offset, unlike
                             claude_fork's -0.001 against a claimed 0.938
sew20    0.942      +0.001   SECONDARY_EDGE_WEIGHT 0.15 -> 0.20
sew25    0.942      +0.001   0.25 -- identical
sew30    0.942      +0.001   0.30 -- identical. The axis is a PLATEAU, not a slope.
union    0.941       0.000   +3,333 nodes bought nothing
div15    0.938      -0.003   +48 divisions cost real score
dc40     0.933      -0.008   -62 divisions. The BIGGEST loss of any arm.
det960   0.942      +0.001   DET_THRESHOLD 0.96 (scored by busyaprime, not by us)
sewdet   0.942      +0.001   SEW 0.20 AND DET 0.96 together -- NOT additive
det955   0.941       0.000   DET 0.955, one step further, gives the gain back
```

**`notes/71`'s node-count hypothesis is falsified.** `union` made the largest output change
of any arm — 2.8% more nodes, 3.1% more edges — and moved the score by exactly zero. The
pre-registered reading was "if the hypothesis holds, the UP arms cluster above `lb941` and
`union` moves most". It moved least. `notes/66` §3b's `claude_submit_topk` result (−14.9%
nodes, −0.038) therefore says something about *that* prune specifically, not about node
count as an axis.

That demotes `dse44` (+1,433), `dse52` (−1,829) and `det960` (+404) together: all three are
node-count arms, and node count has now been measured as not the axis.

**The SEW axis is a plateau and it is exhausted.**

```
SEW 0.15   0.941        SEW 0.25   0.942
SEW 0.20   0.942        SEW 0.30   0.942
```

It saturates at the first step. Not "more secondary weight is better" — "the secondary model
needs to be on at some minimum level, and 0.20 already is". `sew25` and `sew30` bought
nothing over `sew20`, so there is nothing further along this axis and the pre-registered
"run two so it shows a slope rather than a single point" did its job: the slope is flat.

**The public frontier reached 0.942 — and one of the two notebooks that got there is our own
shelved arm.** `busyaprime/biohub-0-942-lb-one-knob-past-the-public-line` is `lb941` with
`DET_THRESHOLD 0.965 -> 0.96`: byte-for-byte what `det960` does. We built it, verified it
(+404 nodes), and shelved it when `union` falsified the node-count hypothesis. Somebody else
submitted it and scored 0.942.

That is a real cost of the demotion, and the lesson is narrow rather than "should have
submitted everything": `det960` was demoted for belonging to a *hypothesis class* that had
just been falsified, when what the falsification actually showed was that node count does not
predict score — it said nothing about whether the detection threshold does.

**So there are now two independently-confirmed +0.001 knobs on different stages, and nobody
has combined them.** `sewdet` (SEW 0.20 + DET 0.96) is running. If the mechanisms are
independent it is worth +0.002 and lands on 0.943, which is rank 100. If it stays at 0.942
the plateau belongs to the pipeline rather than to either knob.

The other public 0.942, `analyticaobscura/biohub-lb-942`, is a different animal: it adds
`PPSWEEP_MAX_ADJ_LOSS`, `PPSWEEP_SELECT_MARGIN` and `VALIDATOR_N_PER_TYPE`, and its axis
reads *"holdout-selected post-process configuration"* — an automated post-processing sweep
inside the notebook, not a knob.

**Divisions do NOT have a sign — they have a sharp optimum, and `lb941` is sitting on it.**

```
div15   +48 divisions   0.938   -0.003
lb941    94 divisions   0.941    ----
dc40    -62 divisions   0.933   -0.008
```

Both directions lose, and *removing* them loses more than twice as much as adding them. The
pre-registered reading was "if `dc40` gains, division count wants to go down". It did not
gain; it lost harder than anything else we have run. **Stop moving divisions.**

Two things follow, and the second is more important than the first.

**The cost cannot be the division term.** `division_jaccard` contributes at most `0.1 × 1.0`
and realistically about 0.006 at the fork's `div_J ≈ 0.0625`. An 0.008 loss therefore has to
be landing on `edge_J`: removing a fork does not just delete the fork, it orphans a branch
and fragments a track, and fragmentation costs edges everywhere downstream.

**The verification clips do not predict graded magnitude.** `dc40`'s counters on the four
clips were −4 nodes and −65 edges — 0.06% of the output — and the graded score moved 0.008.
`notes/66` §1 established the clips are placeholders; this quantifies how badly they
under-report. `tools/verify_arm.py` remains valid for its actual job, which is *did the edit
land and in which direction*, and must never be read as *how big will this be*.

This also retires my own tie-break: `dc40` was ranked above `dse44` partly because
`notes/66` §3b said cutting nodes costs score. `dc40` cut almost no nodes and lost 0.008
anyway, so that reasoning was wrong about the mechanism as well as the size.

Board 2026-09-07 08:05: 3,201 teams, rank 100 needs **0.943**. **Our 0.942 is rank 156**,
not 112 — I reported the optimistic end of a tie band and that was wrong. 0.942 spans ranks
112-218 and Kaggle breaks the tie by submission time, so a late submitter lands at the
bottom of the band. **Read the `Rank` column from the leaderboard CSV; never compute rank as
`count(score > ours) + 1`.**


Counters are `run_stats.csv` summed over the four verification clips, against `claude-arm-lb941`
(119,279 nodes / 115,009 edges / 94 divisions). **Verification-mode output, not graded
output** (`notes/66` §1) — it says whether an edit reached the pipeline and in which
direction, never what it scores.

| arm | base | change | nodes | edges | divisions | run |
|---|---|---|---:|---:|---:|---|
| `lb941` | analyticaobscura 0.941 | none | 119,279 | 115,009 | 94 | complete |
| `adaptive` | rishabhr0y 0.941 | none | **+3,425** | +3,421 | +64 | complete |
| `union` | lb941 | `LINK_MODE → adaptive` | **+3,333** | +3,270 | +1 | complete |
| `sew20` | lb941 | `SEW 0.15 → 0.20` | +101 | +95 | +1 | complete |
| `div15` | lb941 | `DIVERGE_UM 2.25 → 1.5` | +30 | +71 | **+48** | complete |
| `dc40` | lb941 | `DC_SAFE_DIV 0.25 → 0.40` | −4 | −65 | **−62** | complete |
| `ckpt948` | zhuzhenghao 0.948 | none (change inert) | +147 | +256 | — | complete, not submitted |
| `dse44` | lb941 | `DUAL_SEED_EDGE 0.48 → 0.44` | **+1,433** | +1,441 | +9 | complete |
| `dse52` | lb941 | `DUAL_SEED_EDGE 0.48 → 0.52` | **−1,829** | −1,778 | −3 | complete |
| `det960` | lb941 | `DET_THRESHOLD 0.965 → 0.960` | +404 | +384 | 0 | complete |
| `dcgap40` | lb941 | `DC_GAP 0.25 → 0.40` | −48 | −54 | +2 | complete — **not worth a slot** |
| `gap44` | lb941 | `GAP_CLOSE_UM 5.0 → 4.4` | −42 | −42 | +1 | complete — **not worth a slot** |
| `sew25` | lb941 | `SEW 0.15 → 0.25` | +300 | +283 | +1 | complete |
| `sew30` | lb941 | `SEW 0.15 → 0.30` | +474 | +453 | +1 | complete |
| `sewdet` | lb941 | `SEW 0.20` **+** `DET 0.96` | **+505** | +477 | +1 | complete |
| `det955` | lb941 | `DET_THRESHOLD 0.965 → 0.955` | **+697** | +665 | 0 | complete |
| `sew948` | rishabhr0y 0.948 | none | — | — | — | **dead** — self-inconsistent checksum guard (`notes/70` §3) |

## What `adaptive` adds to the picture

It is an unmodified 0.941 in its own right, and its counters explain `union`:

```
lb941      119,279 nodes    94 divisions
adaptive   122,704 nodes   158 divisions     the other public 0.941
union      122,612 nodes    95 divisions     lb941's divisions + adaptive's linker
```

`union`'s +3,333 nodes are **almost entirely the `adaptive` link mode**, which is not an
untested change — it is a configuration that scores 0.941 on its own. That strengthens the
case for ranking `union` first: its large move is a proven one, and what it adds on top is
`lb941`'s division tuning, which is also proven. The cross is between two knowns.

It also shows `adaptive` reaches 158 divisions by a different route than `div15`'s 142 — so
those two arms are partly redundant, and if `div15` pays, `adaptive`'s division count is
part of why the second 0.941 exists at all.

**Not spending a slot on `adaptive` itself.** It claims the same 0.941 as `lb941`, so
`lb941`'s measured score is the control `union` needs, and a slot spent confirming a second
route to a number we already have buys nothing. Revisit only if `union` surprises.

## `dse44`: a large effect from a constant nobody has moved

`DUAL_SEED_EDGE_THRESHOLD` is `0.48` in **all 56** public kernels mined — not swept, not
stepped, inherited. Loosening it to 0.44 moves more than any arm except the two that change
the link mode outright:

```
raw_edges          114,840 -> 116,531   +1,691
edges              115,009 -> 116,450   +1,441
nodes              119,279 -> 120,712   +1,433
```

So `notes/65` §3's "never moved" category is not empty of consequence: the single largest
untouched knob does 1.2% of the node set. Whether that is worth score is a different
question, and `dse52` — the same constant moved the same distance the other way — is running
to answer whether 0.48 is an optimum somebody found and never wrote down.

**Evidence ranking is unchanged by this.** `dse44` has a large mechanical effect and *no*
external support, where `union` has two independently-scored 0.941 halves and `sew20` has
three kernels naming its parameter. It slots below both. It goes above `dc40` on the one
piece of graded evidence available (`notes/66` §3b: cutting nodes cost 0.038), which is a
tie-break between two arms of equal external standing, not the node-count screen `notes/71`
refuses to build.

## The `dse` pair straddles 0.48, symmetrically and largely

```
dse44   DUAL_SEED_EDGE 0.48 -> 0.44    nodes +1,433   edges +1,441   raw_edges +1,691
lb941                       0.48       nodes       0   edges       0
dse52   DUAL_SEED_EDGE 0.48 -> 0.52    nodes -1,829   edges -1,778   raw_edges -2,072
```

A constant identical in all 56 mined kernels, moved ±0.04, moves ±1.5% of the node set in
each direction. Whatever the leaderboard says about this pair is the cleanest single result
available from the whole queue: it tests `notes/71`'s node-count hypothesis *and*
`notes/65` §3's never-moved category at once, with a symmetric control.

Three outcomes and what each means:

* **`dse44` up, `dse52` down** — node count is the axis, and the remaining search should go
  to whatever else adds nodes.
* **Both down** — 0.48 is a real optimum somebody found and did not write down, and the
  never-moved category is worth much less than `notes/65` §3 argued.
* **`dse52` up, `dse44` down** — the node-count hypothesis is backwards, and `notes/66`
  §3b's reading of `claude_submit_topk` needs revisiting.

`det960` moves the same direction as `dse44` by a quarter as much (+404 nodes), so it is a
weak second vote on the same axis rather than an independent question.

## `gap44` is effectively a no-op — the gap radius has bottomed out

The guard fix worked (`resolved 4.4 OK`), and then the arm did almost nothing:

```
gap_candidates          856 -> 739      -117   the pool shrinks as expected
gap_pairs_selected      623 -> 615        -8   the SELECTION barely moves
gap_added_nodes         623 -> 615        -8
nodes               119,279 -> 119,237    -42   0.035% of the node set
```

Narrowing the radius removes 117 candidates and 8 selected pairs, so **selection was not
radius-limited at 5.0**. Something downstream — the density-adaptive step, the DeepCenter
gap veto, or the motion bypass — is choosing which gaps to close well inside 5.0 µm, and the
radius is no longer the binding constraint.

That is `notes/60`'s trap in its honest form: not a cap silently binding, but a parameter
that has simply run out of room. The public 5.8 → 5.0 step was claimed worth +0.001; 5.0 →
4.4 cannot be worth much of anything, because there is almost no output to change.

**Not submitting it.** A 0.035% output change cannot separate itself from run-to-run noise
on the leaderboard, and a slot spent on it buys a number we could not interpret. This is
the first arm the verifier has taken *off* the list rather than confirmed onto it, which is
what it was built for.

## `dcgap40` confirms its own demotion, and the queue is exhausted

`notes/69` §2 demoted this arm on the counters before running it, arguing the DeepCenter gap
gate sees too little traffic to matter. Run, it says exactly that:

```
deepcenter_gap_checked                94 ->  94     the population, unchanged
deepcenter_gap_accepted               17 ->   1     -16   everything it could remove
deepcenter_gap_bypassed_strong_motion 606 -> 606    never reaches the gate at all
nodes                            119,279 -> 119,231  -48   0.04% of the node set
```

Sixteen gap closures across four clips. The prediction and the measurement agree, which is
the useful part — `notes/69`'s reading of `run_stats.csv` was load-bearing and it held.

## Final tally

**Worth a submission slot, ranked:**

```
1  lb941     the public 0.941 reproduced             floor, +0.004 over our 0.937
2  union     LINK_MODE -> adaptive                   +3,333 nodes; crosses two proven 0.941s
3  sew20     SEW 0.15 -> 0.20                        +101; three kernels name this parameter
4  div15     DIVERGE_UM 2.25 -> 1.5                  +48 divisions; the dominant gate
5  dc40      DC_SAFE_DIV 0.25 -> 0.40                -62 divisions; author's own f1 peak
6  dse44     DUAL_SEED_EDGE 0.48 -> 0.44             +1,433; a constant nobody has moved
7  dse52     DUAL_SEED_EDGE 0.48 -> 0.52             -1,829; the mirror of 6
8  det960    DET_THRESHOLD 0.965 -> 0.960            +404; one step past the public step
```

**Rejected, with the measurement that rejected each:**

```
gap44     -42 nodes (0.035%)   selection was never radius-limited at 5.0
dcgap40   -48 nodes (0.04%)    the gate admits 17 of 94 checks; 16 is the whole ceiling
adaptive  a second route to the same claimed 0.941; lb941 is the control
ckpt948   its distinguishing change is inert (notes/68 §2)
sew948    dies in its own checksum guard; the config contradicts itself (notes/70 §3)
```

**The queue is empty.** Eleven arms built, nine ran, two failed for reasons now understood.
Nothing further is worth building without a leaderboard result: the ranking above is by
external evidence and mechanism, and the first score — `lb941`'s especially, which calibrates
our reproduction offset against its claim — is what would tell us which axis to spend the
next runs on.

## Runtime is homogeneous across arms — no arm is individually at risk

Wall-clock from push to complete, every arm on a P100 with the wheelhouse install included:

```
ckpt948  39   div15   38   dse44  39   gap44   39
dc40     42   adaptive 41   dse52  41   det960  41
lb941    51   dcgap40  50
```

38-51 minutes, no outliers — `union`'s extra 2.8% of nodes does not show up as extra time,
and neither does `div15`'s 51% more divisions. So `notes/66` §4's timeout worry, if it
bites, bites every arm equally and cannot be used to rank them.

Bounding the graded rerun is less clean than it looks. `run_stats.csv` reports
`predict_minutes_total = 10.1` for the four clips, and the graded set is ~17× (`notes/66`
§3b). If only prediction scales, the rerun is ~3.3 h. If everything scales, it is
40 × 17 ≈ 11.3 h — the pessimistic figure already recorded. The truth is between, because
setup (wheelhouse install, model loading, repo staging) is fixed while the ILP and the
repair chain are per-clip and are not separately timed.

**The reassurance is empirical, not arithmetic:** `claude_fork` and `claude_forkw085` both
completed graded reruns and returned scores. Both drew T4s, and every arm here drew a P100,
which is slower — so the risk is real but bounded by two successful reruns of the same
pipeline shape. Nothing to do about it in advance; it would show up as a failed submission
rather than a wrong number.

## The SEW ramp is clean and monotone

```
SEW     nodes    edges   raw_edges     LB
0.15        0        0           0   0.941   (lb941)
0.20     +101      +95        +135   0.942   +0.001
0.25     +300     +283        +308   ?
0.30     +474     +453        +496   ?
```

Each step admits more secondary-model edges, monotonically, with no cap intervening — the
parameter behaves exactly as its name says and the effect size grows smoothly. That is the
shape a real gradient has, and 0.20 is where every public notebook stops.

Note the node counts are small: `sew30` adds 474 nodes, an eighth of what `union` added for
zero gain. So whatever `sew20`'s +0.001 came from, it was not volume — which is consistent
with `notes/71` being falsified and makes this axis interesting for a different reason than
the one the arms were originally sorted by.

## `sewdet`'s node count is exactly additive

```
sew20    SEW 0.20              +101 nodes
det960   DET 0.96              +404 nodes
sewdet   SEW 0.20 + DET 0.96   +505 nodes      = 101 + 404, exactly
```

Both edits landed (`DET_THRESHOLD` resolves to 0.96 in the dump; `SECONDARY_EDGE_WEIGHT` is
one of the keys the resolved block omits, so the arithmetic is the evidence). And the two
knobs do not interact *at the output level* — they touch disjoint sets of nodes.

That is encouraging but not decisive for the score. Disjoint outputs make additive score
plausible; they do not guarantee it, because both arms scored 0.942 alone and 0.942 may be a
plateau of the scoring surface rather than a sum of two independent contributions. `union`
already showed a big output change buying nothing.

`det955` continues the detection gradient one step past the newly-public 0.96: +697 nodes
against `det960`'s +404, monotone.

## 0.942 is a ceiling, and `sewdet` is what proves it

```
lb941    0.941
sew20    0.942   sew25 0.942   sew30 0.942     the SEW axis, saturated at step one
det960   0.942                                 the detection threshold
sewdet   0.942   SEW 0.20 + DET 0.96           both together, still 0.942
det955   0.941   DET 0.955                     one step further loses the gain
```

`sewdet` is decisive. Its node count was exactly `sew20 + det960` (+505 = 101 + 404), so both
edits landed and they move disjoint nodes — and the score did not budge. **Two independent
+0.001 knobs that do not add are not two contributions; they are two ways onto the same
shelf.** `det955` hands the gain straight back, so the detection threshold has an *optimum* at
0.96 rather than a direction.

Eleven arms, four distinct axes, one shelf at 0.942 with cliffs below it. That is Tang's
(rank 5, 0.961) statement measured: *"the current ckpt has kind of hit a wall, it's hard to
get more gain from post-processing alone."* Parameter tuning on this checkpoint is finished.

## The way off a plateau is a different kind of change

`analyticaobscura/biohub-lb-942` is one, and it was hiding in plain sight. Its axis reads
*"public 0.939 base + holdout-selected post-process configuration"* and it sets three
variables no other public notebook touches — `VALIDATOR_N_PER_TYPE`, `PPSWEEP_SELECT_MARGIN`,
`PPSWEEP_MAX_ADJ_LOSS`. Those strings appear **only in its config cell**, so the sweep itself
lives inside the shipped pipeline: the support pack already implements a post-processing
search with holdout selection, and nobody except this author has switched it on.

That is categorically different from every arm above. A fixed knob applies one value to every
movie; a holdout-selected sweep chooses a **different post-processing configuration per
dataset**. A plateau in the fixed-knob family says nothing about the adaptive one.

Three arms running: `pp942` (unmodified, to establish the mechanism's score for us),
`pp942sew` (the sweep crossed with our confirmed knob — different families, unlike `sewdet`),
and `pp942n8` (`VALIDATOR_N_PER_TYPE` 4 → 8, widening the evidence the selection rests on).
`pp942n8` carries a runtime risk: `notes/69` §3 puts the graded rerun near 11 h against a
12 h limit, and doubling the validation set adds to it.
