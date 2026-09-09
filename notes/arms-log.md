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

## Which of four tied 0.942 arms to select

`sew20`, `sew25`, `sew30` and `sewdet` all score **exactly 0.942**. Kaggle reports three
decimals and `privateScore` is empty until the competition closes, so there is no finer
resolution available — none of them is "slightly lower" than another. Anything that looks
like a difference is tie ordering on the leaderboard, not score.

The public leaderboard is **29% of the test data** (`MEMORY.md`), so the final standing is
decided on a different ~71%. Four arms tied on the public slice will not be tied on the
private one, and selection is a bet on which generalises.

**The shape of the surface around each choice is the argument.**

```
SEW   0.20 -> 0.942   0.25 -> 0.942   0.30 -> 0.942      a PLATEAU, three points wide
DET  0.965 -> 0.941   0.96 -> 0.942  0.955 -> 0.941      a PEAK, one point wide
```

A plateau means the score is insensitive to the exact value; a peak means it is sensitive.
When the evaluation data changes, a peak is far more likely to slide off its optimum than a
flat region is to fall off its edge. **`sewdet` inherits the peak** — it is `SEW 0.20` *and*
`DET 0.96`, and the DET half is the fragile one. It also makes two changes for the same
measured score as one, which is more surface to be accidentally fitted to the public 29%.

**The fair counter-argument:** `sewdet`'s two halves are each corroborated by other people's
public scores — `SEW 0.20` by three 0.948-claiming kernels, `DET 0.96` by `busyaprime`'s
public 0.942. It is not two guesses stacked. If the private split happens to reward detection
recall more than the public one does, `sewdet` could edge ahead. This is a preference between
indistinguishable options, not a demonstration.

**Selection: `sew25`** — the centre of the measured plateau, furthest from the 0.15 edge where
the score drops to 0.941, and a single change from the base. If more than one final submission
is allowed, pair it with `sewdet` to hedge across the two families.

*Not asserted:* that Kaggle's tie ordering favours the earlier submission. It is the
documented behaviour, but `LastSubmissionDate` in the leaderboard CSV is the team's last
submission rather than the one that produced their best score, so this data neither confirms
nor contradicts it.

## The board is moving much faster than we are

```
2026-09-07 08:05   rank 100 needs 0.943   our 0.942 = rank 156
2026-09-08 06:30   rank 100 needs 0.946   our 0.942 = rank 287
```

The 0.942 band is now **138 teams wide** (ranks 269-406). Rank 100 moved +0.003 in a day
while we moved 0. The gap is no longer one thousandth — it is **four**, and no knob on this
checkpoint produces that.

## The PPSWEEP arms produce identical output — not worth a slot

```
pp942     vs lb941   nodes +0   edges +0   divisions +0   short_track +0   IDENTICAL
pp942sew  vs lb941   nodes +101 edges +95  raw_edges +135                  = sew20 exactly
```

`pp942` is byte-identical to `lb941` on every counter, and `pp942sew`'s deltas are exactly
`sew20`'s. So the holdout-selected post-processing sweep **changed nothing** on the four
verification clips, and `pp942sew` is `sew20` wearing a different notebook.

It did not fail to run: both arms took **~80 minutes** against 38-51 for every knob arm, so
the search executed. The reading that fits is that the sweep evaluated its candidates on
these clips and **selected the default configuration** — which is what an adaptive method
does when the base config is already the best available for that data.

That is not evidence it would select the same way on the graded set, and per-dataset
adaptation is precisely the thing that cannot be checked from placeholder clips. But it is
also not evidence that it would do anything, and the author's own public score with this
mechanism is **0.942** — which we already hold from `sew20`.

**Neither arm is worth a slot.** Same score class as what we have, no measurable difference
on anything visible, and the ~80-minute verification runtime is a warning on top: the graded
rerun is ~17x that work against a 12 h ceiling, and a search loop scales with it.

That leaves `tta946` (running) as the only thing on the table that spans the gap to 0.946.

## `tta946` lands: the 0.946 reproduces, and the gain is code

`claude-arm-tta946` completed 2026-09-08 11:18 on a P100 (wheelhouse fired, `rc=0`).
Unmodified fork of `reyhanksatria/biohub-cell-tracking-0-946-lb`.

```
experiment_tag       edge_feature_tta_0946
EDGE_TTA_ACTIVE      views = 8   mean_abs_feat_delta = 0.323
submission.csv       241,282 rows
nodes / edges        122,791 / 118,491      (lb941: 119,279 / 115,009 — +3,512 / +3,482)
```

**Submit it.** It is +0.004 over everything we hold and the mechanism is confirmed active in
the log rather than inferred from a config dump.

Two things came out of reading it, both in `notes/74`:

1. **The "400-epoch snapshot" in its header is nothing.** `lb941`, `ckpt948` and `tta946` all
   materialise the same primary SHA `12f6881e…` from the same `-50ep-v1` mount; the
   `400ep-snapshot` string is the `name` field inside pilkwang's manifest. The dataset is
   misnamed. Everyone in this lineage runs the same weights, so **the +0.005 is code**.
2. **The mechanism is half-finished.** The eight-view D4 ensemble discards seven of its eight
   *association feature* maps; the 0.946 keeps them for the primary model. The secondary
   model — which runs the same eight encodes at `SECONDARY_DETECTION_WEIGHT 0.80` — still
   discards all eight. `ttasec` finishes it. Free at runtime, and it raises rather than
   silently reproducing 0.946 if the features do not move.

## The rank arithmetic, corrected again

`notes/71` said read the `Rank` column. There is a second half: **the column tells you where
the teams already there sit, not where you would land.** Ties break by submission time, so a
new entrant joins at the *bottom* of its band.

```
2026-09-08 11:23Z, 3,244 teams
score    better   tied      band      we would land at
0.947        31      9     32-40                    41
0.946        40    140    41-180                   181
0.945       180     20   181-200                   201
0.942       279    135   280-414                   415   <- where we are
```

So `tta946` at 0.946 buys **~rank 181**, not rank 100 — a real +106 places, and I had told
you 0.946 *was* rank 100, which is what the column says and not what we would get.

**The top 100 costs 0.947.** And 0.947 is nearly empty (9 teams) precisely because 0.946 is
where the public notebook lives. One thousandth, from something the public notebook does not
already do.

## In flight

```
ttasec     extend the edge-feature TTA to the secondary model   pushed 11:30, running
pp942n8    VALIDATOR_N_PER_TYPE 4 -> 8                          running since 09:56
ttasew20   SEW 0.15 -> 0.20 on the 0.946 base                   built, waiting for a slot
```

## `ttasec` fired, and it re-routed 1% of the graph

`claude-arm-ttasec` completed 2026-09-08 13:11 (P100, wheelhouse). The patch is live on every
frame pair:

```
EDGE_TTA_ACTIVE      views = 8  mean_abs_feat_delta = 0.323   (primary, the author's)
SEC_EDGE_TTA_ACTIVE  views = 8  mean_abs_feat_delta = 0.231   (secondary, ours)   x792
```

The secondary model's features moved by 71% of what the primary's did. Against `tta946`:

```
                nodes      edges     forks
tta946        122,791    118,491    28,600
ttasec        122,735    118,435    28,582
net               -56        -56       -18

churn:   nodes  542 dropped / 486 added      edges  1,234 dropped / 1,178 added
```

**The net counts are the wrong number to read.** A 56-row delta looks like a rounding error;
the actual change is **1,234 edges out and 1,178 in — ~1% of the edge set re-routed** — with
the churn nearly cancelling. This is exactly the case `diff_arms` reports added and dropped
separately for, and `notes/60`'s no-op trap read from the other side: an arm can look inert
in the totals while having changed a percent of the answer.

`run_stats` agrees, loudly: **101 of 284 counters changed**, across every stage —
`raw_nodes` (detection), `motion_relink_*`, `gap_density_*`, `deepcenter_safe_div_*`,
`short_track_*`. Not a cosmetic change confined to one gate.

**Submit it.** Real change, right theory, unknown sign — and the sign is not knowable here,
because `notes/66` closed local scoring and `notes/72` §3 closed local CV.

*One correction to what I claimed when building it:* I said the patch was free at runtime.
The eight forward passes were already happening, but the feature-map arithmetic is not free —
`predict_minutes_total` went 9.30 -> 10.67 (+15%). On the graded set that is ~158 -> ~181
minutes against a 720-minute limit, so it does not threaten anything, but "free" was wrong.

## `ttasecw20` follows from the result

The secondary's features improved by 0.231 against the primary's 0.323, and bought a much
smaller downstream change — because the secondary enters at `SECONDARY_EDGE_WEIGHT 0.15`
behind the low-margin consensus gate. **A weight tuned for single-view features is the wrong
weight for eight-view ones.** `ttasecw20` is `ttasec` + SEW 0.20, pushed.

Unlike `sewdet` — two changes with no separate score for either half — this one decomposes:
`ttasec` is being scored on its own, so `ttasecw20 - ttasec` isolates the weight.

```
ttasec      pushed, complete, SUBMIT       0.946 base + secondary edge-feature TTA
ttasecw20   pushed, running                the same, plus SEW 0.20
pp942n8     running since 09:56            VALIDATOR_N_PER_TYPE 4 -> 8
ttasew20    built, spare                   SEW 0.20 on tta946 without ttasec
```

## `ttasecw20` lands, and the two changes are not independent

Both edits verified in the log: `edge weight = 0.200` and `SEC_EDGE_TTA_ACTIVE` ×792.

```
                nodes      edges     forks
tta946        122,791    118,491    28,600
ttasec        122,735    118,435    28,582
ttasecw20     122,825    118,522    28,610

tta946    -> ttasec       edges  -1,234  +1,178
ttasec    -> ttasecw20    edges  -1,175  +1,262
tta946    -> ttasecw20    edges  -1,633  +1,664
```

If the two changes were independent the last line would show ~2,400 edges dropped. It shows
1,633 — **about 776 of the edges `ttasec` removed are put back by the weight increase.**

So the "different families" reasoning that justified pairing them was wrong. `SECONDARY_EDGE_WEIGHT`
is the *gain* on exactly the pathway the secondary TTA changes the *signal* of; they are one
axis, not two. That does not spoil the experiment — `ttasec` has its own score, so
`ttasecw20 - ttasec` still reads as "raise the gain on the improved signal" — but it is not
the orthogonal stack `sewdet` was, and it should not be described as one.

## The fusion is switched off, and nobody noticed

Reading `tta946`'s log for something else turned up 65 `BIOHUB_RETENTION_GUARD` records.
Every one ends `"use_primary": true` — the dual-seed detection fusion computed, then discarded:

```
44b6_0b24845f   64 frames rejected   median retention 0.842
6bba_05b6850b    1 frame  rejected                    0.892
retention   min 0.453   median 0.842   MAX 0.899   floor 0.90
```

**The highest retention ever observed is 0.899 against a floor of 0.90.** On the frames the
guard evaluates it has never once passed, so `SECONDARY_DETECTION_WEIGHT = 0.80` — the
parameter the entire 0.936 cluster is built on, and the one `notes/64` watched regress when
pushed to 0.85 — is *inert* on those frames. Nobody in the public lineage has looked at this
counter; the guard only writes a line when it rejects, which is why it reads as silence.

`ttaret85` drops the floor to 0.85: it flips the rejections nearest to passing (0.85-0.899),
where the blend discards least, and leaves the guard standing for the frames where retention
collapses to 0.45. Soheil (rank 2) named missing endpoint nodes as his dominant error, so
letting 20% of candidates go on a bad frame is a real risk this arm declines to take.

Three edits — the env var is assigned twice and the later one wins, and the value is guarded
as text, so `_EXPECTED_TEXT` moves too. The run prints
`Frozen frame retention guard applied at 0.85`, so it cannot be silently inert.

```
tta946      complete   SUBMIT    the public 0.946, reproduced
ttasec      complete   SUBMIT    + secondary edge-feature TTA
ttasecw20   complete   SUBMIT    + SEW 0.20 on top of that
ttaret85    pushed               retention floor 0.90 -> 0.85
pp942n8     running              VALIDATOR_N_PER_TYPE 4 -> 8
```

## The phantom mounts are gone

Every arm on the 0.946 base was pushed with six dataset sources: pilkwang's three, plus
reyhanksatria's three re-hosted copies. `/kernels/pull` on our own kernel shows what Kaggle
did with them:

```
datasetDataSources  ['', '', '',
                     'pilkwang/biohub-deepcenter-unet3d-center-prior-v1',
                     'pilkwang/biohub-temporal-unet3d-seed314159-v1',
                     'pilkwang/biohub-tracking-support-pack-50ep-v1']
```

The author's three resolve to empty strings — they are not public, so nothing ever attached —
and every artifact path in the runtime log resolves under `pilkwang/`. The three that do mount
are public, CC0 Public Domain, and have 14,888 / 4,933 / 4,493 downloads between them.

I added the author's copies as insurance, on the reasoning that `/kernels/pull` returned
`['', '', '']` for the source kernel so the right copies could not be identified from
provenance, and that a spare mount costs nothing while a missing one is fatal. The first half
was right and the second half was insurance against nothing: they were never resolvable, so
they were never a fallback. What they *did* do is make the notebook page look like it depends
on data nobody can see.

`TTA946_SOURCES` is now pilkwang's three and nothing else, used by all five arms on this base.
The `.ipynb` files are byte-identical before and after — only the mount list in `_push.json`
changed — so re-pushing runs exactly the same code. Kaggle has no metadata-only update, so
the three completed arms are being re-pushed to pick it up; their existing versions stay
available and submittable either way.

## `pp942n8`: the sweep was under-validated, not inert

```
VALIDATOR: selected 16 held-out TRAIN samples (8 per embryo-type prefix, 2 prefixes found)
Re-writing submission.csv with the selected post-process configuration: tight55
nodes 119,279 -> 119,349   edges 115,009 -> 115,097
```

At `VALIDATOR_N_PER_TYPE 4` the sweep selected the default and `pp942` came out byte-identical
to `lb941`. At 8 it selects **`tight55`** and rewrites the output. So the earlier reading —
"the search ran and chose to change nothing" — was an artifact of eight samples, not a
property of the method.

**Still not worth a slot.** It is a 0.942-class arm (its own axis line says "public 0.939
base"), and 0.942 is rank 415 while the three arms waiting on the 0.946 base are rank ~181 or
better. Worth recording because it reopens the family: the mechanism is a holdout-selected
post-process sweep, and nothing about it is specific to the 0.939 base it currently sits on.

## Correction: the retention guard passes 84% of the time

I read `tta946`'s stdout, found 65 `BIOHUB_RETENTION_GUARD` lines all ending
`"use_primary": true`, and concluded the dual-seed fusion had *never once passed*. **That was
wrong.** The line is printed only when the guard rejects, so every printed record failing is
a tautology, not a finding. Reading a filtered stream as if it were the population is
`notes/68`'s config-dump trap and `notes/71`'s rank arithmetic in a third costume.

The kernel also writes `retention_guard_*.jsonl` — one record per frame, which *is* the
population. Pulled from `tta946`'s output zip, 400 records, 100 per movie:

```
44b6_0113de3b   100 frames    0 rejected   median retention 1.012
44b6_0b24845f   100 frames   64 rejected                    0.864
6bba_05b6850b   100 frames    1 rejected                    1.014
6bba_05db0fb1   100 frames    0 rejected                    1.011

all 400: 65 rejected (16%)   min 0.453   median 1.004   max 1.185
```

**The fusion applies on 335 of 400 frames**, and the median blend *adds* candidates rather
than losing them (retention > 1). `SECONDARY_DETECTION_WEIGHT` is not inert, and the claim
that it was is withdrawn.

What survives is narrower and still worth one measurement: on `44b6_0b24845f` alone the
fusion is discarded on 64% of frames, and that movie's median retention is 0.864 while the
other three sit above 1.00. One embryo where the two detectors disagree. `ttaret85` flips the
**26** frames in [0.85, 0.90) and leaves 39 vetoed below that. A narrow test, priced honestly.

## `ttaret85` v1 died on a guard I did not find

```
RuntimeError: Frame-retention diagnostic contract changed
```

The notebook defends this number in **three** places, not one. `_EXPECTED_NUMERIC` /
`_EXPECTED_TEXT` was the one I knew about (`gap44` taught it). At the very end of the run
there is a second, separate contract check over the guard records:

```python
if float(_guard_record['minimum_retention']) != 0.9 or ...:
    raise RuntimeError('Frame-retention diagnostic contract changed')
_guard_expected_use_primary = bool(primary_candidates > 0 and float(record['retention']) < 0.9)
```

and a third hardcoded `0.9` written into the receipt JSON, which raises nothing but would
have stated a threshold the run did not use. The arm ran the entire pipeline and died on the
verification. **Grep for the value, not for the guard idiom you already know** — that is the
generalisation of `notes/70` and it has now cost two runs.

`ttaret85` is six edits and rebuilt. The only `0.9` left is the fallback default inside
`os.environ.get(..., '0.90')`, which is unreachable while the variable is set.

## Kaggle keeps dataset sources a re-push does not name

Re-pushing with the three-entry list gave different results on different kernels:

```
claude-arm-ttasec      v4   3 sources   clean
claude-arm-tta946      v2   6 sources   the three empties survived a 3-entry push
claude-arm-ttasecw20   v1   6 sources   never re-pushed (the queue dropped it)
```

Same `_push.json` shape, same builder, different outcome — so a shorter list does not reliably
remove what a kernel already carries. Not yet explained; `tta946` is being pushed again to see
whether it converges. If the empties persist, the only certain fix is a fresh kernel slug,
which costs the URL.

## `ttaret85` ran, and it says the guard was right — no slot

The rebuilt six-edit version went through cleanly: `Configuration guard: PASS`,
`Frozen frame retention guard applied at 0.85`, no exception at the contract check.

```
                 nodes      edges
tta946         122,791    118,491
ttaret85       121,459    117,228
                -1,332     -1,263

44b6_0113de3b   25,621 ->  25,621     0        0
44b6_0b24845f   20,744 ->  19,413  -1,331   -1,263
6bba_05b6850b    6,150 ->   6,149      -1        0
6bba_05db0fb1   70,276 ->  70,276       0        0
```

**Entirely confined to `44b6_0b24845f`**, exactly as the per-frame retention data predicted —
that is the one movie where the two detectors disagree, and the only one with frames in
[0.85, 0.90). Letting the blend through on those 26 frames costs **6.4% of that movie's nodes
and 6.5% of its edges**.

Price it against the metric. `adj = edge_J x (1 - 0.1 x ratio)`, so removing 1,332 of 122,791
nodes is worth about **+0.1%** through the ratio term. Removing 1,263 of 118,491 edges is
**-1.1%** on `edge_jaccard` unless those edges were wrong. The magnitudes are 10:1 apart, so
better than **90% of the removed edges would have to be false positives** for this to break
even.

That is not a score prediction — `diff_arms` never makes one — but it is an asymmetry worth
acting on, and it agrees with the mechanism: `44b6_0b24845f` has median retention 0.864 while
every other movie sits above 1.00, so the secondary detector is simply unreliable on that
embryo and the 0.90 floor is the thing that notices. **The guard is not a missed opportunity;
it is load-bearing.**

**Axis closed, no slot spent.** Which is what the diff tool exists for: three arms of evidence
(the stdout misreading, the per-frame jsonl, and now the output diff) resolved without a
submission.

## Mount lists are clean

```
claude-arm-tta946      v3   3 sources   0 empty
claude-arm-ttasec      v4   3 sources   0 empty
claude-arm-ttasecw20   v2   3 sources   0 empty
claude-arm-ttaret85    v2   3 sources   0 empty
```

The `run_arm` version check earned itself immediately: the queue logged **four** discarded
`v0` pushes across those re-pushes, every one of which the old code would have reported as a
successful run.

## Both graded: 0.945, rank 272 — and our own patch is the reason

```
ttasec   (v3)   0.945     the 0.946 base + our secondary edge-feature TTA
tta946   (v2)   0.944     the same base, unmodified
```

**Rank 272 of 3,281**, up from 287 at 0.942. Two things fall out, one good and one that is
worth more than the good one.

### 1. The patch works

`ttasec` beat `tta946` by **+0.001** — same base, one change, ours. That is the first
mechanism this project has invented that has scored, as against reproducing someone else's.
It also makes `ttasecw20` (the same patch plus SEW 0.20) the most valuable untested arm we
hold, since it is the only thing built on a change now known to pay.

**With one caveat I cannot yet remove.** The two submissions did not run on the same hardware.
`tta946` v2 ran on **dual T4** (`Tesla T4\nTesla T4`, 94,420 log chars, the multi-GPU sharding
branch); `ttasec` v3's accelerator is unrecoverable — Kaggle exposes no per-version metadata
and `/kernels/output` only serves the latest. So +0.001 is *consistent with* the patch working
and also consistent with a hardware difference. It needs a same-GPU pair before it is a
measurement.

### 2. The public 0.946 reproduces at 0.944 for us, and that gap is the real prize

198 teams sit on exactly 0.946. We ran that notebook unmodified and got **0.944**. Today's
board:

```
score    better   tied      band      we land at
0.948        29      7     30-36              37
0.947        36     16     37-52              53
0.946        52    198    53-250             251
0.945       250     34    251-284            285   <- us, rank 272
0.944       284     69    285-353            354
```

The 0.946 plateau grew from 140 teams to 198 in a day, and the cliff above it got sharper:
**0.947 lands at rank 53.** So the -0.002 we cannot account for is worth more than every knob
left on the board. Closing it puts `ttasec` at 0.947.

The suspect is ours: `run_arm` requests `machine_shape="gpuT4x2"`, so our T4 draws are
**two** GPUs and take the notebook's sharded-prediction branch, while the public runs almost
certainly take the single-device one. Nobody else is running this configuration.

### What to submit next, and from which version

All three arms' current versions are P100, single-process, and byte-for-byte identical to the
outputs analysed here (verified by sha256), so the versions below are the ones already read:

```
ttasecw20   Version 2    the untested arm: ttasec + SEW 0.20
tta946      Version 3    P100 single-process control -- does the public 0.946 reproduce?
ttasec      Version 4    P100 reading of our best arm, same hardware as the control
```

Two and three are the confound killer: a same-GPU pair prices the patch honestly, and
`tta946` v3 answers whether the missing 0.002 is hardware or the claim.

## The titles lie. Look up the author on the leaderboard.

```
reyhanksatria     rank 293   0.944    <- author of "biohub-cell-tracking-0-946-lb"
analyticaobscura  rank 100   0.946    <- author of "biohub-lb-941" and "biohub-lb-942"
rishabhr0y        rank 108   0.946
pilkwang          rank  56   0.946    <- the model author
```

**`reyhanksatria` scores 0.944** — exactly what our unmodified fork of their notebook scored.
There is no reproduction gap. The title is aspirational, and the notebook's own
`Verified score progression ... 0.946` is a claim, not a graded result. Yesterday's "-0.002 we
cannot account for, worth more than every knob left on the board" was a phantom, and the
`gpuT4x2` sharding theory built on top of it is withdrawn too — `--slice i::n` partitions the
*video list*, so each movie runs through identical code whether it is sharded or not.

Two things follow, one of them good.

**`ttasec` is +0.001, measured.** Base 0.944 (two independent accounts agree — theirs and
ours), ours 0.945. No hardware confound, because the confound was invented to explain a gap
that does not exist. Our secondary edge-feature TTA is the first thing this project invented
that has scored.

**And we are already past the verifiable public frontier.** Every public notebook whose score
we can check tops out at 0.944. We are at 0.945, rank 272. There is nothing left to fork:
a scan of 906 kernels finds the only titles above 0.946 are July's metric-hack notebooks and
the August `948` branch `notes/68` proved inert. **0.948 has to be built.**

## Three arms toward it, and where each came from

### `ttaz16` — the symmetry the model was trained on and nobody averages over

The support pack ships its own training code. `scripts/augmentations.py`:

> *"Random spatial flip: samples uniformly from all 8 axis-aligned symmetries. Each of Z, Y, X
> is independently flipped with probability 0.5."*

Two augmentations were used, brightness and flip, and the flip group is the full {Z, Y, X}
product. Every transform in the public eight-view TTA acts on `(-2, -1)` — Y and X. **Z is
never averaged over, and Z is the axis the model was explicitly trained to be invariant to.**
The rot90 and transpose views that *are* used are not in the training set at all; they work on
Y/X isotropy, not on anything the model was taught.

So this is the lever that produced both gains in this lineage — more of the ensemble the model
already supports — pointed at the one direction nobody has tried. `temporal_unet.py` gives the
layout as `(B, T, C_out, Z, Y, X)`, so `-3` is Z. Sixteen views, `_nv` prints 16, primary
predict time doubles — which is why it is being measured for RUNTIME on the visible clips
before it is ever proposed for a slot.

### `ttadse44` — the threshold identical in all 56 public kernels

`DUAL_SEED_EDGE_THRESHOLD` 0.48, never swept by anyone, gating every candidate edge in every
frame pair. Downward first, on Soheil's diagnosis (rank 2): *"many 'linking' issues actually
originated earlier during node selection."*

### `ttadom` — ported from someone actually at 0.946

Every notebook in this lineage ends its edge stage with

```python
motion_edges = motion_relink_edges(nodes_by_id, stats, learned_edge_probs)
if motion_edges:
    stats['motion_relink_replaced_raw_edges'] = len(edges)
    edges = motion_edges          # the entire ILP edge set, discarded
```

and our own `run_stats` prices it: `motion_relink_replaced_raw_edges` is **67,249** on
`6bba_05db0fb1`. The motion model overwrites essentially every edge the ILP produced.

`rishabhr0y` (rank 108, 0.946) replaces that with a reconciliation — a raw ILP edge survives
if it **strictly beats every conflicting motion edge at both endpoints**, relative confidence
only, strongest first so the graph stays one-parent/one-child, with a sparsity guard. Their
sibling notebook is named `biohub-edge-density-adaptive-on-0943` and is built on this one from
the same 0.941 base, which prices it: **0.941 -> 0.943**.

It is orthogonal to everything we have measured. `ttasec` changes what the association model
*sees*; this changes what happens to its output when the motion model disagrees. Ported by
translation, not copy — their notebook uses double quotes and carries a
`division_reference_motion_edges` line ours does not.

**The arithmetic to the target:** 0.945 + dominance (+0.002) = 0.947, plus either of the other
two = 0.948, which is rank 37.
