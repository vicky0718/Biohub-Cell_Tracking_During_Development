# Arm log — running record, updated as each run lands

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
| `dcgap40` | lb941 | `DC_GAP 0.25 → 0.40` | — | — | — | running |
| `gap44` | lb941 | `GAP_CLOSE_UM 5.0 → 4.4` | −42 | −42 | +1 | complete — **not worth a slot** |
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
