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
| `dse52` | lb941 | `DUAL_SEED_EDGE 0.48 → 0.52` | — | — | — | running |
| `det960` | lb941 | `DET_THRESHOLD 0.965 → 0.960` | — | — | — | queued |
| `dcgap40` | lb941 | `DC_GAP 0.25 → 0.40` | — | — | — | queued |
| `gap44` | lb941 | `GAP_CLOSE_UM 5.0 → 4.4` | — | — | — | re-queue (guard fix) |
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
