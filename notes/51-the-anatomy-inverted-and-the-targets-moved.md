# The anatomy inverted, the medal line moved, and r35 built the budget lever we skipped

A forum re-scrape (79 → 90 threads) and one CPU probe. Three findings, and the first two
change what this project should be doing.

---

## 1. The medal thresholds moved. Every note in this repo quotes stale ones.

```
3,038 scored teams, top 0.963
GOLD    cutoff 0.947   (n=16)      notes say 0.944
SILVER  cutoff 0.936   (n=135)
BRONZE  cutoff 0.935   (n=152)     notes say 0.926
our 0.901  ->  rank ~1388/3038, top 45.7%
```

The gap to bronze is **0.034**, not the 0.025 every note since `notes/44` has been quoting.
The field moved up while we were sweeping config.

## 2. 🚨 The edge anatomy has INVERTED since `notes/26`. Detection is now the ceiling.

`claude_divsweep` computed `edge_anatomy` for all 16 arms and the grading cell never printed
it; `claude_probe_anatomy` reads it back off the kernel output.

```
                    notes/26 (ctl/raw)        now (inc/g2sp6, shipped)
tp                12,909   93.33%            12,962   93.71%
fn_mislink           473    3.42%  <-largest    226    1.63%
fn_detect            238    1.72%               583    4.21%  <- LARGEST
fn_gap               212    1.53%                61    0.44%
```

`notes/26` concluded *"Detection is not the ceiling… it is the smallest recoverable
bucket. The cells are found; the graph joins them wrong."* **That is no longer true.** We
fixed most of the mislinks (473 → 226) and more than doubled the undetected endpoints
(238 → 583). Detection is now **67% of all remaining edge loss**.

And the cause is our own: the arms isolate it.

```
ctl/raw   fn_detect 238        pack default ILP weights
inc/raw   fn_detect 608        ratio0.4_2.0            +370
```

**+370 is exactly `notes/36`'s "fn_detect tax."** `notes/36` measured it, accepted it
because the total rose +0.0221, and was right to — mislinks were 473 then. That trade was
priced when linking was the bottleneck. It has never been re-priced now that it is not.

Note also `ctl/g2sp6` has a **higher raw edge Jaccard** than `inc/g2sp6` (0.9076 vs 0.9037).
The incumbent still wins on total (0.9188 vs 0.9054) via `div_J` and the budget multiplier,
so this is not a reason to switch — it is a reason to stop assuming the edge term is where
`ratio0.4_2.0` helps.

### What perfect linking is worth, and why it is not enough

*(Correcting this run's own output: the probe printed `false-positive edges 0` and an
`edge_J now 0.9371` because it guessed an `fp` key `edge_anatomy` does not return. 0.9371 is
recall, not Jaccard. Recomputed from `summarise`'s real 0.9037: fp ≈ 511.)*

```
now                                    0.9037
every gap+mislink repaired, FPs kept   0.9237   +0.020
...and the mislinked FPs corrected     0.9385   +0.035
gap to bronze                                   +0.034
```

**Perfect linking — all 287 gaps and mislinks fixed — barely reaches bronze at the
optimistic end and misses it at the realistic one.** Detection is where the remaining 583
edges are, and no linker recovers an edge whose endpoint was never matched.

This is what the forum has been saying. Tang (MASTER): *"detection → linking → division.
detection should come first."* Mendrika Ramarlina (MASTER): *"edge recall ≈ node recall² ×
conditional linking accuracy… detector misses are especially expensive."* Soheil Ayati (18
votes, the highest-voted answer in these threads): *"many 'linking' issues actually
originated earlier during node selection, so improving the edge model won't necessarily
help."* Tang again, on the shared checkpoint: *"the current ckpt has kind of hit a wall,
it's hard to get more gain from post-processing alone."*

`notes/26` is the reason this project spent ten runs on the graph. It was correct when
measured and is now stale.

## 3. r35's linker: two mechanisms we never built

`src/track/linker.py` (24.5 KB) and `src/track/ilp.py`, read for the first time. Their
`TrackConfig` carries:

```python
ilp_window: int = 3              # solve_three_frame_ilp -- a WINDOWED formulation
max_pred_nodes: int | None       # "Pivot H -- drop short false tracks to cut |V̂|/φ penalty"
rank_tracks_by_geometry: bool    # "R11 -- rank tracks by link geometry under budget"
drop_isolated_nodes: bool = True
enable_gap_closing: bool = False # they do NOT gap-close at all
prune_distance_percentile: 97.0  # adaptive edge pruning, MAD-based
link_reward_um: float | None     # "cost = dist - reward; lower = reluctant linking"
```

Two of these are levers this project identified and never implemented:

* **Per-dataset node budget.** `notes/04` §9 said it plainly — *"the two datasets that are
  the leaderboard have node budgets 11× apart, 64 vs 698 cells per frame. A detector with
  one global threshold cannot serve both."* We still ship **one global `DET_THRESHOLD`**.
  r35 caps per dataset and ranks tracks by geometry to decide what survives. `notes/46`
  closed *uniform* thinning and `notes/48`/`49` closed *confidence* thinning; **track-level
  geometric ranking under a per-dataset cap is a third selection rule, and it is untried.**
* **A 3-frame ILP window** instead of our global solve.

Independently confirmed on the forum this week (thread 739018, Michael Hernandez + TWEAK,
reading the released evaluator): `adj_edge_jaccard = max(0, edge_J·(1 − 0.1·ratio))` has a
floor but **no ceiling**, so under-prediction pushes it above 1.0. TWEAK's synthetic case:
*"with edge TP/FP/FN fixed at 90/5/5, reducing predicted nodes from 100 to 50 changed
adjusted-edge Jaccard from 0.900 to 0.945."* Nodes that are not endpoints of a kept edge are
pure cost. `notes/45`'s mechanism, confirmed from the other side.

**One unverified claim, flagged not adopted:** Rishabh Roy (EXPERT) states the local proxy
is scored on sparse GT while the LB uses dense labels, so the proxy barely penalises
over-detection. hengck23 (GRANDMASTER) challenged it directly — *"where is it mentioned?"* —
and it is unresolved. Do not build on it.

## 4. What follows

The cheap graph-side work is finished, and this run says so with numbers rather than by
exhaustion:

```
config     closed (notes/44, 49)
divisions  closed (notes/43, 50)
linking    bounded at +0.020..+0.035 -- cannot reach a 0.034 gap on its own
detection  67% of remaining edge loss, and +370 of it is our own ILP tax
```

Two candidates, in cost order:

1. **Per-dataset budget targeting with track-level ranking** (r35's `max_pred_nodes` +
   `rank_tracks_by_geometry`). Cheap — post-processing on cached graphs, no GPU — and it is
   the one selection rule the two closed thinning runs did not test. Grade per embryo
   (`notes/49`).
2. **The detector**, which is what every strong competitor says and what the anatomy now
   shows. `claude_zhpilot` is built and unrun. Expensive, high variance, and its own gate
   analysis puts nothing within 0.16 CV of the bar.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.935 bronze    0.947 gold
```
