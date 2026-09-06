# The run stats say which gate actually binds — and it is not the one I picked

Every run of this notebook family writes `run_stats.csv`: sixty-odd counters per dataset,
covering every gate, cap and repair stage. `claude-arm-ckpt948`'s copy, summed over the four
verification clips, reorders the arm queue and kills one arm's premise.

## 1. The division stage: the divergence gate is 82% of the filtering

```
safe_division_geometric_candidates            356
safe_division_divergence_rejected           1,604   <- 82% of everything proposed
safe_division_mutual_nn_rejected              104
deepcenter_safe_div_checked / rejected    356 / 111
safe_divisions_added                          215
safe_division_skipped_cap                       0   <- volume caps are NOT binding
```

`SAFE_DIV_DIVERGE_UM` rejects **more than every other division gate combined, by an order of
magnitude**. And `skipped_cap = 0` says `notes/60`'s trap — a parameter moves, a volume cap
binds first, the run reproduces its base — is not active here. The gate really is the
constraint.

It is also *stepped and stopped* in `notes/65` §3's sense: `nusrati/0-938` ran **4.5**,
`0-940` moved it to **2.25** as part of a +0.002, and every 0.941 inherited 2.25 unchanged.
Larger is stricter, so the step that paid went *toward* more divisions, and nothing below
2.25 has been published. New arm **`div15`** (2.25 → 1.5), and it goes to the front of the
queue.

## 2. `dcgap40`'s premise was wrong

I added it arguing the DeepCenter *gap* gate "gates a far larger population" than the
division one. The counters say otherwise:

```
deepcenter_gap_checked                        230
deepcenter_gap_accepted                        48
deepcenter_gap_rejected                       182   <- already rejecting 79% at 0.25
deepcenter_gap_bypassed_strong_motion         596   <- never reaches the gate at all
```

Two things at once. The gate sees **230** proposals against the division gate's **356**, so
it is the *smaller* population, not the larger. And at its current 0.25 it already rejects
79% of what reaches it — raising it to 0.40 can only remove the 48 that survive. The ceiling
is a few dozen edges across four clips. **Demoted**, not removed: it is still a
never-moved constant and cheap to run once the arms that can move something have run.

`notes/67` §1's reasoning was sound and its data was the author's calibration table; what it
lacked was this file, which says how much traffic each gate actually sees. Calibration says
where a gate is best; the counters say whether the gate matters.

## 3. Runtime, and why test-time augmentation is out

```
predict_minutes_total    10.12   (total for all four verification clips)
whole run                39 min  (P100, including the wheelhouse install)
```

`notes/66` §3b put the graded set at ~17× the verification set. Prediction alone extrapolates
to ~2.9 h, and the full run — if the non-prediction stages scale per-clip too — to roughly
**11 hours against a 12-hour limit**.

That prices hengck23's test-time-augmentation suggestion out. Even single-axis flip TTA
doubles detection, adding ~3 h, and lands at ~14 h. It is not a question of whether TTA helps
the score; there is no room to run it. Recording that here so it stops coming up.

## 4. Other counters worth having

```
short_track_components_removed     753        OUTPUT_MIN_TRACK_LEN = 6 is doing real work
short_track_nodes_removed        3,265        2.7% of raw nodes
pruned_isolated_nodes               53        confirms notes/66: zero survive to output
dropped_long_edges                  11        cap_edge_length is inert here
gap_candidates / selected    1,124 / 644      gap-closing accepts 57%
gap_skipped_node_cap                 0        no volume cap binding
gap2_candidates / selected     352 / 135
motion_relink_edges            115,869        i.e. essentially every edge is relinked
```

`dropped_long_edges = 11` is worth keeping: hengck23's "long links are almost wrong"
(`notes/58`) is already handled upstream in this pipeline — by the time the length cap
runs there are eleven edges left for it to drop.

## 5. Queue after this

```
1  div15     SAFE_DIV_DIVERGE_UM 2.25 -> 1.5     the gate that does 82% of the filtering
2  dc40      DEEPCENTER_SAFE_DIV_THRESHOLD 0.25 -> 0.40   author's own f1 maximum
3  union     SECONDARY_LINK_MODE -> adaptive     crosses the two public 0.941 routes
4  adaptive  rishabhr0y's 0.941 unmodified       the second floor
5  gap44     GAP_CLOSE_UM 5.0 -> 4.4
6  dse44     DUAL_SEED_EDGE_THRESHOLD 0.48 -> 0.44
7  dse52     DUAL_SEED_EDGE_THRESHOLD 0.48 -> 0.52
8  det960    DET_THRESHOLD 0.965 -> 0.960
9  dcgap40   demoted (§2)
```
