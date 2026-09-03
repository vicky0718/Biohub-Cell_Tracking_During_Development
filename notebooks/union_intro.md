# Do the two detectors find DIFFERENT cells?

```
0.901 submitted (rank ~1388/3038)    0.935 bronze    0.947 gold
pack detector   node recall 0.996     spotiflow   0.547        (notes/47)
```

`notes/47` closed spotiflow, and the reasoning was sound for what it tested: 0.547 recall
against 0.996, and the pack dominating the **recall-per-node curve** at every matched node
count. **Neither of those is a statement about set membership.** Two detectors can sit on
opposite sides of a quality curve and still find different cells, and the union was never
computed — `claude_spotiflow` recorded `pack_recall` and `spot_recall` in separate columns
and never put the two detection sets together.

## Why the question is live again

Two things changed after `notes/47`.

**The budget premise it was judged under is false.** `notes/52` measured `ratio = -0.129`:
we run **12.9% under** the node budget with the multiplier already paying 1.013. Spotiflow
was evaluated when the working assumption was that over-prediction costs budget. At our
actual operating point there is headroom to *add* nodes, which is the one thing an ensemble
must do.

**The ceiling is now measured.** `notes/51` put `fn_detect` at 583 edges, and
`claude_divsweep`'s arms separate it — `ctl/raw` 238 vs `inc/raw` 608 — so ~370 is imposed
by our own ILP weights and **the detector's own share is ~238 edges, 1.72% of GT**. That is
the hard ceiling on any detection-stage ensemble. Above `notes/44`'s 0.0015 floor, roughly
half the bronze gap, and not a guaranteed win.

## What has already been tried, and why this is neither

```
claude_secondary       blended a second EDGE model (temporal-unet3d), adaptive weight
                       from top-2 margin.  +0.0026, t=0.63 -- UNRESOLVED, needs n~147
claude_deepcenter_veto a parallel corrector, but applied to GAP-CLOSING candidates.
                       Bounded at ~0.002 because gap-closing is only worth +0.0013
```

Both act on the **graph**. This acts on **detection**, which is where `notes/51` says the
remaining loss now sits.

## The measurement

`purescore.match_nodes` returns, per predicted node, the GT index it matched or −1 — so the
covered GT set is directly readable. Matching is one-to-one **within a frame**, so the union
is computed by running the matcher on the **concatenated** detections, not by set-unioning
two separate runs (which would let one GT node be claimed twice and overcount the rescue).

Two modes, because the full union is not a deployable operating point:

```
union      every spotiflow detection added        upper bound on recall, worst budget
selective  only spotiflow detections >7um from    the ones that could RESCUE a miss
           any pack detection in the same frame   rather than duplicate a hit
```

7 µm is the scorer's own match radius: a spotiflow detection closer than that to a pack
detection cannot rescue anything the pack already has, so it is pure budget cost.

## Pre-registered predictions

Graded **per embryo**, both means printed (`notes/49`).

1. **Reproduction.** `pack_recall` within 0.005 of `notes/47`'s **0.996** and `spot_recall`
   within 0.02 of **0.547**. Otherwise this is not the same measurement and nothing below
   compares to the record.
2. **The union rescues something.** `union_recall − pack_recall > 0.002`. **This is the
   crux.** If the union recovers essentially nothing, spotiflow's detections are a subset of
   ours, the ensemble cannot work at any weighting, and the direction closes for one run.
3. **The selective union keeps most of the rescue.** `rescued_sel > 0.5 * rescued` — the
   rescues come from spotiflow detections genuinely far from ours, not from re-matching
   noise. If the rescue survives only in the full union, it is a matching artifact.
4. **The rescue is affordable.** Fewer than 20 added nodes per rescued GT node in the
   selective mode. At `ratio = -0.129` there is real headroom, but the multiplier costs 0.1
   per unit of ratio and a rescue that doubles the node count spends more than it earns.
5. **It holds on both embryos.** `notes/07` §3: the test set is a third pair, and a pooled
   result across crops of two says nothing about it.

*If 2 fails, detection-stage ensembling is closed regardless of architecture — parallel
corrector, veto, weighted blend or anything else — because there is nothing to add. That is
the outcome this run is designed to make cheap.*
