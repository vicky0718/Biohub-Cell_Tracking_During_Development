# Ten arms scored. Nine tied or lost. One gained — and it stacks.

The whole batch went to the board. No offline score was used to rank any of it.

```
arm          score   base   delta  knob
ttaret85     0.945  0.944  +0.001  DUAL_SEED_MIN_CANDIDATE_RETENTION 0.90 -> 0.85
ttasecw20    0.945  0.945  +0.000  SECONDARY_EDGE_WEIGHT 0.15 -> 0.20
mrr12        0.945  0.945  +0.000  MOTION_RELINK_RELAXED_UM 10 -> 12
gapdc15      0.945  0.945  +0.000  DEEPCENTER_GAP_THRESHOLD 0.25 -> 0.15
ttadom       0.945  0.945  +0.000  motion-relink confidence dominance
sdw90        0.944  0.945  -0.001  SECONDARY_DETECTION_WEIGHT 0.80 -> 0.90
gap65        0.944  0.945  -0.001  GAP_CLOSE_UM 5.0 -> 6.5
mtl4         0.943  0.945  -0.002  OUTPUT_MIN_TRACK_LEN 6 -> 4
ttadse44     0.943  0.945  -0.002  DUAL_SEED_EDGE_THRESHOLD 0.48 -> 0.44
bew10        0.943  0.945  -0.002  BIDIRECTIONAL_EDGE_WEIGHT 0.15 -> 0.10
```

The prediction recorded in `notes/82` before any of it ran — *"knobs land on the shelf here,
the one exception gave +0.001, and six such do not reach 0.947"* — is what happened. Worth
noting because this project's record of predicting its own results is otherwise poor.

## 1. The one gain, and why it is not just another +0.001

**`ttaret85` scored 0.945 from a base of 0.944.** It carries no secondary-TTA patch — verified
twice, `SEC_EDGE_TTA_ACTIVE` fires 0 times in its log and the patch is absent from its notebook
— so its base is `tta946`, not `ttasec`. The retention floor 0.90 → 0.85 is therefore worth
**+0.001 in its own right**, measured against the right control.

And it is **disjoint from the +0.001 `ttasec` already banks**: one is an inference-time feature
average over eight D4 views, the other a dual-seed candidate floor in post-processing.
Different stages, different data. Nobody has run them together. `ttasecret85` does.

The honest caveat is `sewdet`: SEW and DET were each +0.001 alone, exactly additive in node
count, and worth **nothing** together. Additivity has failed here before. This is one arm to
find out, not a plan.

## 2. The losses bracket three knobs, and two point somewhere untested

A loss says the knob moved the wrong way, which locates the better direction:

```
mtl4       6 -> 4     -0.002   filter is UNDER-aggressive; 6 -> 8 untested anywhere
ttadse44   .48 -> .44 -0.002   threshold wants to rise; 0.52 untested on this base
bew10      .15 -> .10 -0.002   0.15 is at or near optimum -- and the author pinned it twice
```

`bew10` is the informative null: the author hard-pinned `BIDIRECTIONAL_EDGE_WEIGHT` in two
separate places, and the board agrees with them. That one is closed.

The other two are now `mtl8` and `dse52t`.

## 3. Three arms built

```
ttasecret85   ttasec + retention 0.85     the only gain, stacked on the best base
mtl8          ttasec + MIN_TRACK_LEN 8    opposite of mtl4's -0.002
dse52t        ttasec + DUAL_SEED 0.52     opposite of ttadse44's -0.002
```

**A build refusal worth recording.** `ttasecret85` first stacked `RETENTION85_EDITS` verbatim
and `build()` refused: *matched 0x across 0 cells*. `sec_tta_edits()` runs first and **inserts**
`os.environ['BIOHUB_SECONDARY_EDGE_FEATURE_TTA'] = '1'` directly after the `EDGE_FEATURE_TTA`
line — which is precisely the line the retention edit uses as context to tell the notebook's
*two* retention assignments apart. The anchor check caught a silent-wrong-edit before it cost a
run; the first edit now re-anchors on the inserted line and the other five are untouched.
`ttaret85`'s own entry reuses the same hoisted constant, so the two can never drift apart.
