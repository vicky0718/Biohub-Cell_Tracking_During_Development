# What has actually been tried — audit before spending slots

Prompted by a fair challenge: *"proceed only if we haven't tried these."* With 30 submissions
and 45 arms in the registry, "built" and "tried" have drifted apart.

## 1. The gap: 45 arms in the registry, 14 ever scored

Every arm ever submitted, with its score:

```
0.941  lb941    union    det955
0.942  sew20    sew25    sew30    sewdet    ttaz16    ttaz16dom(x2)
0.944  tta946   norelink
0.945  ttasec                                            <- best
0.938  div15
0.933  dc40
```

`ttaz16dom` v4 came back **0.942**, identical to its v3 and to `ttaz16`. Third consistent
reading on that family: Z-flip TTA costs −0.003 and the confidence-dominance port is neutral.
`notes/75`'s conclusion stands, now with a replicate.

**Thirty-one arms were built and never scored.** Most are on the retired `lb941` base and are
not worth a slot. Five are on the current 0.946/`ttasec` base, and four of those have
**completed runs sitting on Kaggle right now**:

```
arm         base          run       knob
ttasecw20   0.946/ttasec  complete  SECONDARY_EDGE_WEIGHT 0.15 -> 0.20
ttadse44    0.946/ttasec  complete  DUAL_SEED_EDGE_THRESHOLD 0.48 -> 0.44
ttaret85    0.946         complete  DUAL_SEED_MIN_CANDIDATE_RETENTION -> 0.85
ttadom      0.946/ttasec  complete  motion-relink confidence dominance
ttasew20    0.946         never ran SECONDARY_EDGE_WEIGHT 0.20, no sec-TTA
```

These cost **zero GPU**. `ttadom` is the weakest of them — dominance was measured neutral by
`ttaz16dom` vs `ttaz16` — but it is still free.

## 2. The six new arms: verified untried

Checked against both the submission list and the registry:

```
arm        knob                              nearest prior arm        ever scored?
mtl4       OUTPUT_MIN_TRACK_LEN 6 -> 4       none exists              NO
sdw90      SECONDARY_DETECTION_WEIGHT .8->.9 none exists              NO
mrr12      MOTION_RELINK_RELAXED_UM 10->12   none exists              NO
bew10      BIDIRECTIONAL_EDGE_WEIGHT .15->.1 none exists              NO
gapdc15    DEEPCENTER_GAP_THRESHOLD .25->.15 dcgap40 (0.40) built,    NO
                                             never submitted
gap65      GAP_CLOSE_UM 5.0 -> 6.5           gap44 (4.4) built, died  NO
                                             on the drift guard
```

Four of the six touch a parameter **no arm in this project has ever moved**. The other two have
a prior arm in the *opposite* direction that was never scored, so the knob is unmeasured on
both sides. None duplicates anything.

One near-miss worth naming: `sdw90` moves `SECONDARY_DETECTION_WEIGHT`, which is **not** the
`SECONDARY_EDGE_WEIGHT` of `sew20`/`sew25`/`sew30`/`sewdet` (all 0.942). One blends a second
UNet's detections, the other its edge logits. Different parameters, different stages.

## 3. Order to submit, ten arms available

Front-loaded by information per slot — largest output change and the one knob that has ever
paid here:

```
1  mtl4        +3,279 nodes   largest effect in the batch
2  ttasecw20   free           SECONDARY_EDGE_WEIGHT, the only knob that ever gave +0.001
3  sdw90       -1,021 nodes   second largest
4  ttadse44    free
5  ttaret85    free
6  mrr12  7  gapdc15  8  gap65  9  bew10  10  ttadom
```
