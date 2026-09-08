# What the public 0.946 actually is, and the half of it nobody finished

`claude-arm-tta946` — reyhanksatria's `biohub-cell-tracking-0-946-lb`, forked unmodified —
completed on 2026-09-08. It is the largest published step in this lineage (+0.005, four
times anything we have measured) and it is worth submitting. This note is about what is
inside it, because reading the code turned one suspicion into a non-finding and produced
one lead.

## 1. The 400-epoch weights are a non-finding

The log prints

```
ARTIFACTS:      /kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1
Artifact name:  biohub-tracking-support-pack-400ep-snapshot-v1
Weight sha256:  12f6881ee3620a831697ca098ff8f48e687a24225f4e048b538deec3562fe771
```

which reads as if the 0.946 mounts a longer-trained checkpoint than everyone else — and
would have meant the gain was weights, not method, and not portable. It is not that.
`lb941`, `ckpt948` and `tta946` all materialise **the same primary SHA** `12f6881e…`, and
all three mount `-50ep-v1`. The `400ep-snapshot` string is the `name` field *inside*
pilkwang's `-50ep-v1` manifest: the dataset is misnamed, and every notebook in this lineage,
ours included, has been running the same snapshot since the beginning.

**So the +0.005 is code.** That is the good outcome — code can be read and extended.

## 2. What the mechanism is

The detector already ran an eight-view D4 ensemble. `model.encode` returns *two* things:

```python
unet_out, det_logits = model.encode(imgs)
```

— association features and detection logits. Every notebook from 0.933 to 0.941 averaged
`det_logits` over the eight views and **threw away seven of the eight feature maps**, then
handed the association head a one-view feature map to pair with an eight-view detection map.
The 0.946 change is to keep them:

```python
_unet_acc = unet_out.clone() if _edge_tta else None
...
    _unet_acc = _unet_acc + _u_flip.flip(dims)          # ×7 views
...
    unet_out = _unet_acc / _nv
```

No extra compute — the eight forward passes were already happening; the author simply stopped
discarding their output. The log confirms it fired:
`EDGE_TTA_ACTIVE views = 8 mean_abs_feat_delta = 0.323`, and 122,791 nodes against lb941's
119,279.

This is what Tang (rank 5, 0.961) meant by *"most of the gains come from model improvements"* —
except that it needed no retraining, which we had assumed model-level changes all did.

## 3. The secondary model still has the bug

`SECONDARY_DETECTION_WEIGHT` is 0.80, so the secondary UNet runs its own eight D4 encodes.
Every one of them looks like this:

```python
secondary_unet_out, secondary_det_logits = secondary_model.encode(imgs)
...
    _, secondary_det_flip = secondary_model.encode(secondary_imgs_flip)   # feature map dropped
```

Eight encodes, eight feature maps discarded, `secondary_unet_out` left at the single
canonical view — and it goes straight into

```python
secondary_feat_src = secondary_model._index_features(secondary_unet_out[:, f_idx], ...)
secondary_logits_pair = secondary_model.predict_edges(secondary_feat_src, ...)
```

whose output is blended at `SECONDARY_EDGE_WEIGHT` and decides `low_margin_consensus`.
**The exact inconsistency the 0.946 fixed on the primary is still there on the secondary,
and the compute to fix it is already being spent.** No public notebook touches it.

`claude-arm-ttasec` is that patch: six verified replacements mirroring the author's own, with
the author's own two guards kept — shape equality, and a raise if the mean feature delta is
zero. An inert patch kills the run instead of quietly reproducing 0.946, which is `notes/68`'s
lesson paid forward. The accumulation is `+=` rather than the author's `x = x + y`, because
this sits on top of the primary TTA's peak on a 16 GB P100.

## 4. The rank arithmetic, stated correctly

`notes/71` recorded the rule after getting this wrong: read the `Rank` column, do not compute
`count(score > ours) + 1`. There is a second half to that rule, and it bites here.

Board of 2026-09-08 11:23Z, 3,244 teams:

```
score    better   tied      band        a new entrant lands at
0.947        31      9     32-40                            41
0.946        40    140    41-180                           181
0.945       180     20   181-200                           201
0.942       279    135   280-414                           415   <- us
```

The `Rank` column at rank 100 reads 0.946, and that is true of the teams already there.
**We would not join at rank 100.** Kaggle breaks ties by submission time, so a new 0.946
arrives at the *bottom* of a 140-team pile-up: **rank ~181**. Still +106 places, and worth
having. But the top 100 does not cost 0.946 for us — it costs **0.947**, which lands at 41.

That reframes the whole endgame. 0.946 is where the public notebook lives, so 140 teams sit
on it exactly; one thousandth above it is nearly empty. The gap we need is one thousandth,
and it must come from something the public notebook does not already do.

## 5. Which is why `ttasec` is the arm that matters

Parameter tuning is finished on this checkpoint — `sewdet` demonstrated it, with SEW and DET
each worth +0.001 alone, exactly additive in node count, and worth nothing together. Knobs
land on the shelf; only mechanisms move it. `ttasec` is a mechanism, it is the same one that
just paid +0.005, it costs no GPU time, and it is applied where nobody has applied it.

Running alongside it, `ttasew20` re-tests `SECONDARY_EDGE_WEIGHT` 0.15 → 0.20 on the 0.946
base. The saturation that closed it was measured against single-view association features; the
argument for re-opening it is that the features are now eight-view, and the two arms read each
other — if better secondary features are worth something, the weight on them should want to
be larger.
