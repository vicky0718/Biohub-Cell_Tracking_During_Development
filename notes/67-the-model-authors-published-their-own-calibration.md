# pilkwang published their calibration inside the artifact, and nobody reads it

Every notebook in the 0.934 → 0.941 band mounts three `pilkwang` datasets and tunes
`BIOHUB_*` values against the leaderboard. Nobody in that lineage cites what is sitting in
the datasets themselves: a threshold sweep, a training history, and a split manifest.
`claude_packls`, `claude_packmeta` and `claude_packcsv` read them. All three are CPU
notebooks — no GPU, no submission slot, minutes each.

## 1. The DeepCenter gate has a published sweep, and the public lineage stopped short of it

`weights/full_frame_center/gate_threshold_metrics.csv`, written by the model's author:

```
 thr    n_pred   recall   precision       f1
0.10    38,272    0.900      0.0486   0.0922
0.15    33,372    0.856      0.0530   0.0998
0.20    28,678    0.814      0.0587   0.1094
0.25    24,029    0.756      0.0650   0.1197   <- where the public lineage stopped
0.30    18,944    0.675      0.0737   0.1329
0.40     9,133    0.392      0.0888   0.1448   <- f1 maximum
0.50     1,337    0.055      0.0845   0.0664   <- the gate collapses
0.60         9    0.000      0.1111   ~0
```

and `gate_summary.json`'s own note:

> *"Sparse-label precision/recall are calibration features, not complete-cell metrics."*
> *"**Use high precision thresholds as conservative node-rescue gates.**"*

The public progression moved `DEEPCENTER_SAFE_DIV_THRESHOLD` **0.12 → 0.25** and that was
half of the 0.939 → 0.941 step. The author's own curve keeps improving to **0.40** and dies
by 0.50. This is exactly `notes/65` §3's *stepped* category, now with the author's data
naming the next step rather than us guessing one.

**And there is a second consumer of the same gate that has never been touched.**
`DEEPCENTER_GAP_THRESHOLD` is `0.25` in **every** public notebook mined — it sat in the
identical column of the config matrix, not the varying one. Gap-closing acts on a far larger
population than divisions do, so if the calibration argument holds anywhere it should show
there first. Two arms: `dc40` and `dcgap40`.

## 2. `checkpoint_last` is the overfit checkpoint, not a better one

`weights/full_frame_center/history.csv`, 500 epochs:

```
epoch      train_loss    val_loss
    1        0.02214      0.04544
    2        0.01233      0.04500   <- val_loss minimum; this is best.pt
  ...
  439        0.00778      0.32969   <- worst
  500        0.00772      0.31674   <- this is checkpoint_last.pt
```

`best.pt` is **epoch 2**. `checkpoint_last.pt` is **epoch 500**, with a validation loss
**seven times worse**, and the config that loads it still asserts
`BIOHUB_DEEPCENTER_EXPECTED_EPOCH = 2`.

That reframes the arm this repo was most excited about two hours ago. `ckpt948` — the
kernel claiming LB 0.948 whose only distinguishing feature is this checkpoint — is swapping
a model chosen at its validation optimum for one trained 498 epochs past it. `lb941last`,
which puts the same checkpoint on the good base, is **deprioritised** for the same reason.
Both stay in the registry: `ckpt948` is already running and will at least say whether the
epoch guard fires.

It also independently confirms hengck23's observation (`notes/58`) that sparse labels
overfit almost immediately. Here it is two epochs, not ten.

## 3. `notes/24`'s "unknowable split_0 membership" is answered

```
deepcenter   split_manifest.json   all 199   train 71   val 128   seed 2026
temporal     split_manifest.json   train 199   test 40 (all inside the 199)
```

The temporal model trained on **all 199** annotated videos, and its own "test" 40 are a
subset of them. This matches topic 730160's claim on the forum.

It does not affect leaderboard validity — `notes/66` established the graded clips are
swapped in and are not any of these — but it does close the last door on local
cross-validation: any local score using these weights is scoring a model on its own
training data. `notes/66` killed local scoring for one reason; this is a second,
independent one.

## 4. What is not there

* **No second split.** Every artifact ships `split_0` only. A cross-split ensemble, which
  would have been the obvious way to buy variance reduction, is not available.
* **No alternative edge model.** `checkpoint_last.pth` (25 MB) sits beside
  `edge_predictor_best.pth` (8.4 MB) in both temporal artifacts, but the size gap says it is
  the full training state rather than an exported predictor — not a drop-in.

## 5. Arm queue after this

```
arm         base       change                                       standing
lb941       lb-941     none                                         running -- the floor
ckpt948     zhuzheng   none (checkpoint_last + narrow divisions)     running -- expectations cut
adaptive    rishabhr0y none                                         queued
dc40        lb-941     SAFE_DIV_THRESHOLD    0.25 -> 0.40           queued -- author's f1 peak
dcgap40     lb-941     GAP_THRESHOLD         0.25 -> 0.40           queued -- never moved by anyone
gap44       lb-941     GAP_CLOSE_UM          5.0  -> 4.4            queued
union       lb-941     SECONDARY_LINK_MODE   -> adaptive            queued
lb941last   lb-941     DeepCenter best -> checkpoint_last           deprioritised (§2)
```
