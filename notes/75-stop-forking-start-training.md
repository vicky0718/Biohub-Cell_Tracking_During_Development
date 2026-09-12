# Stop forking. The pack ships its own trainer and Kaggle already mounts the data.

Board of 2026-09-12, 3,424 teams. **We are rank 589 at 0.945**, down from 272 three days ago
on an unchanged score.

```
score    better   tied      band       a new entrant lands at
0.952        23      7     24-30                          31
0.949        43      9     44-52                          53
0.948        52     28     53-80                          81
0.947        80    252    81-332                         333      <- the public notebook
0.946       332    217   333-549                         550
0.945       549     54   550-603                         604      <- us
```

**252 teams sit on exactly 0.947.** That is one notebook, forked. The plateau moved +0.001 in
three days and swallowed 252 teams; our measured mechanism gains are +0.001 each and arrive
slower than the plateau moves. Forking is a treadmill that runs faster than we do, and every
arm we have left is another fork of the same lineage.

Tang (rank 5, 0.961) said it in August and it is now the only thing left that is true:
*"the current ckpt has kind of hit a wall... most of the gains come from model improvements."*

## What changes the game, and why it is actually available

`notes/33` closed retraining on arithmetic: ~333 h against ~140 h of quota. That was for
training **from scratch, for 400 epochs, on a T4**. Three facts collected since reopen it.

**1. The pack ships the training script.** `tracking_repo/scripts/train_unet_transformer.py`,
49,398 chars, pulled from our own kernel output. Its interface is everything we need:

```
--unet-weights   Path to pretrained UNet weights; loaded with strict=False.   <- fine-tune
--max-iters      Max training iterations per epoch. None = full epoch.        <- fit 12 h
--splits/--split a JSON of folds; default data_dir/dataset_splits.json        <- honest holdout
--epochs --lr --batch-size --window-size --downsample --det-loss-weight ...
```

`best_score = acc * recall` on the held-out split, saved to `edge_predictor_best.pth`.
**Fine-tuning is a supported entry point, not a rewrite.**

**2. Kaggle already mounts the training data.** The competition dataset contains
`train/<movie>.zarr/0/c/<t>/0/0/0` and `train/<movie>.geff/...` — 199 movies with their
ground-truth graphs, mounted read-only in any kernel that lists the competition as a source.
No 87 GB download, no Drive, no laptop. `notes/74`'s streaming design is unnecessary.

**3. The augmentation gap is real and documented in their own code.**
`scripts/augmentations.py` defines exactly two, and the trainer uses exactly those:

```python
from augmentations import brightness_augment, flip_augment
DEFAULT_AUGMENTATIONS = [brightness_augment, flip_augment]
```

No elastic deformation, no affine, no intensity noise beyond a brightness shift. hengck23
suggested elastic in July and nobody in this lineage has tried it, because nobody in this
lineage trains.

## The plan

1. **Train on Kaggle.** A non-submission notebook: mount the pack + the competition data,
   append an elastic (and intensity) augmentation, resume from `edge_predictor_best.pth`,
   train under `--max-iters` inside the 12 h ceiling. ~30 h/week of quota is ~5 runs before
   the deadline.
2. **Validate honestly, in-training.** Hold out ~30 movies from the fine-tune. The public
   checkpoint saw all 199, so the comparison is *biased against us* — which makes a win on
   the holdout real rather than a training-set artifact. This is the first honest validation
   signal this project has had; `notes/66` and `notes/72` §3 closed every previous attempt.
3. **Publish the weights as a dataset** and mount them in an inference arm.
4. **Compound rather than replace.** The pipeline already blends two models. A new checkpoint
   is a *third*, independent one, and `ttasec` is already the machinery for fusing an extra
   model's features properly. A forker cannot copy a checkpoint that is not published.

## The guard this needs

Elastic warping moves the image; the node coordinates must move with it or every label is
silently wrong — the worst failure mode available, because training would simply converge to
nonsense. The augmentation must assert, after warping, that intensity at the warped
coordinates is still consistent with a cell centre, and abort otherwise. Every lesson in
`notes/68`, `70` and the `ttadom` port says: build the check into the thing, not around it.

## The augmentation, verified before it was trusted

`torch` installed locally, so the deformation was unit-tested rather than reasoned about.

**Geometry, pinned first with a numpy simulation of `grid_sample`:**

```
blob in  at (y, x) = (12, 25)      constant field d = (3.0, -2.0)
blob out at (y, x) = ( 9, 27)      = coords - d
intensity at out[coords - d] = 1.000      at out[coords] = 0.000
```

`out(p) = in(p + d)`, so content at input `q` lands at output `q - d`: the coordinate update
**subtracts**. Adding would have been silent and catastrophic.

**Then the real thing, on synthetic movies with nodes at known coordinates:**

```
                        max shift   intensity at warped coords   at the ORIGINAL coords
unwarped                        —                      1.0548                        —
trial 0                   2.73 vox                     0.9425                   0.6773
trial 1                   2.62 vox                     0.9380                   0.5177
trial 3                   2.97 vox                     0.9343                   0.5434
```

and with the coordinate update deliberately disabled, the guard fires:
*"node intensity fell from 1.0548 to 0.5177 -- the coordinates did not follow the image."*

**The guard was wrong anyway, and the test showed why.** Raw intensity separates a correct
update from a broken one only when the background is near zero. Real frames are
quantile-normalised with a bright background, which pulls both ratios toward 1 and closes the
gap. Rewritten to measure **contrast** — node intensity over mean image intensity — and
re-tested across background levels:

```
background   correct: contrast before -> after    broken update
       0.0          73.433 -> 63.829                guard fired
       0.2           5.854 ->  5.301                guard fired
       0.5           3.023 ->  2.794                guard fired
       1.0           2.026 ->  1.910                guard fired
```

The correct update retains 87-93% of contrast at every background level and the broken one is
caught at every background level, including the one where the first guard would have let it
through. The run already in flight carries the first version; its geometry is verified, so
the guard there is insurance rather than the thing being relied on.

## The P100 cannot train this model, and the reason is not memory

Fourth failure, and the first that is not mine. Everything up to the forward pass worked:

```
P100 detected -- installing torch 2.5.1+cu121 ... rc=0
Training movies mounted: 199
Fold 0: 169 train / 30 holdout
elastic_augment installed; baseline eval added
Model parameters: 2,076,706
Loading train (169 datasets)...
RuntimeError: CUDA error: invalid configuration argument
```

The traceback ends in `torch/nn/modules/activation.py` line 1308 — **`MultiheadAttention.forward`**,
not the convolutions. `temporal_unet.py` reshapes to

```python
h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
```

where `S` is the entire downsampled volume, so the attention batch is `B * S` — tens of
millions of sequences. On sm_60 that launch exceeds a CUDA grid limit. It is **not** an
out-of-memory, which is why a smaller batch is a hope rather than a fix: the pack's authors
trained this on sm_80-class cards, where the attention path is a different kernel.

Two changes, both about making the lottery cheap rather than winning it:

* **Batch 8 -> 2.** The conservative read of one data point. Inference on the same card runs
  at `--unet-batch-size 4` and training holds activations for the backward pass as well.
* **Refuse a P100 in the first seconds.** `machineShape` is accepted and ignored on push
  (`notes/65`), so the accelerator can only be re-rolled — and a re-roll is cheap only if the
  run dies before materialising the repo and loading 169 movies, which is where the three
  minutes went. The guard raises `RuntimeError` (not `SystemExit`, which IPython swallows,
  leaving the kernel reported complete — the silent-pass failure this project keeps
  re-learning) with a message containing the phrase `run_arm.py` already retries on.

Everything else in the run is verified working: the mount carries 199 movies, the holdout
splits, the augmentation installs, the trainer patches apply, the model loads at 2,076,706
parameters against the 8.4 MB checkpoint. What is left is drawing a T4.

## Six consecutive P100s, so the model was made trainable on a P100 instead

The re-roll guard worked and the lottery did not:

```
attempt 1..6   error on Tesla P100-PCIE-16GB   (2,497 log chars each)
GAVE UP claude-train-elastic: 6 consecutive P100 draws
```

`MEMORY.md` already recorded eight consecutive P100s on 2026-09-06. This account draws P100s,
`machineShape` is ignored on push, and waiting for a T4 is not a plan.

**So chunk the attention instead.** `_TemporalAttention.forward` is

```python
h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)
h = self.norm(h)
h, _ = self.attn(h, h, h, need_weights=False)      # B*S sequences in one launch
```

with `S` the entire downsampled volume, so the attention batch is millions of sequences.
cuBLAS batched GEMM takes its batch count as a CUDA grid dimension **capped at 65535**, and
beyond that sm_60 answers `invalid configuration argument`. Slicing the batch is
*mathematically identical* — every sequence attends only across its own `T` timesteps, so
there is nothing between slices to lose — and that matters here because we are fine-tuning
pretrained weights and cannot afford the maths to change.

Verified rather than asserted, on 100,000 sequences:

```
sequences 100,000  chunks 4  max |full - chunked| = 0.000e+00      EQUIVALENT
```

and the patch itself was applied to the real `temporal_unet.py` locally before being pushed:
one match, and the result parses. T is 2, so the loop costs seconds an epoch and lowers peak
memory as well. The P100 guard is now informational.

Five failures, every one caught in under four minutes, and the run now gets: mount ->
holdout -> augmentation -> trainer patches -> model load -> forward pass.

## The fine-tune was not fine-tuning anything

Looking at the trainer to answer "can we add parameters" turned up something worse.
`--unet-weights` is loaded like this:

```python
unet = TemporalUNet3D(...)
state = torch.load(unet_weights, map_location="cpu", weights_only=True)
missing, unexpected = unet.load_state_dict(state, strict=False)
```

into the **bare backbone**. But `edge_predictor_best.pth` was saved from the whole
`UNetNodeTransformer`, so its keys are `unet.enc...`, `node_transformer...`, `edge_head...`.
Loaded into a `TemporalUNet3D` every one of those is *unexpected* and every backbone
parameter is *missing* — and `strict=False` turns a total mismatch into a silent no-op.

**The run would have trained from scratch while printing everything a fine-tune prints.**
The flag is not broken; it is for a UNet-only pretrain, and it is the wrong flag for this
file. This is `notes/68`'s trap — an argument that names something real, resolves to nothing,
and falls back quietly — in the one place where it would have cost days rather than a run.

Fixed by restoring the **full** model after construction, association head included, with a
guard that refuses to continue if fewer than half the checkpoint's tensors matched. The
number is printed either way:

```
FULL restore from ...: N/M tensors loaded, K left at init, U unused
```

## Adding parameters: what it costs

```
layers              out_ch   approx params  vs default   restore
[32, 64, 128]           32       1,388,544       1.00x   full
[48, 96, 192]           32       3,101,168       2.23x   NONE - from scratch
[64, 128, 256]          32       5,492,704       3.96x   NONE - from scratch
[32, 64, 128]           64       1,416,224       1.02x   NONE - from scratch
```

(ratios only; the real model is 2,076,706 parameters.)

Every widening changes tensor shapes, so the checkpoint stops matching and the restore guard
refuses — correctly, because a wider model started from random has to beat a **400-epoch**
checkpoint inside what is left of a 30 h/week quota, on a P100 that is slower than the T4 the
333 h estimate in `notes/33` was based on. The knobs exist (`BIOHUB_TRAIN_LAYERS`,
`BIOHUB_TRAIN_OUT_CH`, and `BIOHUB_TRAIN_ALLOW_SCRATCH` to override the guard) but that is a
different project, not a fine-tune.

**The capacity knob that keeps the weights is resolution.** `BIOHUB_TRAIN_DOWNSAMPLE` from
`1,4,4` to `1,2,2`: convolutions do not care about spatial extent, so every pretrained tensor
still loads, and the detector sees **4x the resolution in Y and X**. It costs ~4x compute and
memory and **zero** parameters. `notes/04` measured detection as essentially the whole
contest, and `ttaz16` — the only arm whose local numbers moved node recall and node count in
the right directions together — was also a detection change.

## The baseline line did its job on its first run

```
UNet weights: 64 missing, 136 unexpected
Model parameters: 2,076,706
BASELINE epoch -1 (public checkpoint, no training) | acc=0.0000 | recall=0.0000 | score=0.0000
```

**Every one of the 64 backbone parameters missing and all 136 checkpoint tensors unused** —
the prediction above, confirmed in the wild. The "public checkpoint" scored **zero** on the
holdout because nothing was loaded into it.

That is the entire argument for the baseline eval. Without it, an epoch reaching acc 0.6
would have read as a triumphant fine-tune, and we would have shipped a from-scratch model
trained for 30 short epochs believing it was a 400-epoch checkpoint plus elastic. The full
restore is now in place with a hard floor at half the tensors.

## The augmentation guard fired — and it was the guard that was wrong, twice

```
RuntimeError: elastic_augment: node-to-background contrast fell from 108.3021 to 63.1416
```

Contrast 108 means a near-black background, and the drop was 42% against a 35% threshold.
Not a coordinate bug: at `downsample = (1, 4, 4)` a cell is barely a voxel across in Y and X,
and bilinear resampling of a one-voxel peak loses ~40% of its amplitude however right the
coordinates are. My synthetic test used sigma=2 blobs and lost 13%, which is precisely why it
passed. **The test was easier than the data.**

Rewritten to compare the warped image at the **updated** coordinates against the same warped
image at the **original** ones. Both terms sit on the same interpolated image, so resampling
loss cancels and only the coordinate update is measured.

That rewrite was also wrong, and the test caught it: gating on the *realised* shift is
circular — an update that never happens leaves the shift at zero, the gate never opens, and
the check passes. Six of six cases silently missed. Gated on the displacement the field
**intended** at the nodes instead:

```
 sigma    bg    correct    skipped   sign-flipped
   0.7   0.0         ok     caught         caught
   0.7   0.3         ok     caught         caught
   1.2   0.0         ok     caught         caught
   1.2   0.3         ok     caught         caught
   2.0   0.0         ok     caught         caught
   2.0   0.3         ok     caught         caught
```

sigma 0.7 is the real regime. No false positives, both failure modes caught, at every
sharpness and background tested.

## The first honest measurement this project has ever had

```
FULL restore from .../edge_predictor_best.pth: 136/136 tensors loaded
BASELINE epoch -1 (public checkpoint, no training) | acc=0.9998 | recall=0.9692 | score=0.9690
```

136 of 136. And the number underneath it is worth more than the run: on **30 movies held out
of the fine-tune**, the public checkpoint's edge head scores **accuracy 0.9998 and recall
0.9692**.

**The edge classifier is saturated.** Precision is essentially perfect and the only headroom
is 3% of recall. Whatever separates 0.945 from 0.952 on the leaderboard, it is not the edge
model's raw ability to say yes or no about a candidate pair — which reframes "most of the
gains come from model improvements" as being about *detection*, not association. That agrees
with everything we have measured independently: `ttaz16`, a detection change, was the only
arm whose node recall and node count moved the right way together, and Soheil (rank 2) named
missing endpoint nodes as his dominant error.

It also prices the elastic fine-tune honestly. Three percent of recall is the whole target,
and some of that 3% is unannotated rather than missed.

## Three wrong guards, and the shape of the mistake

```
v1  before-warp vs after-warp          false positive: 108.3 -> 63.1, which is bilinear
                                       resampling of a one-voxel peak, not a coordinate bug
v2  warped@updated vs warped@original,
    gated on the REALISED shift        circular: a skipped update leaves the shift at zero,
                                       the gate never opens, six of six missed
v3  same, gated on voxels MOVED        circular again, for the same reason, and it also
                                       false-fired on a 0.78-voxel warp where both samples
                                       round to the same voxel
```

Every version failed the same way: **gate a check on a property of the output and the failure
you are checking for closes the gate on itself.** The fix is two checks with different gates.

* **A — did the update happen?** Gated only on the *field*, which the update cannot
  influence. Catches a skipped update at every warp magnitude.
* **B — did it move them the right way?** Needs the nodes in different voxels before contrast
  can see anything, so below about a voxel it abstains rather than accuses.

```
 max_shift  sigma    correct    skipped    flipped
       0.8    0.7         ok     caught     MISSED
       1.5    0.7         ok     caught     caught
       3.0    0.7         ok     caught     caught
       6.0    2.0         ok     caught     caught
```

The one MISSED is stated rather than hidden: under a sub-voxel warp a sign error is
undetectable by this instrument, and is bounded by A having already proved the update ran.
