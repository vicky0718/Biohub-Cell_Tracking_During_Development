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
