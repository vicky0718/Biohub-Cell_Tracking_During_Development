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
