# The P100 escape hatch works, and the 0.948 kernel's one change never happened

`claude-arm-ckpt948` completed in 39 minutes **on a Tesla P100**, which settles both of the
things it was run to settle — one of them not the one it was run for.

## 1. The wheelhouse prologue works. P100 is no longer a blocker

```
P100 detected -- installing torch 2.5.1+cu121 from /kaggle/input/claude-torch-wheelhouse/wheels
wheelhouse install rc=0
...
Wrote /kaggle/working/submission.csv with 234,691 rows
```

`notes/65` §1 recorded eight consecutive P100 draws, `machineShape` ignored under both
spellings, and every run dying ninety seconds in. The prologue detects the card from
`nvidia-smi`, installs torch 2.5.1+cu121 and its cu121 dependency closure from the mounted
wheelhouse, and the run proceeds. It costs a few minutes and is a no-op on a T4.

This also removes `notes/66` §4's worry in its most acute form: a P100 run of the
verification set finishes in 39 minutes including the install, and the graded rerun is ~17×
that work, so roughly 11 hours — inside the 12-hour limit, but **not comfortably**. An arm
that adds work on top of the base is worth watching for that.

## 2. `ckpt948`'s distinguishing change did not take effect

The kernel claiming LB 0.948 differs from the 0.941 line in one visible way: it points
`BIOHUB_DEEPCENTER_CHECKPOINT` at `checkpoint_last.pt` instead of `best.pt`. Its own run log:

```
"deepcenter_checkpoint_default": ".../weights/full_frame_center/best.pt"
Trying DeepCenter add-only gate checkpoint: .../full_frame_center/best.pt
Loaded DeepCenter add-only gate checkpoint: .../full_frame_center/best.pt
DeepCenter checkpoint epoch: 2   best_score: -0.04500306242456039
```

**It loaded `best.pt`.** The env var names a path — `/kaggle/input/biohub-deepcenter-unet3d-
center-prior-v1/...` — and the resolved mount is `/kaggle/input/datasets/pilkwang/biohub-
deepcenter-unet3d-center-prior-v1/...`. The path does not exist, the notebook falls back to
its own default resolution, and the default is `best.pt`. Epoch 2, `best_score` −0.04500 —
exactly the `history.csv` row `notes/67` §2 identified as the validation minimum.

So the whole `checkpoint_last` branch — nine public kernels, one claiming 0.948 — is running
the same DeepCenter weights as everyone else. Whatever that kernel scores, it is not scoring
a different checkpoint.

**Three consequences.**

* **`lb941last` is a confirmed no-op**, not merely a bad idea. It makes the same edit to the
  same variable and it will resolve the same way. Removed from the queue rather than
  deprioritised.
* **`ckpt948` is not a 0.948 configuration.** With the checkpoint change inert, what remains
  is `DET_THRESHOLD 0.965` on an otherwise 0.934-era config: `SAFE_DIV_MAX_UM 7.0`,
  `SISTER_MAX_UM 12.0`, no divergence gate, no symmetry gate,
  `DEEPCENTER_SAFE_DIV_THRESHOLD 0.12`. It is a *worse* config than `lb941`, published with
  a better number. It stays in the record as the reason not to trust a kernel title.
* **Every arm's edit has to be read back out of the run log.** This notebook family dumps
  its fully resolved config as JSON, and that dump — not the notebook source — is what says
  whether an edit landed. `notes/60`'s trap (a parameter moves, a cap binds first) has a
  sibling here: a parameter moves and the reader never sees it.

That last one is a check, not a worry: the arms that edit `BIOHUB_*` *values* rather than
*paths* — `dc40`, `dcgap40`, `gap44`, `union`, `dse44`, `dse52`, `det960` — are read from
env directly and appear in the same dump. Each will be verified against it before being
recommended for a slot.

## 3. Node counts

```
claude_fork      118,659 nodes   114,382 edges    LB 0.937
claude-arm-ckpt948  119,426        115,265        (verification mode, unscored)
```

Within 0.6% of the fork on the placeholder clips, which is consistent with §2: nearly the
same pipeline.
