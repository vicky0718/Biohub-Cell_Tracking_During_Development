# Phase 1 — 13× the data made it worse, and the notebook saved the worst weights

`claude_detector_train`, Kaggle 2026-08-21, ~2.5 h on a P100. All 199 datasets, 25 frames
each — 1,775 volumes / ~5,032 annotated cells for the `44b6` fold, 3,200 volumes / ~28,259
cells for `6bba`. **All three pre-registered predictions FALSIFIED.**

## Held-out recall collapses while training loss goes to zero

| epoch | trained on `44b6` | trained on `6bba` |
|---|---|---|
| 4 | **0.8157** | **0.8106** |
| 9 | 0.7444 | 0.7933 |
| 14 | 0.7889 | 0.7485 |
| 19 | 0.7788 | 0.6362 |
| 24 | 0.7827 | 0.5706 |
| 29 | 0.7691 | 0.4636 |
| 34 | 0.7755 | 0.4958 |
| **39** | 0.7191 | **0.2654** |

Training loss over the same span: 0.00107 → **0.00000** and 0.00128 → **0.00000**.

The `6bba` model — the one with 28,259 annotated cells, our largest training set — ends at
**0.2654 recall against DoG's 0.7696**. It is worse than useless, and it got there
monotonically while its loss went to exactly zero.

## 1. 🚨 The notebook saved the final checkpoint, not the best one

It measured the curve above every 5 epochs and then wrote `model.state_dict()` after the
loop. So the `.pt` files on Kaggle are the **epoch-39** weights:

| fold | best checkpoint | saved checkpoint | cost of the bug |
|---|---|---|---|
| train `44b6` | 0.8157 (ep 4) | 0.7191 (ep 39) | −0.0966 |
| train `6bba` | 0.8106 (ep 4) | 0.2654 (ep 39) | **−0.5452** |

`claude_detector_score` would have loaded those and scored garbage — and, because the
champion is reproduced in the same run, it would have looked like a clean, well-controlled
refutation of the whole learned-detector idea. Measuring a selection signal and then not
selecting on it is worse than never measuring it.

## 2. ⭐ Why `masked` collapses, and why it is the loss's fault

A 350 k-parameter network should not memorise 28,259 cells this violently. The mechanism
is specific to the loss:

`make_loss_mask` keeps a voxel only if it is a positive, or **clearly empty** — low
intensity *and* low DoG response *and* far from any annotation. That keeps 26–37 % of
voxels. The remaining ~65 % — the ambiguous middle, which is exactly where unannotated
real cells live and where detection actually happens — **contributes no gradient at all**.

So the model can drive the loss to exactly zero by learning "bright blob → 1, dark
background → 0", a trivially separable problem. Having done that, **its behaviour on the
ambiguous middle is completely unconstrained**, and continued training lets it drift
anywhere. The loss cannot see the drift; held-out recall can.

That reframes `notes/16` §4. The masking idea was right that an unannotated cell must not
be labelled background. It was wrong to conclude the voxel should therefore be dropped:
excluding it removes the contradictory gradient *and* every constraint. `pu` does not have
this hole — it treats unlabelled voxels as a known mixture rather than ignoring them, and
in phase 0b its loss plateaued at 0.0573 instead of collapsing to zero.

## 3. More data did not help even before the collapse

| | phase 0b (240 volumes) | phase 1 best checkpoint |
|---|---|---|
| train `6bba` → eval `44b6` | **+0.0919** | +0.0410 |
| train `44b6` → eval `6bba` | **−0.0155** | −0.0619 |

At its *best* epoch, with 13× the data, phase 1 is worse than phase 0b on both folds.

So `notes/18` §4's headline — "the binding constraint is data volume" — is **not
supported**. The counting argument (590 vs 1,860 cells, and the small one being weak) was
suggestive, but the direct test says otherwise. Two things changed alongside the data
(batch 4 → 8, 25 → 40 epochs), so this is not a clean attribution either; what *is* clean
is that adding data did not rescue anything, and the loss's blind spot is now the better
explanation for everything.

## 4. CPU inference is fast enough — the offline submission path works

Measured on Kaggle: **215 ms per 64³ volume on CPU**, so a ~200-dataset test set at 100
frames is **1.20 h**. That fits inside the 12 h cap with room for linking and CSV writing,
which means the submission does not need a GPU and therefore does not need the P100 torch
workaround at all. One real risk retired.

---

## What to do next

1. **Select the checkpoint on held-out recall.** Non-negotiable, and it alone recovers
   +0.0966 / +0.5452 over what was saved. Evaluate every 2 epochs — the peak is at 4.
2. **Run `pu` at scale alongside `masked`.** §2 predicts `pu` degrades gracefully where
   `masked` collapses, because its loss constrains the ambiguous middle. That is the real
   test of the mechanism, not just a hyperparameter sweep.
3. **Fewer epochs.** 15 is generous when the peak is at 4.
4. Only then score end-to-end. `claude_detector_score` is built and tested but must not
   run against the current weights — it would score the epoch-39 collapse.
