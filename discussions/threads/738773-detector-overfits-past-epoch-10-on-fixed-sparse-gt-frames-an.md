# Detector overfits past ~epoch 10 on fixed sparse-GT frames -- anyone else see this?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738773
- **Topic id**: 738773
- **Author**: nusrati (CONTRIBUTOR)
- **Posted**: 2026-09-01T16:49:55.211798600Z
- **Votes**: 2
- **Comments**: 2

---

## Opening post

Training a 3D UNet detector on sparse-GT point annotations (fixed 8 frames/movie, no per-epoch resampling, r=1 masked focal loss). Real official-metric score peaks around epoch 5-10, then collapses with more training -- not gradually, by more than half (adj_ej 0.40 → 0.07 by epoch 120).

Loss keeps dropping the whole time (0.25 → 0.003). Node recall stays flat/high (~0.98-0.99) throughout. But peaks/frame keeps climbing (1800 → 2100+), and the node-count penalty term in the metric eventually goes negative from over-detection.

Read this as classic overfitting to a small fixed training sample -- model gets better at matching the exact 1432 training frames it's seen, worse at generalizing detection threshold behavior to new data.

Questions for anyone further along:

Did you see the same recall-flat/peaks-climbing pattern past a certain epoch count?
What fixed it for you -- resampling different frames per epoch, augmentation, early stopping on the real metric instead of loss, something else?
Roughly what epoch count / recall level did your detector actually converge at before it started hurting the real score?

---

## Comments (2)


### hengck23 (GRANDMASTER) — 2026-09-02T18:01:07.553Z — 1 votes

cyan is kaggle ground truth annotation. A lot of such nodes in faint intensity are causing problems

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff94359f9b05ea644a7a049d9e2732bec%2FSelection_4746.png?generation=1788372012504153&alt=media) 

![
](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F88e15383975f28ac477b2c7276ffe2ce%2FSelection_4745.png?generation=1788372031229506&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-02T01:35:36.987Z

this is the effect of sparse annotations. the unlabelled nodes "changes from noisy targets to negative targets after 10 epoches".
