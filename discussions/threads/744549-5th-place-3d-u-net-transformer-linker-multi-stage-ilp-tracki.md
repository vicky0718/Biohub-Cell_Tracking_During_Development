# 5th Place: 3D U-Net + Transformer Linker + Multi-stage ILP Tracking

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744549
- **Topic id**: 744549
- **Author**: Tang (MASTER)
- **Posted**: 2026-09-30T08:35:33.326167700Z
- **Votes**: 10
- **Comments**: 0

---

## Opening post

## Overview

The pipeline has three parts:

- **Detector**: a 3D U-Net that finds the cells.
- **Linker**: a transformer that links cells between two frames.
- **Tracking**: ILPs that build the tracks.

The detector and the linker are used as a 5-fold ensemble with TTA. The outputs of the 5 models and the 8 TTA views are averaged.

## Validation design

5-fold CV over clips, stratified by clip type (44b6 or 6bba) and by the number of divisions per clip.

The training data has two kinds of problem frames:

- **Duplicated frames**: a frame is exactly the same as the next one.
- **Drift frames**: the whole image jumps between two frames, as in the figure.
<p align="center">
  <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2312281%2F3a2a5bd402f9b0594e83dd4c0c8b2dee%2F1_drift.png?generation=1790757018703959&alt=media" width="600">
</p>
<p align="center"><em>Drift between two frames. After shifting back by 14 µm, the cells match.</em></p>

In validation, I removed the links of these frames from both the prediction and the ground truth. My drift fix gave a big gain in CV, but no gain on the LB. So the hidden test data has no large drift. I think this is also one reason why many people saw CV and LB that did not match.

In my experiments, training without the duplicated frames and with the drift corrected raised the private score of the same fold-0 model by 0.011, but the public score dropped. So I did not use it in the final model.

## Model

- **U-Net**: the temporal 3D U-Net from the [public baseline](https://www.kaggle.com/code/thibautgoldsborough/unet-baseline-inference-submission). I use average pooling to make each frame 4× smaller in X and Y before the U-Net.
- **Linker**: a transformer on the cells of frames t and t+1 (self-attention inside each frame, cross-attention between the two frames). It learns four things:
    - **Link score**: for every pair of cells in t and t+1, is it the same cell?
    - **Identity similarity**: do two cells look alike? Each cell gets a 16-dim appearance vector from the U-Net features, and the same cell in t and t+1 should have similar vectors. This helps to separate cells that are close together.
    - **New cell**: is this cell in t+1 new, with no matching cell in t (for example, it just came into view)?
    - **Division**: does this cell in t divide into two? Divisions are rare and easy to miss, so a separate head, with a higher weight on division examples in the loss, gives a direct signal for them.

<p align="center">
  <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2312281%2F91f30a25f1c7c6fe17ac80ec71acee34%2F2_linker.png?generation=1790757225545384&alt=media" width="700">
</p>
<p align="center"><em>The linker. Top: structure. Bottom: its three outputs on a small example, where cell B divides into 2 and 3, and cell 4 is new.</em></p>


- **Pretraining**: I first trained on 4 public ZebraHub datasets. Then I fine-tuned on the competition data.
- **Training**: AdamW, lr 1e-4, cosine schedule, batch size 8. Augmentation: flips and rotations in XY, gamma and brightness shift.

## Tracking

An ILP picks which cells and links to keep, and where tracks start, end or split. Confident cells, high-score links and links between cells with similar identity vectors are cheaper to keep. A track start is cheaper when the new-cell score is high, and a split is cheaper when the division score is high.


<p align="center">
  <img src="https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2312281%2Faad0ac87f3def95d70775d9b1c24743c%2F3_tracking.png?generation=1790757170783889&alt=media" width="800">
</p>

1. **ILP1**: solve with confident cells only (detection confidence p ≥ 0.9).
2. **ILP2**: solve again with more cells (p ≥ 0.6). It fills gaps, but also adds some false divisions.
3. **No new divisions**: remove every division that ILP1 did not have. The extra cell keeps its own track.
4. **ILP3 (re-link)**: keep the cells and divisions, and solve only the links again. Weak links (score 0.2 to 0.3) can help during the solve, and are removed after.
5. **Repairs**:
    - ILP4 joins a track end and a track start that are 2 to 5 frames apart, and fills in the missing positions.
    - Unclear links are matched again, using how similar the cells look.
    - A track that ends at frame t and one that starts at t+1 are joined by a weak link (score 0.1 to 0.3) if the cells look similar.
6. **Smoothing**: the detected positions jump around a little from frame to frame. For each cell, I fit a straight line through the positions of its track, from 4 frames before to 4 frames after. The new position is 80% the line and 20% the original position. After that, if a track misses one frame, I fill in the missing position.

## Results

The main improvements, as the change in LB score:

| Method | Public LB | Private LB |
|---|---|---|
| Linker: New cell and Division heads | +0.020 | +0.024 |
| Pretraining on ZebraHub | +0.014 | +0.016 |
| 5-fold ensemble | +0.008 | +0.007 |
| TTA with 8 flips and rotations | +0.007 | +0.005 |
| Tracking: detection confidence as the cell cost in the ILP | +0.008 | +0.020 |
| Tracking steps 2–3: ILP2 and No new divisions | +0.003 | +0.007 |
| Tracking step 5: Repairs | +0.001 | +0.002 |
| Tracking step 6: Smoothing | +0.004 | +0.005 |

---

## Comments (0)

*(none)*
