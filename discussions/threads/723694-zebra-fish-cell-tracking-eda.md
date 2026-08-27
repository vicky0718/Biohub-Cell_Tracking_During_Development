# Zebra fish cell tracking EDA

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723694
- **Topic id**: 723694
- **Author**: NevilleAndrade (CONTRIBUTOR)
- **Posted**: 2026-07-07T20:27:00.074615800Z
- **Votes**: 1
- **Comments**: 2

---

## Opening post

Hi all. This is my first competition. Though not my area of expertise I really like to learn about this field. I understand that the geff files contain the annotations that the researchers have tagged on the cell. I created a Z-MIP (Z axis - Maximum Intensity Projection) of one of the datasets (44b6_0113de3b) shown below

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F346284%2F16441ba7414e137ba377fa38df577fa8%2FScreenshot%202026-07-07%20220625.png?generation=1783455221394141&alt=media)

Considering just the one embryo_id = 44b6_0113de3b is my understanding correct that one of the problems is to track this one annotated cell from the beginning frame 0 to 100 though the annotation is at the 50th frame?

---

## Comments (2)


### Thibgolds (CONTRIBUTOR) — 2026-07-07T22:17:02.137Z — 1 votes

Hi, the embryo_id is just 44b6, the other part is the crop id. You might have picked an example with a single annotation, but the vast majority of crops have many annotated cells, which can be at any time point (0 to 100). The task is to track ALL the cells in the video, not just the ones that we provide annotations for, at test time you are only evaluated on the tracks that we have annotated in the test set.

#### ↳ NevilleAndrade (CONTRIBUTOR) — 2026-07-08T05:02:18.677Z

> Thank you.
