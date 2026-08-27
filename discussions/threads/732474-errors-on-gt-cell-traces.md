# Errors on GT cell traces

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732474
- **Topic id**: 732474
- **Author**: Tim Krige (EXPERT)
- **Posted**: 2026-08-03T08:02:09.584495200Z
- **Votes**: 6
- **Comments**: 2

---

## Opening post

Hello all,

I have noticed a that quite a few annotated cells do not track the same cell as I do frame to frame.

I have made a gui to help me understand this data, and part of it is to show the GT annotations against my own selections (human, not ML) to see if I am able to track these cells. Frequently I am selection a cell that is not what the GT states is the correct cell. 

I find that I agree with my algorithms 95% of the time, and GT only 5% of the time in these cases, and I have seen quite a few of them. 

In some cases GT does not seem to see that the cell has split, or that the cell has moved slightly, but in most cases I am very confident that I have selected the same cell frame-to-frame. 

Has anyone else seen this? Is there a method to handle it?

---

## Comments (2)


### xaxipiruli (EXPERT) — 2026-08-11T10:03:41.530Z — 2 votes

My point of view if useful:

Two things worth keeping in mind

1. Distribution shift. The Zebrahub acquisitions differ from the competition data (imaging setup, embryos, developmental windows), so the raw image statistics won't line up one-to-one. Expect to need some domain adaptation rather than direct transfer.

2. The tracks are algorithm-generated, not manually curated. The *_tracks.csv lineages were produced with Ultrack (Bragantini et al., Royer Lab / CZ Biohub, Nature Methods 2025), an automated segmentation + linking pipeline. They're high quality, but they're effectively pseudo-labels: they carry Ultrack's own systematic biases and error modes, especially around cell divisions and densely packed regions. Only small subsets were manually corrected for validation, not the full atlas.

#### ↳ Tim Krige (EXPERT) — 2026-08-18T10:04:36.883Z

> Thanks! I have not used Zebrahub at all yet, so point 1 is not an issue in my pipeline, but good to mention it.
> 
> On point 2: Is there a source for this? I do not see that the competition explicitly states this.
> 
> In any case, is this competition attempting to track cells and their divisions, or to predict incorrect labels? With this dataset it seems like the GT errors are forcing my model to attempt to guess when a cell will be lost by GT rather than to predict accurate cell locations/divisions.
> 
> An example: each cells volume after a split should be 0.5 of the volume of the parent before the split. However, in GT this is not the case. On average, volume after split in GT is closer to 0.75. This shows that either I cannot measure volume (honestly possible) or GT did not correctly label the split. I labelled my own dataset and measure this same measurement at 0.5.
