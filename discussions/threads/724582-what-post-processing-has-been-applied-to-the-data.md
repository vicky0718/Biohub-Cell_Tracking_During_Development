# What post-processing has been applied to the data?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724582
- **Topic id**: 724582
- **Author**: Rohan Asokan (CONTRIBUTOR)
- **Posted**: 2026-07-12T06:04:10.640986300Z
- **Votes**: 3
- **Comments**: 3

---

## Opening post

Is the data raw intensity values at each z layer, or has there been some normalization or sharpening operation performed intentionally or unintentionally by the recording device?

---

## Comments (3)


### Jordão Bragantini (CONTRIBUTOR) — 2026-07-13T14:57:44.790Z — 1 votes

The microscope acquired multiple views of each volume, which were then fused.
There may be small intensity deviations between views, so they were linearly scaled to match a reference view.
In case you're interested, the microscope is custom-built and based on this design, https://www.janelia.org/sites/default/files/Library/Tomer%202012_0.pdf

#### ↳ Rohan Asokan (CONTRIBUTOR) — 2026-07-13T16:04:17.657Z

> This was a cool rabbit hole to go down into. Thanks! There are some intensity patterns I have noted along slices of individual data, so was wondering how the exact process might work.

#### ↳ ↳ Rohan Asokan (CONTRIBUTOR) — 2026-07-13T16:21:15.300Z

> > If you find the time, could you please share a rough diagram of the mechanical setup of optics? This would help remove some setup dependant noise and data quality degradation.
