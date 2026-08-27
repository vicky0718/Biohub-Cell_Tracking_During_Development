# beware of jumps in ground truth track

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724283
- **Topic id**: 724283
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-07-10T12:47:17.916832900Z
- **Votes**: 40
- **Comments**: 16

---

## Opening post

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F421105476d0308ddaf76f6be42659a9e%2FSelection_4367.png?generation=1783687617945117&alt=media)

At first, i thought not all ground truths are correct. Then, on inspection, i find the following are quite common
1) some frames are frozen. i.e subraction of vol at t and t+1 gives exactly zero
2) there are suddent jumps in time (global movement of whole volume)  

i.e. the time step are not constant, maybe due to limitations of acquisition equipment

---

## Comments (16)


### g john rao (MASTER) — 2026-07-11T05:05:58.217Z — 1 votes

this too falls under sparse annotations i think

> Annotations are sparse — not every cell in every frame is labeled.

and we can't reconstruct the entire embryo either

#### ↳ hengck23 (GRANDMASTER) — 2026-07-11T05:15:06.200Z — 2 votes

> Each volume is a crop. Maybe can reconstruct like jigsaw puzzle

#### ↳ ↳ g john rao (MASTER) — 2026-07-11T07:27:12.527Z — 2 votes

> > if the spatial crops are independently sampled time windows, we can only approximate the full embryo (stitching all samples together) reconstruction (not the lineage reconstruction, as that's the competition, only the FOVs matter)
> > 
> > What we have
> > 
> > Every training video provides:
> > 
> >   - An embryo prefix: 44b6 or 6bba
> >   - Exactly 100 frames
> >   - Volume shape 64 × 256 × 256
> >   - Frame spacing of 1 second
> >   - Voxel scale (1.625, 0.40625, 0.40625) µm
> >   - Raw fluorescence appearance
> >   - Sparse within-video tracks
> >   - An estimated total number of cell nodes
> > 
> > 
> > What is missing
> > 
> > The metadata does not provide:
> > 
> >   - Absolute acquisition time or developmental stage
> >   - Clip start time within the original recording
> >   - Global spatial/crop origin
> >   - Orientation or registration transform
> >   - Persistent cell IDs between clips
> >   - Whether two clips overlap or are consecutive
> >   - Any guarantee that all clips tile one continuous acquisition
> >   - Dense lineages or anatomical landmarks
> > 
> > 
> > edit: FOVs is the competition format - lack of computation for full embryo is the likely reason for it. but since i think each cell depends on every other cells in the embryo is the idea behind reconstructing the full embryo for a full picture view of everything.
> > 
> > although, there are many ways to stitch it together - but it will only be an approximation since we lack the full data needed.

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-11T08:03:34.213Z — 1 votes

> > It's somewhat similar to the temporal/sequencing issue in CMI3. Of course, this heuristic strategy can also compensate for the missing frames.

#### ↳ ↳ g john rao (MASTER) — 2026-07-11T10:48:14.990Z — 5 votes

> > okay, digging deeper, it is a systematic fault surely
> > 
> > Scanning every consecutive frame pair in all 199 training videos by comparing their stored compressed chunks byte-for-byte.
> > 
> > ```
> >    Group                  Videos    Videos affected    Duplicate pairs    Pair rate
> >   ━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━    ━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━   ━━━━━━━━━━━
> >    44b6                       71                  0                  0           0%
> >   ─────────────────────    ────────    ─────────────────   ─────────────────   ───────────
> >    6bba                      128                114                947        7.47%
> >   ─────────────────────    ────────    ─────────────────   ─────────────────   ───────────
> >    Entire training set       199                114                947        4.81%
> > ```
> > 
> >   - 57.3% of all training videos contain at least one frozen transition.
> >   - 89.1% of 6bba videos are affected.
> >   - Approximately one in every 13 adjacent pairs in 6bba is frozen.
> >   - No exact duplicates occur in 44b6.
> > 
> > 
> > Evidence that this is systematic
> > 
> > Several different 6bba samples have exactly the same freeze schedule. For example:
> > 
> >   6bba_05b6850b
> >   6bba_07477033
> >   6bba_5b28472a
> > 
> > all freeze after the same frame indices:
> > 
> > 4, 12, 27, 42, 52, 57, 59, 62, 66, 76
> > 
> > 
> > Likewise, these three share another identical schedule:
> > 
> >   6bba_1f58c2f6
> >   6bba_20852818
> >   6bba_80d12824

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-11T11:02:00.433Z — 1 votes

> > You analyzed it very thoroughly. I only started suspecting something during cross-validation—their adj_edge_jaccard values are around 0.5 and 0.7.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-07-11T11:04:20.747Z — 1 votes

> > This is a huge problem if test data is not the same as we learned and memorised the freezing unintentionally. It also means rule-based heuristics need to be changed to avoid overfitting etc.
> > 
> > “all freeze after the same frame indices:” this means they are cropped from the same master big volume… which i will be interested "to jigsaw it" back
> > 
> > ---
> > 
> > do we need to probe the hidden test frame for frozen frame?

#### ↳ ↳ g john rao (MASTER) — 2026-07-11T13:31:34.027Z — 1 votes

> > a simple conditional tracking would work too

#### ↳ ↳ Tom (MASTER) — 2026-07-11T15:55:51.073Z — 1 votes

> > Videos sharing the schedule	Frames duplicated after index…
> > 6bba_05b6850b · 07477033 · 5b28472a	4, 12, 27, 42, 52, 57, 59, 62, 66, 76
> > 6bba_1f58c2f6 · 20852818 · 80d12824	1, 14, 19, 43, 44, 45, 49, 57, 61, 66, 73
> > 
> > See html

#### ↳ ↳ g john rao (MASTER) — 2026-07-25T06:30:11.603Z — 1 votes

> > conditional copy wouldn't work because the GT differs as well 
> > 
> > details here: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/729082

#### ↳ ↳ Ace (EXPERT) — 2026-08-08T09:50:56.223Z — 1 votes

> > and what about the embryos types,  In training data there are two distinct types of Embryo, what if there is 3rd or fourth type, how well do I interpolate within embryos I've already seen," not "how well do I generalize to an embryo I haven't.

### Timmy Juicehouse (EXPERT) — 2026-07-11T07:50:56.430Z — 2 votes

I personally believe this is a fault in the imaging system causing repeated frames or dropped frames. Zebrafish embryo development is a continuous dynamic process. From the cleavage stage to the gastrulation stage, cells are constantly undergoing division, migration, and tissue deformation. Even in slower developmental phases, there are still sub-pixel-level cell displacements and fluctuations in signal intensity/grayscale. It is impossible for the entire 3D volume to show completely zero change, let alone produce perfectly repeated frames where subtraction yields exactly 0.

### victor (CONTRIBUTOR) — 2026-07-17T20:01:23.637Z

submission stuck gang report in

### emtcole (CONTRIBUTOR) — 2026-07-16T21:36:23.747Z

can coorberate this

### Angantyr (CONTRIBUTOR) — 2026-07-10T19:57:17.873Z

I wonder if this could come from a sudden movement of the entire setup?

Also, technically we should just let it be if the Ground Truth says so but a preprocessing of the signal/ground truth would be a nice addition.

### Tom (MASTER) — 2026-07-10T14:11:17Z

Some cells are natural long jumpers
