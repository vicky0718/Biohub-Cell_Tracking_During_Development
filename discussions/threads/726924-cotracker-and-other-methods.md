# Cotracker and other methods

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726924
- **Topic id**: 726924
- **Author**: Bharat (CONTRIBUTOR)
- **Posted**: 2026-07-17T04:31:05.227909200Z
- **Votes**: 5
- **Comments**: 11

---

## Opening post

I came across Cotracker which is small and trained on tracking a point on videos. Though it's not 3D but I can give it a try. https://co-tracker.github.io/
There are also other methods like https://omnimotion.github.io/ .
Wanted to know if anyone tried them?

---

## Comments (11)


### Jawad Ahmed (CONTRIBUTOR) — 2026-08-06T10:07:20.853Z — 1 votes

One thing worth checking before investing in a 3D adaptation: how much of your score is actually lost to linking versus detection? I measured the failure split on a per-movie basis and the answer surprised me. Once node recall is high, nearest-neighbour on physical distance alone already resolves the large majority of links, and the residual errors concentrate in a small subset where two candidates sit within roughly one cell radius of each other.

That matters for CoTracker specifically. Its advantage is temporal coherence across long trajectories, but the cases that are actually costing points are short-range ambiguities inside a single frame pair. Group motion helps when a cell is lost for several frames; it doesn't obviously help when two neighbours are both 2 µm from the parent in the same step.

#### ↳ hengck23 (GRANDMASTER) — 2026-08-06T17:10:21.830Z — 1 votes

> i agree. co-tracking (or other tracking) alone will not win the competition. It is the lineage (cell division) that will decide the winner.
> 
> Another key is post processing to repair the tracks (this is like MNS/box adjustment/box filtering in object detection competitions)
> 
> my suggestion is to learn track without cell division first (i.e. all tracks has only one BIRTH and DEATH). cell dvision is then handled at post-processing or stage 2 (e.g. classifier to decide if there is a split based on appearance changes and longer track cues). it is difficult even for humans to decide if there is cell division just based on two frames.

### Vinayak Pathak (CONTRIBUTOR) — 2026-08-03T08:24:32.743Z

Cotracker may not help here I tried it, since it does not take into account the morphological changes that happen during division and the pixel to pixel mapping may be lost so you ,ay have to edit a bit of original architecture,

#### ↳ hengck23 (GRANDMASTER) — 2026-08-03T08:35:12.560Z — 2 votes

> Key of cotracker is to track all together, instead of tracking each individually. Cotracker did it by attention between all query temporally and spatially

#### ↳ ↳ Vinayak Pathak (CONTRIBUTOR) — 2026-08-03T09:17:33.917Z

> > Sure thing!

#### ↳ ↳ Vinayak Pathak (CONTRIBUTOR) — 2026-08-03T09:26:36.303Z

> > You are right, So basically should I just adapt the key principle of the COTracker and improve upon it I guess it would give me good result. Let me try and let you know. I guess I just started to use their code directly.

#### ↳ Bharat (CONTRIBUTOR) — 2026-08-03T09:53:48.500Z

> I found cotracker really good when I converted the 3d to 2d(mip along z) It is able to track the visible cells properly where the classical methods (LSA, motion, ILP) failed. Mostly during the long shifts in direction.
> But it won't work on 3d and we can't finetune it because the transformer's position awareness will break. We can try creating a similar arch for 3d. One thing we have to account for is that the current cotracker takes query points externally, and it is not aware of birth, death, divisions. It just tries to track the given query point in all frames (visible or not).

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-08-03T10:40:41.850Z

> > The bottom line is “do the tracks of others tell us about the target track”? Eg moving in groups or same patterns or relative spatial locations are preserved. If yes, the co-tracking all will help. Then to make it work, we need to have the correct formulation and data. We need to reformulate in sparse 3d and chatgpt etc can help

#### ↳ ↳ Bharat (CONTRIBUTOR) — 2026-08-03T10:57:50.603Z

> > you mean, we can give a grid of points and note their motion across time and use that(motion) to guide the classical method to assign edges?
> > Attached the video of a single track on 44b6_0113de3b. I gave the 2d video and the query point taken from geff file. It did track it and also handled the motion shifts.

### hengck23 (GRANDMASTER) — 2026-07-25T19:11:41.430Z

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F41e3cc42dc0de799feda946174d00330%2FSelection_4403.png?generation=1785006700020472&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Facb9f25bd185cb07f75dbb312208dd0d%2FSelection_4404.png?generation=1785006890371376&alt=media)

### hengck23 (GRANDMASTER) — 2026-07-17T10:44:46.163Z

https://arxiv.org/abs/2411.14833  
Cell as Point: One-Stage Framework for Efficient Cell Tracking  

this is cell tracking based on CoTracker3, RAFT etc . There is comparsion with Trackastra
