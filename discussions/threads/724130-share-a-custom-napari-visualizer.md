# Share a custom napari visualizer

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724130
- **Topic id**: 724130
- **Author**: Tom (MASTER)
- **Posted**: 2026-07-09T16:55:59.111354100Z
- **Votes**: 44
- **Comments**: 8

---

## Opening post

repo: https://github.com/tom99763/celltrack-studio 

* support uploading script to test your own post-processing
* know your score on this video
* check your matching between frame t to frame t+1
* observe your predicted trajectories and errors in 3d view 


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2Fcbfe42f675f3d938ab23b511046f224a%2Fnapari.png?generation=1783616306670559&alt=media)

---

## Comments (8)


### Tom (MASTER) — 2026-08-21T00:33:12.047Z — -1 votes

After 30 days suspension. I finally back

### hengck23 (GRANDMASTER) — 2026-07-10T04:23:54.487Z — 2 votes

there is a napari chatgpt plugin
https://github.com/royerlab/napari-chatgpt

seems that this competition is a good testbed for biomedical image agent
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb7dbb6b7dd0f585e8117890ea79fafe9%2FSelection_4357.png?generation=1783657431797421&alt=media)

#### ↳ Jordão Bragantini (CONTRIBUTOR) — 2026-07-13T20:18:07.787Z — 2 votes

> This interactive tool might also be useful, https://github.com/funkelab/motile_tracker

#### ↳ ↳ Tom (MASTER) — 2026-07-14T05:26:45.417Z

> > @jookuma  Tree representation is better for this challenge. Really thanks for sharing this I'll update my design

### Md Feroz Ahmed (EXPERT) — 2026-07-09T17:36:33.977Z

This is a really impressive and useful contribution! A custom napari visualizer for inspecting cell tracking results in 3D space and time is extremely valuable for debugging and improving models.

### Mr. Ahtasham Ul haq (CONTRIBUTOR) — 2026-07-11T14:43:46.227Z — -2 votes

This is the kind of tool sharing that moves a competition forward. Most public notebooks are just clones.

3D trajectory and frame to frame matching is the debugging step people skip, then wonder why their tracking model scores fine but looks wrong visually. Metrics on cell tracking can look okay while the actual trajectories are jumping between cells.

Does it handle two cells crossing paths where the model swaps their IDs? That failure is invisible in aggregate scores and obvious the second you watch the video.

### Joseph Adamski (CONTRIBUTOR) — 2026-08-24T15:51:58.507Z

Thank you this was super helpful.

### xaxipiruli (EXPERT) — 2026-07-10T00:08:26.873Z

Thanks, It was so useful :D
