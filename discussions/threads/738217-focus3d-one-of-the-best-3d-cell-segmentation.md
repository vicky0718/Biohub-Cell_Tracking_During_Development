# focus3d : one of the best 3d cell segmentation

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217
- **Topic id**: 738217
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-08-30T15:24:33.907558500Z
- **Votes**: 17
- **Comments**: 15

---

## Opening post

Here are the results i tried on one of the kaggle train dataset

https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff26b419d2743cbd3570314408b10e3f8%2FSelection_4730.png?generation=1788103458247804&alt=media)

---

## Comments (15)


### hengck23 (GRANDMASTER) — 2026-09-01T09:19:47.130Z — 1 votes

An idea that is too much for the competition but could be feasible in long term cell tracking research. I have been looking at video generation deep net. You can have a depth map as prompt then generate anime or life movie.

So it is easy to create 3d virtual cell in blender and add motion. Then you can style it to create fluorescent microscopy volume.    

In fact with infinite data you can simply convert 4d to 4d end to end. From volume back to bender model.

#### ↳ nusrati (CONTRIBUTOR) — 2026-09-01T09:42:06.583Z

> yup super idea, but for an undergrad, thats not manageable in our routine. But I surely would like to contribute to it if someones upto it.

### hengck23 (GRANDMASTER) — 2026-08-31T15:45:28.467Z — 3 votes

elastic augmentation  
so actually you have dense data for training  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F936529f234ae0f09dcafc7f0960ecc77%2FPeek%202026-08-31%2023-44.gif?generation=1788191083712245&alt=media)

### Russell Kirk (EXPERT) — 2026-08-31T10:05:52.303Z — 1 votes

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F27681700%2F61aba3aa541a88660eb267eb7c12608a%2FScreenshot%20from%202026-08-31%2006-00-18.png?generation=1788170628363876&alt=media)


<--i want to share pictures too :D

### hengck23 (GRANDMASTER) — 2026-08-31T07:41:29.500Z — 1 votes

https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1  
ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy  

another shortcut is :  
FOCUS3d --> label -->augmentation (e.g. affine, elastic deform) to create window of T=2 pairs.  
then you can train link transformer etc..

#### ↳ Tom (MASTER) — 2026-08-31T09:44:45.063Z

> @hengck23  It looks like there are even more great ideas to me now

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-08-31T10:34:59.190Z

> > you can just randomly make some grid points that are non-background, then "track/link them" in next "augmented frame" as pretraining or aux loss

### hengck23 (GRANDMASTER) — 2026-08-31T04:04:38.893Z — 1 votes

FOCUS-3D (instance segmentation) --> HOCT (tracking)    
https://github.com/royerlab/hoct/tree/main  
https://arxiv.org/abs/2607.11754  
Higher-Order Cell Tracking Transformer  


---


THICK BLUE: kaggle annotation  
OTHER THIN: nearest track from HOCT  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F335a3eeb48ab9d2b799e6bf2c0ed973a%2FSelection_4733.png?generation=1788148939168370&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F6ae538c0de22bd26ba2f5bc2c8a2f1d7%2FSelection_4734.png?generation=1788148953830155&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F07a0f1d2c25bad1d915b127224cf6e71%2FSelection_4735.png?generation=1788148976594585&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F13a5e112eb312200b3bc018e42c7e078%2FSelection_4736.png?generation=1788149017218352&alt=media)

### hengck23 (GRANDMASTER) — 2026-08-31T00:34:08.573Z — 1 votes

Once you have dense segmentation label, you can use many opensource tracker like hoct, itec, trackastra to make dense tracks for better training.

Then you can do longer range tracking over window of 5 or 8 (instead of 2)

#### ↳ Rishabh Roy (EXPERT) — 2026-08-31T09:05:24.563Z

> Are you able to use this segmentation in your code ? @hengck23

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-08-31T09:50:37.440Z

> > You can download hf spaces gradio code and modify from there. It is self contained

#### ↳ ↳ Rishabh Roy (EXPERT) — 2026-08-31T10:37:22.100Z

> > would love to see this work

### Qiwei (MASTER) — 2026-09-02T09:56:12.787Z

This is a draft version, where FOCUS‑3D is only used as a detector directly：
https://www.kaggle.com/code/qiweiyin/focus3d-nuclei-physical-pp-submit?scriptVersionId=346624807

#### ↳ hengck23 (GRANDMASTER) — 2026-09-02T12:49:19.523Z — 1 votes

> You should use focus3d, then measure
> 1. Hitrate of sparse annotation ( also distance error)
> 2. Compare num of detected nodes with estimated number of nodes
> 
> ```
>     geff_meta = GeffMetadata.read(
>         zarr_file.replace(".zarr", ".geff")
>     )
> 
>     est_num_nodes = float(
>         geff_meta.extra["estimated_number_of_nodes"]
>     )
> 
> ```
> —-
> 
> Also you should evaluate link transformer or other link model given gt location + other location and compared detected location + other location.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-02T12:51:42.613Z — 1 votes

> > Further, i think gt annotation must have used some open source cell instance detector. I suspect it it cellpose3d or stardist3d with manual collection.
