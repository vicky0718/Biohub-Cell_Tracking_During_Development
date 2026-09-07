# focus3d : one of the best 3d cell segmentation

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217
- **Topic id**: 738217
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-08-30T15:24:33.907558500Z
- **Votes**: 23
- **Comments**: 31

---

## Opening post

Here are the results i tried on one of the kaggle train dataset

https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff26b419d2743cbd3570314408b10e3f8%2FSelection_4730.png?generation=1788103458247804&alt=media)

---

## Comments (31)


### hengck23 (GRANDMASTER) — 2026-09-06T08:51:47.800Z — 2 votes

updated results

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F9e8d5f4c47591d7e21251755f284fcb8%2FSelection_4806.png?generation=1788684703562892&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F54fceef06b3911402346a7f5ebd371f6%2FSelection_4805.png?generation=1788684641113578&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T09:18:04.263Z — 1 votes

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F65e9183d4ded303e8bd71201170f37ad%2FSelection_4809.png?generation=1788686281728622&alt=media)
> 
> still feel that it is not good enough (does not help if gap is more than one missing frame) ... need to dream about it

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T10:48:52.160Z

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1fc32a4c8e74689f84192fd6b1dd5e9d%2FSelection_4810.png?generation=1788691729613470&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-05T19:41:24.460Z — 2 votes

The trick to transforming Focus3D annotation to Kaggle-like annotation  
1) Just pretrain with Focus3D annotation   
2) find matches (hit) of Focus3D and Kaggle zyx to compute diff dzyx  
3) have two heads: one to predict Focus3D zyx, another head dzyx, then Kagge zyx = Focus3D +dzyx  

hint: might as well let diff head predict values in um (instead of quantized 64x64x64 coord) 

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fe52d3a2127e86cfcfcf93c3c78f9cf1d%2FSelection_4791.png?generation=1788637476144738&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F69f74bce646bd3f4c2ff78d45bb4e4e4%2FSelection_4792.png?generation=1788637813426997&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7c169ccb2d1d5567e7fc500bdf120d1c%2FSelection_4793.png?generation=1788637823261833&alt=media)

-- 

This is a classical trick for OOD adaptation with few data. We have a prior model, then we shift to a new domain using A*ptior + B

keywords: domain calibration, domain adaptation

#### ↳ Rishabh Roy (EXPERT) — 2026-09-05T21:54:49.850Z — 1 votes

> why not use ultratrack directly ?

### hengck23 (GRANDMASTER) — 2026-09-05T02:07:51.003Z

i basically solved the cell detection problem. During development, chatgpt offers some interesting solutions: 2d to 3d:  
paper:  
1.  u-Segment3D — “Universal consensus 3D segmentation of cells from 2D segmented stacks”.  
https://github.com/DanuserLab/u-Segment3D   
https://www.biorxiv.org/content/10.1101/2024.05.03.592249v3  

2. Seg2Link: an efficient and versatile solution for semi-automatic cell segmentation in 3D image stacks
https://github.com/WenChentao/Seg2Link

### hengck23 (GRANDMASTER) — 2026-09-04T13:26:45.227Z — 1 votes

experiment on training a unet3d with dense cell centroids from focus3d + kaggle annotation:  
https://www.kaggle.com/code/hengck23/cell-point-detector  
(6 sec per volume on one T4 gpu) 


here is recall rate on kaggle node annotations:  
input 64x64x64 (one volume)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Facdf250b20a9f3efcc3328dbdcbe14e4%2FSelection_4771.png?generation=1788528152996386&alt=media)

plan:
- (1) train on external data (there are many opensource 3d zebrafish embryo cells, especially those from biohub) 
- (2) i haven't apply tricks like augmentation, SWA weight averaging, etc ...
- (3) another ranker head to push probability upwards so that i can have less predicted nodes
- (4) maybe a head to predict node density so that i can predict kaggle estimate node count (metric hack)  


(3),(4) may not be necessary, because tracking can recover missing cells (or adaptively adjust prob threshold)

---

i divide the problems into steps:
1) train a good cell detector first  
2) then get cell detector feature (+ modify vector) to make link transformer  
3) if you analyse post-processing, you will find that they use heuristics to join cells if the broken gap interval is   small. This means that if we train link transformer with window >2 (eg like 5) results is better. i.e. learn the "joining" instead of heuristics. but the risk is that data lis limited and maybe heuristics is better?

#### ↳ hengck23 (GRANDMASTER) — 2026-09-04T13:39:53.937Z

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F2d7f3d5429dda866877f5b1331afb2c6%2FSelection_4773.png?generation=1788529191797128&alt=media)
> 
> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F834de770caf26c33624489fbcef64537%2FSelection_4774.png?generation=1788529381336076&alt=media)

#### ↳ ↳ Rishabh Roy (EXPERT) — 2026-09-04T16:15:45.133Z

> > This is awesome . Thanks for sharing your findings . Would love to implement this . Will try if this fits under kaggle 12 hour window run

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

### hengck23 (GRANDMASTER) — 2026-09-07T03:00:29.040Z

another idea
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F17a1759af0e5e5ef2acfad20a57f84c7%2FSelection_4813.png?generation=1788750027595991&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-06T11:46:51.953Z

Another trick, but maybe will overfit if you don't have sufficient data. Kaggle annotations may not be best for tracking. Let gt be the Kaggle annotation; you can refine ground truth to gt+dzxy so that it is still within the 7um error limit but drastically improves link probability. Also coord could be subpixel and use F.graid sample to sample feature.

### hengck23 (GRANDMASTER) — 2026-09-06T03:42:42.483Z

training usually dense FOCUS3d annotation + frame/augmented frame actually works.  
with dense node and edge (pairing) annotation, i can train up to 200 epochs without overfitting.

I design my own transformer following the SuperGlue framework for keypoint matching: alternating self-frame attention and cross-frame attention.

validation: unseen sample_id + kaggle annotation:

```
6bba_337b1b3a
division excluded in this test

{'num_gt_nodes': 1272, 'num_matched_nodes': 1272, 'node_recall': 1.0, 
'num_gt_edges': 1209, 'num_edges_both_nodes_matched': 1209, 'num_correct_edges': 1166, 
'edge_recall_end_to_end': 0.9644334160463193, 
'edge_recall_given_nodes': 0.9644334160463193, 
'mean_edge_rank': 0.060891938250428816, # e.g. top1, top2 ... 
'mean_edge_prob': 0.9515399047913187}
```

I have chatgpt to do all the coding, while i check. i think this can be automatic once i get new external data. 



more visualisation and code coming up. !!!!
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb8ffc8ecd5ee733ab8d0144baaca33d0%2FSelection_4796.png?generation=1788669728505646&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T03:59:31.693Z

> The implication is that you do not need kaggle annotation to train. So you can do online training on hidden data in theory

### YanngYT (CONTRIBUTOR) — 2026-09-04T04:39:39.543Z

I tried to use focus-3d to segment cells, but it timed out when submitting. Are you doing dense segmentation first when tracking? Or is it just using focus-3d to generate a dense annotation set for training tracking?

#### ↳ hengck23 (GRANDMASTER) — 2026-09-04T04:58:02.320Z

> First verify the recall error of focus3d on kaggle nodes after trying several parameters search. If you are satisfied, train link/track transformer on focus3d zyx(eg centroid of instance label)
> 
> Finally train a point predictor using simple unet to distill focus3d results.
> 
> 
> Getting the point is usually not the issue.  But we want to minimise no of predicted nodes with near 100% recall rate

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

### unknown — 2026-09-03T13:38:57.213Z

*(empty)*
