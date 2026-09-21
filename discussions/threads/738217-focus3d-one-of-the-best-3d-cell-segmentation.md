# focus3d : one of the best 3d cell segmentation

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/738217
- **Topic id**: 738217
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-08-30T15:24:33.907558500Z
- **Votes**: 38
- **Comments**: 50

---

## Opening post

Here are the results i tried on one of the kaggle train dataset

https://huggingface.co/spaces/Qinghua-thu/FOCUS-3D

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff26b419d2743cbd3570314408b10e3f8%2FSelection_4730.png?generation=1788103458247804&alt=media)

---

## Comments (50)


### hengck23 (GRANDMASTER) — 2026-09-10T00:40:24.790Z — 2 votes

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F53f2b08efba45290cfcb9e985cc2114f%2FSelection_4877.png?generation=1789000712859568&alt=media)

It seems that i cannot get Ultrack oversegment to work.   

When the cell is isolated and no oversegment is required, Ultrack centroid is close to Kaggle ground truth

#### ↳ Satwik (MASTER) — 2026-09-10T01:36:07.310Z — 1 votes

> Ultrack segmentation proved ineffective for me as well compared to just using our own detector. What Ultrack did seem to do well for me was generating dense tracks, using ultrack-td contours from FOCUS3D segmentations. I have not been able to acheive any meaningful result yet from any of these so far however.

### hengck23 (GRANDMASTER) — 2026-09-06T08:51:47.800Z — 4 votes

updated results

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F9e8d5f4c47591d7e21251755f284fcb8%2FSelection_4806.png?generation=1788684703562892&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F54fceef06b3911402346a7f5ebd371f6%2FSelection_4805.png?generation=1788684641113578&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T09:18:04.263Z — 2 votes

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F65e9183d4ded303e8bd71201170f37ad%2FSelection_4809.png?generation=1788686281728622&alt=media)
> 
> still feel that it is not good enough (does not help if gap is more than one missing frame) ... need to dream about it

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T10:48:52.160Z

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1fc32a4c8e74689f84192fd6b1dd5e9d%2FSelection_4810.png?generation=1788691729613470&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-07T03:00:29.040Z — 1 votes

another idea
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F17a1759af0e5e5ef2acfad20a57f84c7%2FSelection_4813.png?generation=1788750027595991&alt=media)

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

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-09T00:01:24.747Z — 1 votes

> > thanks. i did not know utlrack segmentation until i find this.
> > https://github.com/royerlab/ultrack-td/blob/main/examples/zebrahub.py

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

### hengck23 (GRANDMASTER) — 2026-09-05T02:07:51.003Z

i basically solved the cell detection problem. During development, chatgpt offers some interesting solutions: 2d to 3d:  
paper:  
1.  u-Segment3D — “Universal consensus 3D segmentation of cells from 2D segmented stacks”.  
https://github.com/DanuserLab/u-Segment3D   
https://www.biorxiv.org/content/10.1101/2024.05.03.592249v3  

2. Seg2Link: an efficient and versatile solution for semi-automatic cell segmentation in 3D image stacks
https://github.com/WenChentao/Seg2Link

### hengck23 (GRANDMASTER) — 2026-08-31T15:45:28.467Z — 4 votes

elastic augmentation  
so actually you have dense data for training  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F936529f234ae0f09dcafc7f0960ecc77%2FPeek%202026-08-31%2023-44.gif?generation=1788191083712245&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-01T09:19:47.130Z — 1 votes

An idea that is too much for the competition but could be feasible in long term cell tracking research. I have been looking at video generation deep net. You can have a depth map as prompt then generate anime or life movie.

So it is easy to create 3d virtual cell in blender and add motion. Then you can style it to create fluorescent microscopy volume.    

In fact with infinite data you can simply convert 4d to 4d end to end. From volume back to bender model.

#### ↳ nusrati (CONTRIBUTOR) — 2026-09-01T09:42:06.583Z

> yup super idea, but for an undergrad, thats not manageable in our routine. But I surely would like to contribute to it if someones upto it.

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

### hengck23 (GRANDMASTER) — 2026-09-20T08:19:52.773Z

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3797f41439de90d44f4b187b8a534e16%2FSelection_4977.png?generation=1789903449214112&alt=media)

expensive experiments. me trying to find a model that doesn't over estimate num of nodes

### hengck23 (GRANDMASTER) — 2026-09-16T11:17:17.323Z

unroll joint node and edge detection

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fdad2664b515e75c2e9f4fa97c7c687b6%2FSelection_4971.png?generation=1789557435322018&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-16T11:19:20.290Z

> in the same principle, we can unroll 5 frame prediction from 4xpairwise results
> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F9d8ee53206a7d71c3372c34a4e393002%2FSelection_4972.png?generation=1789558129242359&alt=media)
> 
> softamx is used as ranking loss. hence target is not single class. So this is naturally a listwise ranking problem. target is rank = kaggle metric score

### hengck23 (GRANDMASTER) — 2026-09-15T12:57:29.937Z

trick: the best way to reduce nodes is to cluster them and represent them by centeroid

### hengck23 (GRANDMASTER) — 2026-09-14T03:30:11.587Z

i find a trick. ultrack segmentation pt model gives foreground and boundaries probabilities, which are good for estimating "T\_est, estimated no of  nodes in a volume seq) in kaggle annotation.

### hengck23 (GRANDMASTER) — 2026-09-12T01:46:55.597Z

how to implement learnable ultrack-style multiple hypotheses?

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5b72f742a4302f2b31c6e134983716cd%2FSelection_4934.png?generation=1789177603466391&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F082ad8fb3c1d7b04b0b555ca312867b2%2FSelection_4935.png?generation=1789177613952710&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-12T01:50:58.050Z

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fe4d5de9bebc930e331310265e44785d5%2FSelection_4936.png?generation=1789177783358472&alt=media)
> 
> this is the key: selection of the best hypothesis in ultrack is not based on one frame, nor two frames  ... it is based on all frames (best trajectory)!
> 
> 
> How to implement differentiable IPL over window of say T=5,10 frames?

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-12T02:00:18.647Z

> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F06cc638f2647f7cef413f330c65ebbb3%2FSelection_4938.png?generation=1789178245293021&alt=media)
> > 
> > so both unet3d (stage1) and link trasnformer(stage2) are merely node and link proposal generators. we need a third stage to create trajectories and evaluate all them at train time so that IPL score can become valley at the correct GT solution.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-12T02:07:29.700Z — 1 votes

> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F5207b7e083b045ed2d9424c55d68c19b%2FSelection_4940.png?generation=1789178847724226&alt=media)

#### ↳ ↳ Satwik (MASTER) — 2026-09-12T03:01:33.700Z

> > I tried following this path for a few days and I have some benchmarks I can share. A detector trained on dense FOCUS3D labels- recall 1 on held out 41 video validation set across both embryos. A transformer linker trained on sparse GT annotation scores about 0.809 edge jaccard on CV and about 0.83 on LB ( scores are after using Ultrack ILP) . My plan was to use Ultrack to generate dense tracks on FOCUS3D segmentation and distill it down to a simpler model that works with centroids ,  but FOCUS3D with Ultrack only got an edge jaccard of 0.7. I tried training a model on these dense edges, and added GT labels to ultrack pseudo labels and assigned a higher weight to GT tracks but that performed poorly as well. I believe detection in itself requires some temporal context or a learning signal from the downstream task to be able to effectively work.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-12T06:00:52.820Z — 1 votes

> > my e2e node detector and link transformer trained on dense focus3d annotation + augmented frames has: validation: edge jaccard 0.902/0.896 for without/with ILP(my version).
> > 
> > on train set, it is about +2.

### hengck23 (GRANDMASTER) — 2026-09-10T10:43:51.150Z

let's try again. see if repo is detailed enough to repeat segmentation results...

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbd63a659a989989be36b9560f933eebf%2FSelection_4894.png?generation=1789036983651376&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-10T04:34:37.563Z

need to set uncertainty weights

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F480ff2dcc663fba1d8ee7404f528f983%2FSelection_4880.png?generation=1789014846921779&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-10T03:20:56.770Z

My friend said my approach was wrong. There are ambiguities and there is inly partial labels. Instead of learning perfect predictors, the focus should generate hypothesis and test.eg different way to link up assume with and without division and score hypothesis

### hengck23 (GRANDMASTER) — 2026-09-09T20:08:13.220Z

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F785aea4f57ece422f1e995ca430d6e4d%2FSelection_4870.png?generation=1788984621673535&alt=media)

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

#### ↳ hengck23 (GRANDMASTER) — 2026-09-06T03:59:31.693Z — 1 votes

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

### unknown — 2026-09-12T12:23:58.180Z — 1 votes

*(empty)*

### unknown — 2026-09-03T13:38:57.213Z

*(empty)*

### unknown — 2026-08-31T10:05:52.303Z — 1 votes

*(empty)*
