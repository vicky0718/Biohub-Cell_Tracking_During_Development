# Everything by hengck23 — 43 posts across 13 threads

Scraped from the competition forum, sorted by votes. hengck23 is a Kaggle
GRANDMASTER and the most technically substantive voice in these threads.


---

## [3v] focus3d : one of the best 3d cell segmentation
*comment d0 — 2026-08-31T15:45:28 — thread 738217*

elastic augmentation  
so actually you have dense data for training  

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F936529f234ae0f09dcafc7f0960ecc77%2FPeek%202026-08-31%2023-44.gif?generation=1788191083712245&alt=media)


---

## [2v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-17T04:35:11 — thread 723655*

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F20e5cb2adbba2365f79bbb79b5b5024e%2FSelection_4384.png?generation=1784262681228749&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1ed4a3ecc2a5dedc98fcdc44ba8ef0bd%2FSelection_4385.png?generation=1784263063174669&alt=media)


center fig: red is t=0, green is t=1   

Visualisation results from host repo baseline code (temporal unet + link transformer). i show max prob link.  
- long links are almost wrong (easy to filter such results)


looking at the results, link are short. maybe a conv  3d CNN (T as channel) unet to detection motion object is good? i.e. rather than detect zxy center, we segment "center of motion = line connecting zyx0 and zyx1."


---

## [2v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d1 — 2026-07-08T18:05:02 — thread 723655*

i have been playing with it. i think the winning formula is to generate dense tracks for training (1) and(2) below  
1) 3d point is easy to generate (e.g. opensource cellpose, etc)  
2) short track (2 frame or 3 frame) is easy to generate. it can be the utlrack, or rule base heuristics or open source tracker.  

if 1 and 2  can get good results, then ILP or min-cost graph-cut network will generate the long tracks required for submission.

currently the rule-based graph correcting post processor in public notebook should be used for generating new links in (2) for training. Imagine if i use 5 open source tracker, and using consistency, i can have more short tracks (currently we have less than 1% link labelled and the rest are unlabelled)

(2) only need locations. there are opensource zebrafish data with long (real and synthetic) dense tracks (lineage) without microscopy images that can be used too.


https://chatgpt.com/c/6a4e91cd-8d90-83ec-bf7a-825b27a9e284


---

## [2v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d2 — 2026-07-09T17:32:44 — thread 723655*

i change strategy a bit:
1) use opensource to make tracks, measure local LB score
2) ensemble opensource and own/ai heuristics,  measure local LB score
3) when i have good LB score, these become dense pesudo labels.

opensouce trackers are very good for short tracking. 
tricks:
- cellpose etc to provide dense 3d tzyx
- use only  tzyx for opensouce tracking. some opensouce like ultrack needs segmentation labels as input, i synthetically rendered 3d ball as input


---

## [2v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-10T17:35:26 — thread 723655*

solving the sparse annotation for cell center detection.     
1) observation : DoG peak detection somehow work. i.e., microscopy cell are blobs.    
2) instead of classification binary problem, we reformulate it as learnable peak detection. at each pixel location after unet logit head, loss = softmax of pixel over his neighbour.  those pixel without annotation are not computed in loss at all     

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7a94ffe3bbada2de65d720c1fead0bd7%2FSelection_4369.png?generation=1783705181623495&alt=media)

hint: ask chatgpt make a margin loss version: peak is at least T greater than neighbour


---

## [2v] Share a custom napari visualizer
*comment d0 — 2026-07-10T04:23:54 — thread 724130*

there is a napari chatgpt plugin
https://github.com/royerlab/napari-chatgpt

seems that this competition is a good testbed for biomedical image agent
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb7dbb6b7dd0f585e8117890ea79fafe9%2FSelection_4357.png?generation=1783657431797421&alt=media)


---

## [2v] beware of jumps in ground truth track
*comment d1 — 2026-07-11T05:15:06 — thread 724283*

Each volume is a crop. Maybe can reconstruct like jigsaw puzzle


---

## [2v] Cotracker and other methods
*comment d1 — 2026-08-03T08:35:12 — thread 726924*

Key of cotracker is to track all together, instead of tracking each individually. Cotracker did it by attention between all query temporally and spatially


---

## [2v] Question about the node-count adjustment in the metric (adj_edge_jaccard can exceed 1)
*comment d0 — 2026-09-02T14:14:33 — thread 739018*

yes. N_pred is a metric hack. From competition point of view: "it is better to detect just enough annotated nodes (up to N_est or less)" rather than all nodes". This change the way on how you treated unlablled data, apart from sparse labelled.

Obviously, the sparse annotation are not random tracks. they are difficult tracks (or tracks that are annotated by open source ultrack and hand corrected by host). you should expand sparse annotation to dense N_est annotation that is closest to sparse annotation. then train your model to detect N_est targets or less.

---

but of course, you can focus on division track and pay less attention on N_est. that is another separate strategy


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-16T04:42:56 — thread 723655*

i find a metric hack:  
you make a graph. if your just repeat your tracks (giving new id) your edge\_jaccard is not affected. this is the cause the kaggle metric don't penalize fragmentation, duplication. hence you can create multiple almost smiliar tracks to improve TP. But there is the node num correction (aka adj\_edge\_jaccard) But you can over come this by duplicating/perturbing at the correct length and correct location.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F0c8f49fc633511f2736c46579fa5b7f3%2FSelection_4383.png?generation=1784177205409163&alt=media)


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d1 — 2026-07-18T01:17:36 — thread 723655*

I think the node correction can both increase and decrease original edge jacard score?


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-10T13:29:32 — thread 723655*

Something struck  my mind. Since it is video and tracking, unsupervised pretraining should help. Hence we no need dense track or point


Tracking needs dense, spatially precise features. The newer V-JEPA 2.1 specifically targets dense, spatially structured and temporally consistent features ….  



V-JEPA 2.1 behaves like unsupervised optical flow in feature space,


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-10T03:07:03 — thread 723655*

3d cuda optical flow: https://github.com/yongxb/OpticalFlow3d  
3d optical flow kalman filter tracker : https://byotrack.readthedocs.io/en/latest/  


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff09cbac39ef7361221ceef47e9382955%2FSelection_4356.png?generation=1783652776657450&alt=media)

hint: use synthetic image or synthetic motion is eaiser to debug your code


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-09T23:39:28 — thread 723655*

Open-source optical flow tracker + Kalman filter. 
The visualisation is messy. anyone has a better suggestion?

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F96f536ce5845530631cda7458673780e%2FSelection_4347.png?generation=1783640249216751&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd9a4b36a962237c4eb72da62851d472f%2FSelection_4348.png?generation=1783640266346613&alt=media)


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d1 — 2026-07-10T02:30:20 — thread 723655*

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbac01c8b8fc39e9f60fddbe7aacec3d3%2FSelection_4351.png?generation=1783650518238328&alt=media)

results of ultrack and byotrack. They tend to disagree when there is "large predicted movement". this is because the optimization cost forces it to "always" link to something

"link cost < termination cost + new-track cost"


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d2 — 2026-07-10T11:07:44 — thread 723655*

in theory, linking can be done by CNN.

i illusrate with a 1d example:  the axes are x and time t

```
input (rendered detected cell point), treat as image:

.......x..........x....... 
..................x....... 
.......x..........x....... 
.......x..........x....... 
.......x................. 
.......x................. 
.......x................. 
.......x...x............. 
.......x...x............. 


ouput after unet


.......S..........S....... 
........I.........I....... 
........I........I....... 
........I..........E....... 
........I................. 
........I................. 
........I................. 
........I...S............. 
.......E..E............. 


```


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d2 — 2026-07-10T16:44:46 — thread 723655*

Once you get detection point, you can plot everything once in zxy and color code using t. Hence it become 3d conv ( instead of 4d conv)you can think of recolor the points from t to linkage colors ( s,l,d,e ). Temporal unet —> ordinary unet ( like detecting blood vessel)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7ef2a07a8d2b8265e2ea95cf85512075%2FSelection_4368.png?generation=1783702607348734&alt=media)

if you use feature from temporal unet e.g. 32, then you create 1+32 channel (one is for time, and you copy 32 values from temporal to ordinary unet) in the tracking ordinary unet


---

## [1v] simple idea:"Your Affinity Field Tells Your Fate"
*comment d0 — 2026-07-07T19:39:53 — thread 723655*

feasibility study:

```
all_df = []
for f in glob_file[::2]:
    print(f)
 
    nodes_df, edges_df = read_geff(f) 
    print(nodes_df.head())
    print(edges_df.head())

    link_df = compute_link_displacement_stats(
        nodes_df,
        edges_df,
        scale=(1, 4, 4),
    )
    all_df.append(link_df)


link_df= pd.concat(all_df).reset_index(drop=True)
print(link_df.shape)
print(link_df[["dz_sub","dy_sub","dx_sub"]].describe())
print(np.percentile(np.abs(link_df["dz_sub"]), [50,90,95,99]))
print(np.percentile(np.abs(link_df["dy_sub"]), [50,90,95,99]))
print(np.percentile(np.abs(link_df["dx_sub"]), [50,90,95,99]))



(63751, 10)
             dz_sub        dy_sub        dx_sub
count  63751.000000  63751.000000  63751.000000
mean      -0.318066     -0.011200     -0.234604
std        1.149520      0.878445      0.989713
min      -37.000000    -13.000000     -9.750000  ###???
25%       -1.000000     -0.500000     -0.750000
50%        0.000000      0.000000      0.000000
75%        0.000000      0.500000      0.250000
max       35.000000     12.500000      5.750000 ###???
[1. 2. 2. 4.]
[0.5  1.25 1.75 3.  ]
[0.5  1.5  2.   3.75]
```

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F30872efbdb6d455fabf83b8dcbe210ea%2FSelection_4333.png?generation=1783453191851967&alt=media)


---

## [1v] beware of jumps in ground truth track
*comment d2 — 2026-07-11T11:04:20 — thread 724283*

This is a huge problem if test data is not the same as we learned and memorised the freezing unintentionally. It also means rule-based heuristics need to be changed to avoid overfitting etc.

“all freeze after the same frame indices:” this means they are cropped from the same master big volume… which i will be interested "to jigsaw it" back

---

do we need to probe the hidden test frame for frozen frame?


---

## [1v] Cotracker and other methods
*comment d1 — 2026-08-06T17:10:21 — thread 726924*

i agree. co-tracking (or other tracking) alone will not win the competition. It is the lineage (cell division) that will decide the winner.

Another key is post processing to repair the tracks (this is like MNS/box adjustment/box filtering in object detection competitions)

my suggestion is to learn track without cell division first (i.e. all tracks has only one BIRTH and DEATH). cell dvision is then handled at post-processing or stage 2 (e.g. classifier to decide if there is a split based on appearance changes and longer track cues). it is difficult even for humans to decide if there is cell division just based on two frames.


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d0 — 2026-09-01T09:19:47 — thread 738217*

An idea that is too much for the competition but could be feasible in long term cell tracking research. I have been looking at video generation deep net. You can have a depth map as prompt then generate anime or life movie.

So it is easy to create 3d virtual cell in blender and add motion. Then you can style it to create fluorescent microscopy volume.    

In fact with infinite data you can simply convert 4d to 4d end to end. From volume back to bender model.


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d0 — 2026-08-31T07:41:29 — thread 738217*

https://www.biorxiv.org/content/10.1101/2025.07.23.666425v1  
ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy  

another shortcut is :  
FOCUS3d --> label -->augmentation (e.g. affine, elastic deform) to create window of T=2 pairs.  
then you can train link transformer etc..


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d0 — 2026-08-31T04:04:38 — thread 738217*

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


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d0 — 2026-08-31T00:34:08 — thread 738217*

Once you have dense segmentation label, you can use many opensource tracker like hoct, itec, trackastra to make dense tracks for better training.

Then you can do longer range tracking over window of 5 or 8 (instead of 2)


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d1 — 2026-09-02T12:49:19 — thread 738217*

You should use focus3d, then measure
1. Hitrate of sparse annotation ( also distance error)
2. Compare num of detected nodes with estimated number of nodes

```
    geff_meta = GeffMetadata.read(
        zarr_file.replace(".zarr", ".geff")
    )

    est_num_nodes = float(
        geff_meta.extra["estimated_number_of_nodes"]
    )

```
—-

Also you should evaluate link transformer or other link model given gt location + other location and compared detected location + other location.


---

## [1v] focus3d : one of the best 3d cell segmentation
*comment d2 — 2026-09-02T12:51:42 — thread 738217*

Further, i think gt annotation must have used some open source cell instance detector. I suspect it it cellpose3d or stardist3d with manual collection.


---

## [1v]  Quick question for anyone above the 0.94 line — is the detector still a 3D UNet heatmap for you, or did you move to something else ? 
*comment d0 — 2026-09-02T23:41:57 — thread 738276*

i suggest you go through ultra api. there are many track post processing tools. try to 
1) use some opensource cell instance segmentation to get instance labels
2) use it as input to ultrack
3) then use ultrack to link
4) use post processing tools to correct /TrackEdit

make some manual annotations to appreciate the problems of ultrack and what can be solved by post processing.
Do it a few rounds and i think you can discover improvement points and how to get more data


---

## [1v]  Quick question for anyone above the 0.94 line — is the detector still a 3D UNet heatmap for you, or did you move to something else ? 
*comment d2 — 2026-09-02T23:44:50 — thread 738276*

Thanks for the reply. I checked the forum and competition webpage, but where is it mentioned that " LB is scored against dense labels. " I think all are using sparse labels like the downloaded train?


---

## [1v] Detector overfits past ~epoch 10 on fixed sparse-GT frames -- anyone else see this?
*comment d0 — 2026-09-02T18:01:07 — thread 738773*

cyan is kaggle ground truth annotation. A lot of such nodes in faint intensity are causing problems

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff94359f9b05ea644a7a049d9e2732bec%2FSelection_4746.png?generation=1788372012504153&alt=media) 

![
](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F88e15383975f28ac477b2c7276ffe2ce%2FSelection_4745.png?generation=1788372031229506&alt=media)


---

## [1v] I think we should use GNN for higher scores
*comment d0 — 2026-09-02T01:27:29 — thread 738778*

think of data before model


---

## [0v] good visualisation of the task
*OPENING POST — 2026-07-06T22:04:40 — thread 722668*




---

## [0v] simple idea:"Your Affinity Field Tells Your Fate"
*OPENING POST — 2026-07-07T17:37:36 — thread 723655*




---

## [0v] beware of jumps in ground truth track
*OPENING POST — 2026-07-10T12:47:17 — thread 724283*




---

## [0v] What if anyone will get 0.999 on lb, then competition will got cancled? I am curious, does any previous competition get 0.99 percentile on LB?
*comment d1 — 2026-07-15T11:56:50 — thread 725015*

lb more than 1.0 is possible. (hint: get your num of nodes correct using texture regression like human head counting, i.e. concept of density)


---

## [0v] Cotracker and other methods
*comment d2 — 2026-08-03T10:40:41 — thread 726924*

The bottom line is “do the tracks of others tell us about the target track”? Eg moving in groups or same patterns or relative spatial locations are preserved. If yes, the co-tracking all will help. Then to make it work, we need to have the correct formulation and data. We need to reformulate in sparse 3d and chatgpt etc can help


---

## [0v] Cotracker and other methods
*comment d0 — 2026-07-25T19:11:41 — thread 726924*

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F41e3cc42dc0de799feda946174d00330%2FSelection_4403.png?generation=1785006700020472&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Facb9f25bd185cb07f75dbb312208dd0d%2FSelection_4404.png?generation=1785006890371376&alt=media)


---

## [0v] Cotracker and other methods
*comment d0 — 2026-07-17T10:44:46 — thread 726924*

https://arxiv.org/abs/2411.14833  
Cell as Point: One-Stage Framework for Efficient Cell Tracking  

this is cell tracking based on CoTracker3, RAFT etc . There is comparsion with Trackastra


---

## [0v] not all sparse GT edge are correct
*OPENING POST — 2026-07-25T04:01:20 — thread 729053*




---

## [0v] Very dim nodes?
*comment d0 — 2026-08-31T00:39:19 — thread 737896*

It can be false positive or results of interpolation. Eg annotation label frame t=1 and t=3 and interpolate for t=2


---

## [0v] focus3d : one of the best 3d cell segmentation
*OPENING POST — 2026-08-30T15:24:33 — thread 738217*




---

## [0v] focus3d : one of the best 3d cell segmentation
*comment d2 — 2026-08-31T10:34:59 — thread 738217*

you can just randomly make some grid points that are non-background, then "track/link them" in next "augmented frame" as pretraining or aux loss


---

## [0v] focus3d : one of the best 3d cell segmentation
*comment d2 — 2026-08-31T09:50:37 — thread 738217*

You can download hf spaces gradio code and modify from there. It is self contained


---

## [0v] Detector overfits past ~epoch 10 on fixed sparse-GT frames -- anyone else see this?
*comment d0 — 2026-09-02T01:35:36 — thread 738773*

this is the effect of sparse annotations. the unlabelled nodes "changes from noisy targets to negative targets after 10 epoches".
