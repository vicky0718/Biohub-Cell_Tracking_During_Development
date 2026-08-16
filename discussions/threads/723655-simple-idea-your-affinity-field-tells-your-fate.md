# simple idea:"Your Affinity Field Tells Your Fate"

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723655
- **Topic id**: 723655
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-07-07T17:37:36.697254400Z
- **Votes**: 22
- **Comments**: 19

---

## Opening post

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fc2e84c73dc0f3992d61c0c742e0c2975%2FSelection_4328.png?generation=1783445842204878&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F2273352e7963197f144ce11501586ae2%2FSelection_4329.png?generation=1783445854887011&alt=media)


"Temporal Affinity Fields for 3D Cell Lineage Reconstruction"  
- predicting a local vector field that tells the graph optimizer how cells should connect   

Hint: create flow GT for supervision using optical flow and sparse annotation tracks

---

## Comments (19)


### hengck23 (GRANDMASTER) — 2026-07-16T04:42:56.547Z — 1 votes

i find a metric hack:  
you make a graph. if your just repeat your tracks (giving new id) your edge\_jaccard is not affected. this is the cause the kaggle metric don't penalize fragmentation, duplication. hence you can create multiple almost smiliar tracks to improve TP. But there is the node num correction (aka adj\_edge\_jaccard) But you can over come this by duplicating/perturbing at the correct length and correct location.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F0c8f49fc633511f2736c46579fa5b7f3%2FSelection_4383.png?generation=1784177205409163&alt=media)

#### ↳ Thibaut Goldsborough (CONTRIBUTOR) — 2026-07-16T17:04:28.493Z — 1 votes

> Hi, perfectly duplicated edges are filtered out before scoring the edge Jaccard, this might be why you are not seeing a change in the Jaccard score, each ground truth edge can only used for a single TP. If you duplicate the edges but modify the nodes very slightly, then you should see a drop in Jaccard. This should not allow you to hack (i.e. increase) your score.  Please contact us if you can actually game the metric.

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-17T13:11:05.160Z — 1 votes

> > How to hack metric:
> > 
> > 1. Real matching nodes only need to be in the same weakly connected component​
> > 
> > 2. Forks only need to exist anywhere within that component​
> > 
> > 3. Unmatched false forks are generally not counted as FP
> > 
> > 4. I'm now trying the next... not sure it works:
> > - 4.1.  ~~just copy high-confidence, continuous, linear non-division trajectories.~~
> > - 4.2. Shifted track targeting 7µm node matching

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-17T13:25:14.247Z — 1 votes

> > I fully understand the original intention behind designing this metric. When I performed local validation, I found that achieving very high accuracy (e.g., 0.999) is quite easy, but balancing recall and other metrics is relatively difficult, because excluding negative samples is a very challenging task. I believe that refining the metrics is also an indispensable step in advancing this research, as your current work is already pushing the limits.

#### ↳ ↳ Thibaut Goldsborough (CONTRIBUTOR) — 2026-07-17T22:33:20.300Z — 1 votes

> > Indeed the division metric can be gamed. We're working on a patch and will update the scoring asap. I will make a separate post explaining this.

#### ↳ hengck23 (GRANDMASTER) — 2026-07-18T01:17:36.170Z — 1 votes

> I think the node correction can both increase and decrease original edge jacard score?

### hengck23 (GRANDMASTER) — 2026-07-17T04:35:11.853Z — 2 votes

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F20e5cb2adbba2365f79bbb79b5b5024e%2FSelection_4384.png?generation=1784262681228749&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F1ed4a3ecc2a5dedc98fcdc44ba8ef0bd%2FSelection_4385.png?generation=1784263063174669&alt=media)


center fig: red is t=0, green is t=1   

Visualisation results from host repo baseline code (temporal unet + link transformer). i show max prob link.  
- long links are almost wrong (easy to filter such results)


looking at the results, link are short. maybe a conv  3d CNN (T as channel) unet to detection motion object is good? i.e. rather than detect zxy center, we segment "center of motion = line connecting zyx0 and zyx1."

### Tom (MASTER) — 2026-07-08T06:01:56.357Z — 3 votes

I just start to develop flow approach then seeing your post. Welcome back @hengck23

#### ↳ hengck23 (GRANDMASTER) — 2026-07-08T18:05:02.130Z — 2 votes

> i have been playing with it. i think the winning formula is to generate dense tracks for training (1) and(2) below  
> 1) 3d point is easy to generate (e.g. opensource cellpose, etc)  
> 2) short track (2 frame or 3 frame) is easy to generate. it can be the utlrack, or rule base heuristics or open source tracker.  
> 
> if 1 and 2  can get good results, then ILP or min-cost graph-cut network will generate the long tracks required for submission.
> 
> currently the rule-based graph correcting post processor in public notebook should be used for generating new links in (2) for training. Imagine if i use 5 open source tracker, and using consistency, i can have more short tracks (currently we have less than 1% link labelled and the rest are unlabelled)
> 
> (2) only need locations. there are opensource zebrafish data with long (real and synthetic) dense tracks (lineage) without microscopy images that can be used too.
> 
> 
> https://chatgpt.com/c/6a4e91cd-8d90-83ec-bf7a-825b27a9e284

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-07-09T17:32:44.350Z — 2 votes

> > i change strategy a bit:
> > 1) use opensource to make tracks, measure local LB score
> > 2) ensemble opensource and own/ai heuristics,  measure local LB score
> > 3) when i have good LB score, these become dense pesudo labels.
> > 
> > opensouce trackers are very good for short tracking. 
> > tricks:
> > - cellpose etc to provide dense 3d tzyx
> > - use only  tzyx for opensouce tracking. some opensouce like ultrack needs segmentation labels as input, i synthetically rendered 3d ball as input

### hengck23 (GRANDMASTER) — 2026-07-10T13:29:32.203Z — 1 votes

Something struck  my mind. Since it is video and tracking, unsupervised pretraining should help. Hence we no need dense track or point


Tracking needs dense, spatially precise features. The newer V-JEPA 2.1 specifically targets dense, spatially structured and temporally consistent features ….  



V-JEPA 2.1 behaves like unsupervised optical flow in feature space,

### hengck23 (GRANDMASTER) — 2026-07-10T03:07:03.517Z — 1 votes

3d cuda optical flow: https://github.com/yongxb/OpticalFlow3d  
3d optical flow kalman filter tracker : https://byotrack.readthedocs.io/en/latest/  


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff09cbac39ef7361221ceef47e9382955%2FSelection_4356.png?generation=1783652776657450&alt=media)

hint: use synthetic image or synthetic motion is eaiser to debug your code

### hengck23 (GRANDMASTER) — 2026-07-09T23:39:28.950Z — 1 votes

Open-source optical flow tracker + Kalman filter. 
The visualisation is messy. anyone has a better suggestion?

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F96f536ce5845530631cda7458673780e%2FSelection_4347.png?generation=1783640249216751&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd9a4b36a962237c4eb72da62851d472f%2FSelection_4348.png?generation=1783640266346613&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-07-10T02:30:20.060Z — 1 votes

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbac01c8b8fc39e9f60fddbe7aacec3d3%2FSelection_4351.png?generation=1783650518238328&alt=media)
> 
> results of ultrack and byotrack. They tend to disagree when there is "large predicted movement". this is because the optimization cost forces it to "always" link to something
> 
> "link cost < termination cost + new-track cost"

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-07-10T11:07:44.763Z — 1 votes

> > in theory, linking can be done by CNN.
> > 
> > i illusrate with a 1d example:  the axes are x and time t
> > 
> > ```
> > input (rendered detected cell point), treat as image:
> > 
> > .......x..........x....... 
> > ..................x....... 
> > .......x..........x....... 
> > .......x..........x....... 
> > .......x................. 
> > .......x................. 
> > .......x................. 
> > .......x...x............. 
> > .......x...x............. 
> > 
> > 
> > ouput after unet
> > 
> > 
> > .......S..........S....... 
> > ........I.........I....... 
> > ........I........I....... 
> > ........I..........E....... 
> > ........I................. 
> > ........I................. 
> > ........I................. 
> > ........I...S............. 
> > .......E..E............. 
> > 
> > 
> > ```

#### ↳ ↳ Tom (MASTER) — 2026-07-10T11:24:05.523Z — 1 votes

> > That's good reformulation. Consider do
> > X, t
> > Y, t
> > Z, t
> > Three heads
> > 
> > Seems SDF would work?

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-07-10T16:44:46.160Z — 1 votes

> > Once you get detection point, you can plot everything once in zxy and color code using t. Hence it become 3d conv ( instead of 4d conv)you can think of recolor the points from t to linkage colors ( s,l,d,e ). Temporal unet —> ordinary unet ( like detecting blood vessel)
> > 
> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7ef2a07a8d2b8265e2ea95cf85512075%2FSelection_4368.png?generation=1783702607348734&alt=media)
> > 
> > if you use feature from temporal unet e.g. 32, then you create 1+32 channel (one is for time, and you copy 32 values from temporal to ordinary unet) in the tracking ordinary unet

### hengck23 (GRANDMASTER) — 2026-07-10T17:35:26.270Z — 2 votes

solving the sparse annotation for cell center detection.     
1) observation : DoG peak detection somehow work. i.e., microscopy cell are blobs.    
2) instead of classification binary problem, we reformulate it as learnable peak detection. at each pixel location after unet logit head, loss = softmax of pixel over his neighbour.  those pixel without annotation are not computed in loss at all     

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F7a94ffe3bbada2de65d720c1fead0bd7%2FSelection_4369.png?generation=1783705181623495&alt=media)

hint: ask chatgpt make a margin loss version: peak is at least T greater than neighbour

### hengck23 (GRANDMASTER) — 2026-07-07T19:39:53.660Z — 1 votes

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
