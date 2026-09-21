# magic or overfitting?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/740145
- **Topic id**: 740145
- **Author**: hengck23 (GRANDMASTER)
- **Posted**: 2026-09-08T13:31:27.038327900Z
- **Votes**: 13
- **Comments**: 12

---

## Opening post

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fb9aa6b1321cfd69983aab968deba4789%2FSelection_4826.png?generation=1788875194680167&alt=media)

e.g. at epoch 8:
refine\_prob >= 0.0888, remove 13.13% peak detection and has kaggle recall of 99.45%. Error distance from kaggle annotation in 64x64x64 canonical voxel is refined from 1.1766 to 0.969

---

## Comments (12)


### hengck23 (GRANDMASTER) — 2026-09-09T08:49:45.653Z — 2 votes

Improved cell detection is not used in inference but to create better dense training samples. Here you can see that better localisation clearly improves edge recall significantly.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F17e8b575e301310ace4f3ff8614a2bba%2FSelection_4851.png?generation=1788943783835062&alt=media)



note: ultrack has few node detector. But I haven't checked if they align with the Kaggle annotated point yet.
https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-daxi.pt    
 https://public.czbiohub.org/royerlab/ultrack/unet_weights/unet-simview.pt

### hengck23 (GRANDMASTER) — 2026-09-08T18:26:15.160Z — 1 votes

update
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F73c31f1a9f19a6fd92fc4b44fb6011f5%2FSelection_4838.png?generation=1788891973475998&alt=media)

how i know augmentation data is more difficult? I report metrics for different data groups. ChatGPT is incredible for debugging algorithms; just ask him how. This is how a beginner can become an expert in machine learning.

### hengck23 (GRANDMASTER) — 2026-09-15T15:28:38Z

end2end cell linking code:  
https://www.kaggle.com/code/hengck23/end2end-cell-linker-raw-edge-ja-0-9-no-ilp   

it shows the limit of using link probability only (no ILP, no gap filling). It has raw edge Jaccard of about 0.90. Trained without kaggle annotation. Use 20% of the data at t=0,5,10. ... Use dense3d label + augmentation to simulate cell movement.  

Tricks of getting good results is to analyse reason of FP (and MISS). e.g. multiple nodes matched to GT node and kaggle metric only consider first 2. your edge may be linking to other matching node (not the first 2)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-15T21:21:05.163Z

> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fae857e8cb99cbf535c6d0c18abe3aaa5%2Ft025_src375_fp30_gt27000420.png?generation=1789507198113607&alt=media)
> 
> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ff9496f88345de6f846e568fa00b552d1%2Ft022_src545_fp232_gt24000373.png?generation=1789507470719267&alt=media)
> 
> being trained from synthetic augmentation, the linker is very precise. below slow an fp edge in validation

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-15T21:27:34.920Z

> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F06347bc685a1d516332ea039b06b18d9%2Ft015_src524_fp457_gt95000000036.png?generation=1789507652632840&alt=media)
> > 
> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fbab6a5c982e52e0ff5b45ea80a7c698f%2Ft038_src91_fp64_gt118000000036.png?generation=1789507859626966&alt=media)
> > 
> > here you can see effects of domain shift. kaggle annotation is based on Ultrack segmentation i think it sometimes annotates "cell corners". my annotation is based on focus3d, will is cell center.

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-15T21:40:08.357Z

> > i think i can catch some kaggle annotation error?
> > ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Ffe9076b2fd8dd1666ed1bdce66c40084%2Ft062_src225_fp157_gt64000750.png?generation=1789508406517877&alt=media)

#### ↳ ↳ Satwik (MASTER) — 2026-09-15T23:07:23.423Z

> > there are definitely errors in kaggle annotations. here is a check I was doing of missed cells for my model, and the red + is GT label. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3459992%2Fb69c5828e02d7032e8ec6eed84562c55%2F0BDD02D9-EB9B-411C-A078-F64EE35699EC.jpg?generation=1789513638003132&alt=media)

### hengck23 (GRANDMASTER) — 2026-09-09T12:27:44.490Z

i use ChatGPT to make a trajectory of nodes (using slice as appearance ), with evaluation results like hit,fp,miss. It turns  out that quite a number of FPs are actually very close to the truth node.


![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F35f7a6efb326453d9dd8c14b6985cfb9%2FSelection_4859.png?generation=1788956851020692&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2Fd20d3f9daebdc021de9fb96a707b30ad%2FSelection_4860.png?generation=1788956862377427&alt=media)

#### ↳ hengck23 (GRANDMASTER) — 2026-09-09T12:35:18.217Z — 1 votes

> It actually means that during linking, my zxy must change, the node must shift to better position. So actually you cannot detect and fixed a location.  
> 
> 
> Or i need to guess the location model in kaggle annotation

### Mohit (EXPERT) — 2026-09-08T19:35:47.550Z

Idk y but  it feels overfitting

### hengck23 (GRANDMASTER) — 2026-09-08T14:54:38.210Z

how to analyze your edge transformer

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F86fdf404585661419220b50525609e15%2FSelection_4832.png?generation=1788879742049134&alt=media)

Obviously, the strategy is:
1) increase correct edge probability so that you can threshold to reduce fp  
2) fill the gap either by learning a net or heuristics ( 1-frame gap correction seems achievable)  

how to increase edge probability:
- "appearance feature smiliarity" : maybe better localisation? resolution? larger region/scale. need to visualise the error case, but i can imagine difficult cases are: cell density is too high and everyone is similarly packed, or the cell is too faint and disappears in the next frame, or the movement (and neighbours) is too large and looks different  

TRICKS!!!
- refine only candidates at 99% cutoff to avoid complex appearance features for edge probability
so i need to push edge recall from 0.966 to 0.990 for 99% cutoff. Then a refine module to better choose/rank from these candidates with more complex computation, or attention from multiple frames etc.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F3140beb4e7364cae6f958da162fbb0e0%2FSelection_4837.png?generation=1788881686957802&alt=media)

locate the ceilings

### Navneet (CONTRIBUTOR) — 2026-09-09T10:08:27.853Z

Thank you for the magic @hengck23
