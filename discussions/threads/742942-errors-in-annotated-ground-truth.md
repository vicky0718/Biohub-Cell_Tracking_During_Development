# errors in annotated ground truth

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742942
- **Topic id**: 742942
- **Author**: Quantizr (CONTRIBUTOR)
- **Posted**: 2026-09-24T09:23:26.765014900Z
- **Votes**: 7
- **Comments**: 6

---

## Opening post

I've been looking over the sparsely annotated ground truth data for a while, and it seems like a good amount of the data is annotated incorrectly. I don't have exact stats but I've noticed some tracks where some of the centroids are not within 7um of the correctly linked cell for a given track, some links are just incorrect, or most recently I've been looking at cell divisions and here is something that is annotated as a cell division. You can clearly see that the blue annotated daughter cell was already there in previous frames and was not the result of a division (and at t+2 the centroid is significantly offset from the cell vs t+1 and t+3). Or its just near impossible to see even with contrast changes.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14167270%2Ffaf5df9316c95030c6f3b6c84930e9c9%2FScreenshot%20from%202026-09-24%2002-12-01.png?generation=1790241141046506&alt=media)

I just hope the test set isn't annotated this badly... but that might explain why some models which I think should do worse on manually annotated data seem to do better on the actual test set...

---

## Comments (6)


### Peter Green (CONTRIBUTOR) — 2026-09-26T21:27:26.690Z

Which training example is this?

### Alexandre Moritz (CONTRIBUTOR) — 2026-09-26T21:14:42.633Z

I did a labeling tool with AI to help me to review them faster 
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F658934%2Fc39c1f2272dad8d3b3dda82f3eef4c25%2Flabeling.png?generation=1790457260626145&alt=media)

### Himanish Goel (CONTRIBUTOR) — 2026-09-26T09:31:01.590Z

We noticed some errors in GT too, from tracks to splits, sometimes a split is marked at the wrong coordinates and sometimes the daughter cell is dropped 1 frame later. 

Trying to build an honest model is probably going to hurt score unless we specifically go and replicate the same noise that the GT has, 
or the test set could actually be better labelled than GT.

### Satwik (MASTER) — 2026-09-24T12:28:58.843Z

I agree, I have had similar observations. My detector has about 0.975 recall on competitio ndata, and from what I have tested (on multiple other datasets too) , it seems to detect almost every cell perfectly well , except a few lighter ones. ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3459992%2F8507612cd9ba0f30e4da18ce88b9aabe%2F0BDD02D9-EB9B-411C-A078-F64EE35699EC.jpg?generation=1790252870463349&alt=media)

### Quantizr (CONTRIBUTOR) — 2026-09-24T09:50:25.447Z

Like genuinely where is the cell division here???
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14167270%2Fc95ce39dd5f4b3a8314372590e8e7e7b%2FScreenshot%20from%202026-09-24%2002-48-10.png?generation=1790243301564530&alt=media)
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F14167270%2F1d9e02799c000265eaf85ba6b20b845e%2FScreenshot%20from%202026-09-24%2002-47-53.png?generation=1790243309591150&alt=media)

#### ↳ sghwr (CONTRIBUTOR) — 2026-09-24T10:03:51.637Z

> thats genuinely true from my side. we checked all fp originated from our model and do a blind test using our human eyes, the result found that ~70% of those fp can actually be tp since some of the annotation may be truly noisy🫠
> 
> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F29779493%2F67ac7b459492aa15733357277aa2db54%2Fimage%20(1).png?generation=1790244214320719&alt=media)
