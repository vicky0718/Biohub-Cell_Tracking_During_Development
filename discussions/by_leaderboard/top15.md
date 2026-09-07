# Forum posts by the current top 15 teams

29 posts, 23 usernames. Newest snapshot of `leaderboard_full.csv`.


## rank 1 — Sergio Alvarez (0.970) — @sersasj — 2026-07-01

*thread: Has anyone experienced extremely long scoring times?*

> Hi @rahulparmeshwar. Here's the snippet from the overview page:
> >When a notebook is submitted for rerun, a new hidden test set is swapped in. The size of the hidden test set is approximately the same size as the training dataset.
> 
> The training set has 199 image volumes (T=100, Z=64, Y=256, X=256), so the hidden test set will be roughly that same size. I haven't submitted, but ~200 second per volume should fit in 12h
> 

## rank 1 — Sergio Alvarez (0.970) — @sersasj — 2026-08-15

*thread: Possible big leaderboard shakeup*

> Hi @tweakai, by plugin do you mean something that optimizes/refines the tracking graph, or do you mean you take the public notebook detections and apply your own tracking method on top?
> 
> Got curious about it, but no worries if you can’t share more
> 

## rank 1 — Sergio Alvarez (0.970) — @sersasj — 2026-09-05

*thread: Metric problem*

> Hey @antonoof, you’re sure that your submission removed the edges? Every node/edges removal I've done reflected public lb/cv score. 
> About the unbounded score, the hosts are aware. They've commented in other posts ready, its the expected behavior
> 

## rank 1 — Sergio Alvarez (0.970) — @sersasj — 2026-09-05

*thread: can we turn auto research to auo data generator?*

> I've tried the synthetic volume idea for ~2 weeks with no clear improvement in my scores, so I dropped it. Still, it would be cool to see this strategy work here, as it helped improve a little in the CZII competition. (Polnet was used there: https://github.com/anmartinezs/polnet)
> 
> I suggest to send DaXi microscope paper as reference (https://www.nature.com/articles/s41592-022-01417-2)
> 
> Here is one example of synthetic volume I tried to use:
> 
>  ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F2221915%2F508c44ad1c064184a3a589b7de41f5dd%2Fsynthetic_z_stack_optimized.gif?generation=1788641223173538&alt=media)
> 

## rank 2 — Soheil Ayati (0.966) — @soheilayati — 2026-08-23

*thread: Stuck at 0.928*

> Hi! Here is what I found out that may help you as well:
> 
> Break your missed edges into two categories: missing endpoint nodes and incorrect associations. In my case, many "linking" issues actually originated earlier during node selection, so improving the edge model won't necessarily help.
> 
> Always validate complete movies using the official scorer and movie-level OOF splits. Edge-level random CV can be highly misleading. Once you identify which category dominates, focus exclusively on that bottleneck.
> 
> Good luck and have fun!
> 

## rank 5 — Tang (0.961) — @hirotetsu — 2026-08-15

*thread: Possible big leaderboard shakeup*

> I guess what we can do is:
> - Build a reliable CV, and trust it.
> - Use external datasets or synthetic data to make our model more robust on division cells prediction.
> 

## rank 5 — Tang (0.961) — @hirotetsu — 2026-08-16

*thread: What is the best model for this domain so far?*

> I'd recommend retraining the model instead of just using the public ckpt.
> unet+transformer is a strong approach, but there are still things you can improve, like preprocessing, model architecture
> the current ckpt has kind of hit a wall, it's hard to get more gain from post-processing alone.
> 

## rank 5 — Tang (0.961) — @hirotetsu — 2026-08-19

*thread: What is the best model for this domain so far?*

> the common approach for this task is basically two parts: modeling and track optimization, aka post-processing.
> for modeling, use gpt or something to rebuild the training pipeline from the public notebook. just treat it like a normal CV task , here are a lot of similar competitions , so you can check those.
> as for track optimization, maybe some algorithm or correction model. i'm not working on that part. so I don’t know much about it yet.
> 

## rank 5 — Tang (0.961) — @hirotetsu — 2026-08-27

*thread: what layer did ur gains actually come from*

> improvements com from all of them.
> it's hard to say which one matter most, but i think there's a "correct" order to work on them:
> detection -> linking -> division.
> detection should come first, once detection is solid, it's easier to improve others.
> 

## rank 5 — Tang (0.961) — @hirotetsu — 2026-08-29

*thread: what layer did ur gains actually come from*

> glad to help.
> in my case, most of the gains come from model improvements.
> 

## rank 6 — TWEAK (0.957) — @antonoof — 2026-07-18

*thread: Zarr import*

> # First install dependences
> 
> ![f1](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17954496%2F217ec3c8f4de6cf1a688fa2f2278ff11%2Ff1.png?generation=1784374380922525&alt=media)
> 
> 
> # Next save
> 
> 
> 
> ![f2](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F17954496%2F7402c5023e9faf489b99801be191f558%2Ff2.png?generation=1784374404526108&alt=media)
> 
> https://www.kaggle.com/code/antonoof/import-kaggle?scriptVersionId=336231500
> 

## rank 6 — TWEAK (0.957) — @antonoof — 2026-07-21

*thread: zarr import error in submitted notebook*

> https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726955#3500046
> 

## rank 6 — TWEAK (0.957) — @tweakai — 2026-08-15

*thread: Possible big leaderboard shakeup*

> I can see why it may appear that way. I can't speak as to what others in the top 10 are doing, but we are not focused on 0.0001 or Division J; we are working on a universal plugin that the bio cell team can plug into their current pipeline with minimal changes. We have tested our plugin with every available unique public notebook and model, with gains ranging from 0.030, 0.040, to 0.050 instantly just attaching our plugin. We've seen gains from a single public model reach a score of 0.940 untuned. We are not focused on the 0.0001 or tuning to the hidden.
> 

## rank 6 — TWEAK (0.957) — @antonoof — 2026-09-02

*thread: Cell Tracking In-Person Workshop*

> I am ready to work in the biohub team =)
> 

## rank 6 — TWEAK (0.957) — @antonoof — 2026-09-05

*thread: Metric problem*

> I deleted the ones that metric ignores by its own definition. And there is an experiment that shows this without a single controversial point: remove only edges, do not touch nodes — then the node counter and multiplier do not change at all, and any shift would be purely edge-based.
> 

## rank 6 — TWEAK (0.957) — @antonoof — 2026-09-06

*thread: Metric problem*

> maybe I was mistaken. I tried deleting edges, deleted them more than 1/2, the metric did not change, deleted 2/3, the metric did not change. Okay, I think it's just me, I'll share the results after the competition.
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-01

*thread: Rule-based is surprisingly strong? (currently 7th/344teams / gold zone, no learning)*

> Hi, may I know your division jacard CV?
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-01

*thread: Rule-based is surprisingly strong? (currently 7th/344teams / gold zone, no learning)*

> I think a lot of stuffs can play around with those rules. Might need to reformulate the problem statement. I believe a certain learning signal can make deep learning model powerful.
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-02

*thread: Only 2 groups of embryo_id？*

> Hi @thibautgoldsborough, the test set also consists of two groups of embryos? and the distribution is similar to train set: 71:128?
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-03

*thread: How do people use AI tools in competitions*

> Some tips:
> * plan idea by yourself, even better if you implement it by yourself and let it optimize your code :)
> * ask it to make [tutorial notebook](https://www.kaggle.com/code/tom99763/tracksdata-tutorial-evaluation-mock-tests) (Iike this one I made) for this challenge and upload to kaggle. then spending serveral days manually go through every step and data flow to understand data. 
> * keep generating [html report ](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/717109)and explaination to check how well it understand the challenge, that's also a good tool for showing your teammate your results
> * repositery organization
> * do not copy public notebook and ask it to gain lb
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-04

*thread: Resource sharing for cell tracking challenge*

> Finally division got learning signal!
> 
> ![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4310004%2F8f869234903a7d25d2b54ecad8a04d66%2F123.png?generation=1783143916774683&alt=media)
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-08

*thread: simple idea:"Your Affinity Field Tells Your Fate"*

> I just start to develop flow approach then seeing your post. Welcome back @hengck23
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-10

*thread: simple idea:"Your Affinity Field Tells Your Fate"*

> That's good reformulation. Consider do
> X, t
> Y, t
> Z, t
> Three heads
> 
> Seems SDF would work?
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-10

*thread: beware of jumps in ground truth track*

> Some cells are natural long jumpers
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-11

*thread: beware of jumps in ground truth track*

> Videos sharing the schedule	Frames duplicated after index…
> 6bba_05b6850b · 07477033 · 5b28472a	4, 12, 27, 42, 52, 57, 59, 62, 66, 76
> 6bba_1f58c2f6 · 20852818 · 80d12824	1, 14, 19, 43, 44, 45, 49, 57, 61, 66, 73
> 
> See html
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-07-14

*thread: Share a custom napari visualizer*

> @jookuma  Tree representation is better for this challenge. Really thanks for sharing this I'll update my design
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tahaalshatiri — 2026-07-21

*thread: Any update on the re-scoring timeline?*

> I've just joined and been waiting for 3 days for the rescore, as it's hard to do expirements with a misleading leaderboard and public notebooks
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-08-21

*thread: Share a custom napari visualizer*

> After 30 days suspension. I finally back
> 

## rank 8 — Vibes & Edges Trade-Off (0.954) — @tom99763 — 2026-08-31

*thread: focus3d : one of the best 3d cell segmentation*

> @hengck23  It looks like there are even more great ideas to me now
> 