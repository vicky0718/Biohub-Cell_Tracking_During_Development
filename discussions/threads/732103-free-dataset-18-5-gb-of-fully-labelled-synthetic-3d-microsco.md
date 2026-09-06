# [Free Dataset] 18.5 GB of fully-labelled synthetic 3D microscopy — 165k labelled divisions

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103
- **Topic id**: 732103
- **Author**: José Freitas (EXPERT)
- **Posted**: 2026-08-01T21:12:10.502460400Z
- **Votes**: 47
- **Comments**: 9

---

## Opening post

I got tired of trying to train a division model on ~304 events, so I built a synthetic dataset and I'm releasing it free (CC0).

**The problem, concretely.** The training ground truth is sparse — roughly 2.8% of nuclei are annotated, and across all 199 videos there are about 304 division events. Divisions carry 10% of the metric (`score = adjusted_edge_jaccard + 0.1 * division_jaccard`), and most public solutions I've seen simply leave that term at zero. That is understandable: you cannot fit a division model to a few hundred scattered examples.

**What I generated.** 18.5 GB where every nucleus is labelled:

- **1,539 static volumes** at native 256×256 resolution, centroids with sub-voxel precision
- **2,174 time sequences** with the complete lineage graph — nodes, edges, and mitosis events
- **165,267 labelled divisions** across 4,056,226 nodes

That is roughly 540× more mitosis supervision than the competition ground truth provides.

**It is a physical model, not a GAN.** Dark medium, physically-sized ellipsoidal nuclei with a super-gaussian profile (the real radial profile is a flat top, not a gaussian — I measured it), light emission that scatters and accumulates where cells are dense, anisotropic PSF, then Poisson + read noise. Three details I'd flag as the ones that actually matter:

1. **The pooling matches the evaluator exactly.** The official pipeline downsamples XY by 4 with a *stride* (`vol[:, ::4, ::4]`), not a block mean. A block mean averages noise away and would hand you data that is cleaner than what your detector really sees. I generate at native resolution and apply the identical stride.
2. **Detectability is calibrated, not assumed.** A classical DoG detector recovers ~0.89 of the synthetic nuclei in a moderately dense field, ~0.76 when crowded; on the real annotated nuclei it recovers 0.91–0.94. So if anything the synthetic volumes are slightly *harder*, not easier. The notebook measures this live at three densities so you can judge the gap yourself.
3. **Tissue geometry is fitted from the real detections** — the embryo surface, its curvature, its thickness — and then resampled. The geometry is learned from real data; the coordinates are generated. No real positions are copied.

Motion is calibrated on the real lineage edges: 1.86 µm/frame median step, +0.30 lag-1 directional persistence, 7.24 µm sister separation.

**Explorer notebook (runs end to end, all figures computed live):**
https://www.kaggle.com/code/josefreitasalvesneto/synthetic-3d-microscopy-data-for-cell-tracking

**Dataset + full generator source (open, nothing hidden):**
https://www.kaggle.com/code/josefreitasalvesneto/biohub-synthetic-dataset

**Two things I want to be upfront about**, because I'd want to know them before spending 18 GB of quota:

- The division rate is **deliberately inflated** (4.07% of nodes vs ~0.26% in reality). A model starved of examples never learns mitosis. Re-weight your loss by the real rate if you need calibrated priors.
- Distribution match is partial. Nucleus texture and contrast are the weakest axes. The value here is the **labels** — density, lineage and mitosis — much more than photorealism.

Happy to answer anything about the generator. If you try it and it breaks, or it helps, I'd genuinely like to hear which — both make the next version better.

---

## Comments (9)


### hengck23 (GRANDMASTER) — 2026-09-05T09:00:18.660Z — 2 votes

if i got time, i want to do this:  
1) it is difficult to track over 100 frames, but it is easy to track over eg 5 to 10 frames.  
2) it is also easy to manually inspect and correct (1) above too.  
3) so we can easily use video diffusion, gan, flow etc to make synthetic data for 10 frames (at 64x64x64),   conditioned on tracks and the first frame  

---

After sucess of dense cell training and dense pair (link) training, i now proceed to do 5-frame training:  
- augmentation (elastic transform over 5 frames)  
- dense pesudo label Kaggle tracks using opensource ultrack, ITEC, HOCT ....
- make a new intertace for manual inspection and correct for short tracklet

### hengck23 (GRANDMASTER) — 2026-09-04T23:42:33.660Z — 1 votes

There is another similar solution. Convert real microscopy image to simplified eg ellipsoid image using unet that output ellipsoid ( just like bonding box in instance object detection). Then you create sythetic unet output to feed into link transformer.

The problem is then shift to instance detection. Need to find lots of real microscopy image with cell division.

—-

I haven’t moved to cell division yet. But my current strategy is to have good cell detection and tracking. Then infer division from long track ( stage 3 ). I assume death/birth can only happen at image boundary,  so maybe we can detect splits from long tracks, ie i treat division as tracklet association problem, tracklet as node association

### José Luiz Luna-Xavier (CONTRIBUTOR) — 2026-09-04T18:56:39.873Z

Hi José (José Freitas Alves Neto),

I wanted to flag three specific things in your generator that I think deserve more attention than a quick "thanks, nice dataset" — because they show real domain awareness rather than just synthetic-data box-checking.

1. Matching the stride, not a block mean. Noticing that the official pipeline downsamples XY with vol[:, ::4, ::4] (a stride) rather than a block-average, and reproducing that exactly, is a subtle but important call. A block mean averages noise away and would silently hand a detector cleaner statistics than it will ever see at inference — a train/eval mismatch that's easy to miss and hard to debug after the fact, since it doesn't show up as an obvious bug, just as a quiet leaderboard gap. Catching this before training anything is genuinely sharp.

2. Self-critical detectability calibration. Reporting that your DoG baseline recovers less on the synthetic volumes than on the real annotated nuclei (0.89/0.76 vs 0.91–0.94) is the opposite of what most synthetic datasets do. It's tempting to tune synthetic data until it "looks easy" and quietly inflates confidence. Publishing numbers that go the other way is a good-faith signal that's rare to see, and it makes the dataset much more trustworthy to build on.

3. Sister-separation calibrated from real lineage edges. The 7.24 µm figure, tied back to actual BioHub lineage data rather than an assumed value, is directly useful — it's one of the few external anchors we have for reasoning about candidate-graph radius choices around division events specifically, as opposed to continuation motion in general.

Taken together, these aren't cosmetic details — they're the kind of choices that decide whether a synthetic dataset actually transfers or just looks good in isolation. Really solid work, and thank you for documenting the caveats as openly as the strengths (the inflated division rate, the texture/contrast gap) — that honesty is what makes it usable rather than just impressive.

One thing I'd love your take on, if you have a moment: could you clarify exactly what a single "division" is counted as in the 165,267 figure — a mitotic parent/fork, a daughter node, or a parent→daughter edge? We're (my team) trying to align units before doing any prior-reweighting on our end, and getting that straight from you directly would save some guesswork.

Thanks again for releasing this — genuinely useful contribution to the competition. Excellente Contribuição, forte abraço, José Luiz.

#### ↳ José Freitas (EXPERT) — 2026-09-05T01:57:43.933Z

> Hi José Luiz,
> 
> Thanks for reading it that closely — and for asking about units rather than assuming them. Happy to pin it down precisely, because it's the kind of thing that's easy to get wrong by exactly one factor of two.
> 
> A "division" in the 165,267 figure is a mitotic parent — one fork event. Not a daughter, and not a parent→daughter edge.
> 
> Concretely, in the generator: when a track divides, I record the parent's node id (its node at frame t-1), and I record it once per daughter — then store sorted(set(...)). The set() is doing real work there: it collapses those two entries back down to a single event. So the divisions array in each seq_XXXX.npz holds node indices of mitotic parents, which matches how it's documented in the explorer — "a node with two outgoing edges is a division."
> 
> Since every division in the generator produces exactly two daughters, the conversions are exact. 165,267 mitotic parents (fork events) is the published figure. That corresponds to 330,534 daughter nodes, and 330,534 parent→daughter division edges, out of 4,056,226 total nodes. So if you're working in edge units or daughter units, double it; if you're working in event units, take it as published.
> 
> And for the reweighting specifically: the 4.07% figure is in that same unit — mitotic parents divided by all nodes, i.e. 165,267 / 4,056,226. So it's directly comparable to the ~0.26% real rate I quoted; both are "fraction of nodes that are a dividing parent." No conversion needed on your side, as long as you count the real rate the same way. That's a ratio of roughly 15.7x, if you're scaling a positive-class weight.
> 
> One gotcha I should have put in the post, since you're about to parse these files: both daughters inherit the parent's track_id. So nodes[:, 4] is a lineage/clone id, not a unique track — after a division, two distinct tracks share it. The unambiguous lineage lives in edges (and a node with two outgoing edges is exactly a mitosis). If you build your candidate graph off track_id expecting uniqueness, it will look like cells teleport.
> 
> While I'm at it, one honest caveat on the detectability numbers you mentioned: the 0.91–0.94 real reference comes from the sparse annotations (~2.8% of nuclei), and annotators tend to label unambiguous cells — so that reference is probably optimistic, and I haven't measured how real recall varies with density. The notebook says this too, but it's worth repeating, since it means the synthetic-vs-real gap may be a little smaller than the raw numbers suggest.
> 
> If you do the prior-reweighting and it moves anything for you, I'd genuinely like to hear it — that's the feedback that improves the next version.
> 
> Forte abraço,
> José

### Ace (EXPERT) — 2026-08-08T09:47:15.693Z — 2 votes

This is a great effort, thanks for releasing it. One thing I haven't seen raised yet: our real training set turns out to come from only 2 actual embryos, and they differ a lot  annotation sparsity of roughly 1% vs 9%, and division counts of 26 vs 125 across their samples. Given that real embryo-to-embryo heterogeneity, I'm curious whether the synthetic generation process models that kind of variation (density, noise, crowding) across its volumes, or whether it's drawn from one more canonical/idealized regime.

#### ↳ José Freitas (EXPERT) — 2026-08-09T14:02:08.883Z — 1 votes

> Hello, I'm happy to help! I understand your doubt, the divisions were generated inspired by real movement, we might not have total accuracy, but we can use my data to pretrain a model and then fine-tune it with the real data, that helps it converge faster!

### Juan Neira (CONTRIBUTOR) — 2026-08-06T06:30:17.397Z — 2 votes

First and foremost, thank you for your effort Jose and for sharing.

I just started in this competition, so I'm not sure if I will use your data yet, but If I do, I will let you know.

#### ↳ José Freitas (EXPERT) — 2026-08-06T12:03:18.437Z — 1 votes

> Thank you very much! I'm happy to help! If you have any questions, I'm available!

### hengck23 (GRANDMASTER) — 2026-09-05T12:48:14.990Z

synthetic data from biohub  
https://virtual-embryo-zoo.sf.czbiohub.org/
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F113660%2F744fe1d219782a2f1503130d654c3bd3%2FSelection_4780.png?generation=1788612493147291&alt=media)
