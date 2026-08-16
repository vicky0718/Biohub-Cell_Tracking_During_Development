# [Free Dataset] 18.5 GB of fully-labelled synthetic 3D microscopy — 165k labelled divisions

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/732103
- **Topic id**: 732103
- **Author**: José Freitas (CONTRIBUTOR)
- **Posted**: 2026-08-01T21:12:10.502460400Z
- **Votes**: 35
- **Comments**: 4

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

## Comments (4)


### Ace (EXPERT) — 2026-08-08T09:47:15.693Z — 2 votes

This is a great effort, thanks for releasing it. One thing I haven't seen raised yet: our real training set turns out to come from only 2 actual embryos, and they differ a lot  annotation sparsity of roughly 1% vs 9%, and division counts of 26 vs 125 across their samples. Given that real embryo-to-embryo heterogeneity, I'm curious whether the synthetic generation process models that kind of variation (density, noise, crowding) across its volumes, or whether it's drawn from one more canonical/idealized regime.

#### ↳ José Freitas (CONTRIBUTOR) — 2026-08-09T14:02:08.883Z — 1 votes

> Hello, I'm happy to help! I understand your doubt, the divisions were generated inspired by real movement, we might not have total accuracy, but we can use my data to pretrain a model and then fine-tune it with the real data, that helps it converge faster!

### Juan Neira (CONTRIBUTOR) — 2026-08-06T06:30:17.397Z — 2 votes

First and foremost, thank you for your effort Jose and for sharing.

I just started in this competition, so I'm not sure if I will use your data yet, but If I do, I will let you know.

#### ↳ José Freitas (CONTRIBUTOR) — 2026-08-06T12:03:18.437Z — 1 votes

> Thank you very much! I'm happy to help! If you have any questions, I'm available!
