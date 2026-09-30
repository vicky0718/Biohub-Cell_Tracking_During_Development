# From-scratch 2.5D ConvNeXt detector + ILP tracker — Private 0.939

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744485
- **Topic id**: 744485
- **Author**: hjyact (CONTRIBUTOR)
- **Posted**: 2026-09-30T00:49:55.120631400Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

Code: https://www.kaggle.com/code/hjyact/biohub-model-inference (Version 47 is the one that scored 0.939)

Thanks to the hosts for this competition. It was a really fun problem, and a painful one at times.

First, a confession. My best model scored 0.939 on private (0.929 public), which would have been somewhere around 22nd–25th. I didn't pick it as a final submission. I went with two versions that looked better on the public LB, and both ended up at 0.917. So this isn't my official rank. But I built this model from scratch and I'm happy with how it held up, so I wanted to write it down.

## Overview

The pipeline is pretty standard:

detect nuclei in every frame → link them over time → find divisions.

What mattered was mostly the details of the detector and how I validated things.

## Detector

- 2.5D UNet with a ConvNeXt-nano encoder (timm). The depth pooling in the decoder is borrowed from the 2.5D design in hengck23's CZII solution.
- Outputs: a Gaussian centre heatmap and a sub-voxel offset.
- Input is 4 channels: frames t-1, t, t+1, plus a local contrast channel. I added the contrast channes I was missing. More on that below.
- Gaussian target with sigma 2.5um. Peaks at least 5um apart.
- Main model ("foldA"): trained 100 epochs on all 128 6bba videos plus half of the 44b6 videos. The o was never used for training, and I used it to judge everything.

I also trained a second detector with a narrower target (sigma 2.2um, 6bba only). I tried about ten cel, and the narrow-sigma one complemented the main model best on held-out videos.

## Combining the two detectors

This took me a while to figure out. Merging the two models' *coordinates* after peak picking always made things worse. Averaging their *probability maps* before peak picking worked, and gave about +0.006 on the LB the first time.

The second model runs in fp16 on the second T4 so both fit in the 9 hours. There's also a time check that drops the second model if the run gets slow. One of my earlier versions with more models in fp32 timed out.

## Linking and divisions

- ILP linking. Edge costs come from a small model trained on the GT tracks.
- Some track clean-up after that: smoothing, gap bridging (up to 12um), removing short and duplicate
- Divisions come from a separate small division model, and each parent is linked to two children. Threshold 0.8.

Small thing that cost me a lot of time: the division threshold had been tuned for an older detector, and it quietly cut divJ in half when I swapped detectors.

Submit integer coordinates. Floats looked +0.002 better locally but were −0.007 on the LB.

## How I validated

Honestly this is probably the most useful part.

The two embryos behave very differently. One filter I had gave +0.004 on 44b6 and −0.036 on 6bba. After that I judged everything on both embryos, using videos the model never trained on, and with the official scorer on the full pipeline output. Quick proxy metrics fooled me more than once. I retracted two "improvements" in one day because of that.

On my 71-video set, anything under ~0.004 was noise, and I tried not to pick the best of many runs.

I think this is why the model went up on private instead of down.

## Things I found out about the data

- The score is mostly about the nodes. With perfect detections, even a simple distance-based linker gets 98% of the edges right. None of my fancier linkers beat the simple one.
- Over half of the "missed" cells weren't really missed. There was a node about 3.35um off, just in the wrong spot.
- Missed cells weren't dark. They were in crowded areas with brighter background. That's why I added t the missed cells, local maxima went from 4.2% to 14.2%.
- Almost all false-positive edges connect two real cells. Only one end is matched to the wrong GT cell. You can't fix that by looking at one edge at a time.
- The labels are sparse, so cells look farther apart than they are. The real nearest-neighbour distance is about 5.8um. My first duplicate-removal radius of 7um was deleting real neighbours.

## What didn't work

This is the long part. Numbers are from my held-out videos unless I say LB.

Detection:
- A DETR-style transformer tracker (queries that detect and track at once). I spent a lot of time on jJ vs ~0.87 for the normal pipeline. It kept producing duplicates, and tracking ate into detection.
- Centre-voting head: −0.008 to −0.016.
- DSNT coordinate head: way too many nodes.
- Keeping full z resolution in the decoder: +0.0007, not worth it.
- Narrower sigma on the main model: fixed some cells and broke more. Worse on both embryos. (It only worked as the second model.)
- Longer temporal context (±3 frames): slightly better J but too many nodes.
- Copy-paste augmentation with max blending: made ghost labels and hurt. Look at your augmentations i

Data:
- I generated about 2,500 fully labelled synthetic sequences. I had to make them denser to look like the real data. Pretraining or fine-tuning on them didn't beat the baseline.
- Zebrahub pretraining: no gain.
- Pseudo-labels: the gains and the drift came from the same unlabelled areas, so I couldn't separate them.

Recovering missed cells:
- Deconvolution of the heatmap: net −3 peaks.
- CLEAN-style residual peeling: found 33% of the missed cells but added 35% more nodes (−0.036).
- Splitting blobs by mass: didn't work (AUC 0.31).
- Learned coordinate refiners: worked on one embryo, didn't transfer to the other.
- FOCUS-3D as a second detector: helped coverage but ~112 s per frame.

Linking and post-processing:
- About nine different linkers (flow-based and others): none beat the simple one.
- 36 post-processing ideas on a fixed graph: zero wins. The ceiling I measured was about +0.003.
- A classifier for bad edges: AUC ~0.57.
- A classifier for extra nodes: AUC 0.9999 on synthetic, 0.59 on real data.
- Filtering dark detections turned out to be overfitting to one embryo. Removing the filter gave +0.028 on the LB, my biggest single jump.
- Capping node count per node: −0.069. Per track: slightly positive.

Divisions:
- The real limit was the division model missing parents. Loosening gates didn't help.
- Rules like "connect track starts to a nearby single-child parent" added hundreds of edges per video and lost 0.034.
- Geometric filters on the daughters didn't work because real divisions look asymmetric at the node l

Ensembling:
- 3–5 detectors: no gain, and one version timed out.
- Separate models per embryo, routed by video name: +0.008 locally, nothing on the LB.
- Training the second model longer made it a worse partner.

## Public vs private

| Version | Public | Private |
|---|---|---|
| foldA + sigma22 (best) | 0.929 | 0.939 |
| foldA + a1long + all-199 model | 0.932 | 0.938 |
| all-199 primary + a1long | 0.932 | 0.938 |
| foldA + anisoxy199 | 0.931 | 0.938 |
| foldA alone | 0.926 | 0.936 |
| foldA + zaniso | 0.938 | 0.928 |
| same, threshold 0.35 | 0.938 | 0.914 |
| routing by embryo | 0.938 | 0.928 |

My highest public scores were my lowest private scores. Everything I had checked on held-out embryos moved up together. Next time I'm keeping at least one honestly validated model in my final two.

Congrats to the winners, and thanks for reading.

---

## Comments (0)

*(none)*
