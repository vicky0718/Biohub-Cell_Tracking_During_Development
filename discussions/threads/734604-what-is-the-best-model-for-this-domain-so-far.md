# What is the best model for this domain so far?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734604
- **Topic id**: 734604
- **Author**: Moawiz (CONTRIBUTOR)
- **Posted**: 2026-08-12T05:01:34.624924500Z
- **Votes**: 1
- **Comments**: 2

---

## Opening post

I have tried pilkwang model family (3d Unet) so far, without further fine tuning but it seem to be stuck at a certain score.
I have tried a lot of post processing techniques, it does push score locally but LD score drops.
im not sure if its a Detector problem if yes what models should i use before jumping into training/fine-tuning.

---

## Comments (2)


### Mendrika Ramarlina (MASTER) — 2026-08-12T05:17:31.760Z — 1 votes

I would first decompose the score before switching models.

The metric:

**Adjusted edge Jaccard + 0.1 × division Jaccard**

An edge is correct only if both cells are detected and then linked correctly. A useful approximation is:

**Edge recall ≈ node recall² × conditional linking accuracy**

This means detector misses are especially expensive. Divisions are even more demanding because the parent and both daughters must be detected before a linker or division ranker can recover the event.

I would measure four things separately:

1. **Detection:** node recall within the 7 µm matching radius and predicted-node count versus the provided estimated count.
2. **Linking:** linking accuracy only where both GT endpoints were successfully detected.
3. **Detection ceiling:** the best possible edge score using your current detections with an oracle or GT-aware linker diagnostic.
4. **Divisions:** parent/daughter detection availability, candidate recall@K, ranking quality, and final division TP/FP/FN.

Then keep the detector fixed while comparing linkers, and keep the linker fixed while comparing detectors. Otherwise, post-processing can appear to improve locally by changing the candidate set and graph simultaneously.

From my experiments, Pilkwang's UNet/node-transformer is still a very strong backbone. 

If conditional linking is already high, I would focus on detector calibration/precision and division ranking before spending much more time on a new linker.

#### ↳ Moawiz (CONTRIBUTOR) — 2026-08-12T14:54:00.887Z

> the detector is finding the real cells, but it is also producing an enormous junk candidate pool. The linker then has to choose among those candidates which ends up with wrong linking
