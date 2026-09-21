#  Public 0.947-stack weights were trained on all 199 train videos — any hold-out from train is in-sample (manifest check + an LB counter-example)

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742064
- **Topic id**: 742064
- **Author**: Eric (CONTRIBUTOR)
- **Posted**: 2026-09-19T13:00:01.979783200Z
- **Votes**: 2
- **Comments**: 0

---

## Opening post

I checked the split_manifest.json shipped with the public secondary model (pilkwang/biohub-temporal-unet3d-seed314159-v1, weights/unet_transformer/split_0/): the method is unet_transformer_alltrain_seed314159_v1 and the train list contains all 199 videos; the 40 "test" videos in that manifest are a subset of them. The DeepCenter manifest (pilkwang/biohub-deepcenter-unet3d-center-prior-v1) lists 71 train videos. The 50ep pack itself ships no manifest, so I can't verify it, but others here have reported the same all-train setup.
Consequence: any local hold-out drawn from the 199 train videos is in-sample for the detectors in the public stack. Steps that correct detector errors look good locally because the detector has memorised those videos, so there is little left to correct.
A concrete example from my runs: a sub-voxel peak refinement gained +0.0078 on the 12 official validation videos that contain divisions (official metric, CSV round-trip), but −0.002 on the public LB, consistently across two submissions.
What I now trust locally: only models that never saw the evaluation videos (e.g. a detector trained with those videos held out). Everything else I decide on the LB.
Has anyone found a local setup that tracks the LB for the public stack?

---

## Comments (0)

*(none)*
