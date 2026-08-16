# Post-patch: is the 0.91+ frontier separated by the edge term or by divisions?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728551
- **Topic id**: 728551
- **Author**: Arul Prasad S P (CONTRIBUTOR)
- **Posted**: 2026-07-23T18:08:45.090986600Z
- **Votes**: 6
- **Comments**: 2

---

## Opening post

Trying to reconcile something after the division-metric patch — would love the community's (and host's) read.

Across the public approaches I can score with the patched metric, two things look consistent:

- **adjusted_edge_jaccard** plateaus around ~0.90–0.91 — the field is on the same split_0 stack, and the post-proc knobs (disappearance cost, D4/9-way TTA, gap-closing) seem to top out there.
- **division_jaccard ≈ 0** for essentially everyone under the patch.

Yet the post-patch top is **0.914–0.929**, so the separation comes from somewhere. For those above ~0.914 — not asking for methods — is your gain on the **edge term** (a genuinely better/retrained model pushing adj_edge past ~0.91), or on **divisions** (getting div_jaccard up to ~0.15–0.2 that the patch accepts)? Just which term, if you're willing to say.

**A finding to give back:** I tried the host's HOCT (general_v0) as a division-aware tracker by feeding my own point detections in as synthetic 3D balls (to get around it needing instance masks). On my train movies it under-performed a tuned ILP on the edge term (it over-links), produced spurious forks rather than recovering real divisions, and ran ~45 min/movie on the open-source solver — so it's not submission-viable as-is with this input. Almost certainly the uniform-ball masks being OOD for a model trained on real morphology, not a knock on HOCT itself. Sharing in case it saves someone the same detour; happy to detail the setup. Or if someone has tried this and worked out that would be helpful too.

My hunch is that a cleaner edge model would let the ILP form divisions natively (divisions riding on long-range link recall) — i.e. the two terms aren't independent. Curious whether that matches what the frontier teams see.

---

## Comments (2)


### mikelou1 (EXPERT) — 2026-07-28T19:01:40.093Z

I don't know how I did in edge division but I think its something like 0.2

#### ↳ Arul Prasad S P (CONTRIBUTOR) — 2026-07-29T03:23:53.583Z

> 0.2 on divisions?
