# Model changes don't move the LB above the public plateau — what actually did for you?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/741749
- **Topic id**: 741749
- **Author**: hikaggler (EXPERT)
- **Posted**: 2026-09-17T06:31:17.691927Z
- **Votes**: 5
- **Comments**: 7

---

## Opening post

Hi all. Over the last three weeks I've been trying to get above the ~0.947 public-notebook plateau by changing the detector and linker, measuring each change one variable at a time on a 24-video hold-out stratified across both embryo lineages. What surprised me is that almost nothing on the model side moves the score. I'd like to hear from people who saw the same thing, or who found something that did work.

What did NOT move (hold-out adj_edge_jaccard, paired SE ≈ 0.006):
- Deeper / wider U-Net (width 32 → 48, 4.0M params): no gain. Deeper nets inflate the node count and lose on the N_pred term
- Longer training (60 → 131 epochs): edge Jaccard goes up, the node-count term eats it
- Strong augmentation, Gaussian heatmap loss, local-softmax peak loss: neutral or negative
- Wider temporal window (2 → 9 frames): recovers ~0.1% of GT nodes
- A trained division head (pair softmax over candidate daughters): local division Jaccard 0.07 → 0.19, but ±0.001 on the LB
- Post-processing knobs (gap closing, short-track removal, thresholds): saturated, same as the public sweeps

What DID move:
- Pretraining on the public synthetic dataset (+0.012–0.018 over from-scratch). This was the only big lever I found
- Two-model fusion, but only when conditional: use the second model only on columns where the primary is uncertain. Plain averaging does nothing

Questions:
1. Where do you think the 0.947 → 0.955+ gap actually lives: detector, linker, or graph construction?
2. How well does your local CV track the LB? In the 0.93–0.95 band mine has r ≈ −0.2, so I can't rank weight-level changes locally at all
3. If a model change (not post-processing) moved your LB, I'd be grateful to hear even just what kind of change it was

Happy to share more numbers on any of the above if useful.

---

## Comments (7)


### MOHAMMADJAFAR ZAMANI (CONTRIBUTOR) — 2026-09-18T08:18:39.780Z

Thanks for starting this thread — this is close to the problem I’m seeing. My best public score is 0.947 on the published U-Net + transformer/graph stack. I tested controlled temporal-context changes to detector logits and association variants. The effects are inconsistent across the two image-source groups: a change can suppress false detections in one group while reducing valid peaks or associations in the other, so I have not found a defensible local gain to promote.

For people who broke past this plateau, which diagnostic was most predictive of LB movement: node-count/calibration error, missed detections, edge assignment, or division handling? Did you need a different detector/pretraining regime, or could a carefully gated change to the published stack do it? I’m especially interested in validation designs that correlate with the LB without tuning repeatedly on the same embryos. High-level guidance is enough.

#### ↳ hikaggler (EXPERT) — 2026-09-18T11:03:17.987Z

> 
> @MOHAMMADJAFAR ZAMANI — at a high level: of the four you list, node-count calibration predicted my LB movement far better than missed detections did. I can change recall by a point and the LB does not react, while the N_pred ratio term shows up every time. What you describe — a change that removes false detections in one group and valid peaks in the other — is what I see as well, so I keep a hold-out stratified across both lineages, fix it once and never retune on it, and compare runs with a paired standard error rather than a difference of means. The part I would stress most: post-processing settings chosen locally do carry over to the LB, but the local ranking of two trained models does not, so I stopped choosing weights that way. On your other question, the pretraining regime was the only change on my side that clearly moved anything; carefully gated changes on top of the published stack gained me almost nothing.

### TWEAK (EXPERT) — 2026-09-17T17:48:19.580Z

Could you tell us more about "Pretraining on the public synthetic dataset (+0.012–0.018 over from-scratch). This was the only big lever I found"? What is your model/models base score before you add in your division and division head? Are you using anything from the public notebooks that would cause your model/models to converge on 0.947?

#### ↳ hikaggler (EXPERT) — 2026-09-18T07:21:37.373Z — 1 votes

> Fair question, so let me be explicit: the 0.947 next to my name is a public notebook score, not mine. My own pipeline, run end to end by me, is at 0.939. It's built on the host's published baseline — the temporal 3D U-Net detector, the transformer edge scorer and the ILP from the official repo — not on the public notebook chain. That's also why the "model changes don't move the LB" observation in my post is about the 0.939, not the 0.947.
> 
> 　Base scores: a single detector with my own linker and post-processing, no fusion and no division head, is around 0.92. Probability-level fusion of three detectors gets most of the way to 0.939, and the pair-softmax division head adds at most ~0.002 on top — locally it lifts division_jaccard from 0.07 to 0.19 on a 124-video OOF, which barely shows up on the LB. For scale, an ablation with division creation switched off cost me 0.022, so my division_jaccard is roughly 0.22.
> 　
> 　On the synthetic pretraining: pretrain detector and linker on the public CC0 synthetic set for 80 epochs, then fine-tune on the 175 real videos for 60 epochs with the same recipe I use from scratch. On a 24-video hold-out that's 0.9146 → 0.9269. Three caveats: it pushes the node count up, so part of the gain is paid back through the N_pred term; it buys nothing on divisions (the generator doesn't model mitosis over time); and more synthetic data isn't obviously better — transfer to real data peaks very early and then decays.
> 
> 　And a question back, if you don't mind: did you get to 0.957 on top of the host's baseline pipeline, or is it a different pipeline altogether? I'm not asking for the recipe — just whether the remaining 0.018 is reachable from the published U-Net + transformer + ILP at all, or whether people up there have replaced it. That alone decides how I spend the last eleven days.

#### ↳ ↳ Ogurtsov (MASTER) — 2026-09-18T10:13:21.050Z

> > Could you please clarify what public synthetic data did you use?
> > 
> > >whether the remaining 0.018 is reachable from the published U-Net
> > 
> > It's reachable to some extent but pretty risky.

#### ↳ ↳ hikaggler (EXPERT) — 2026-09-18T11:03:22.247Z — 1 votes

> > @Ogurtsov — it's the CC0 synthetic set that José Freitas shared in this competition's discussion (the output of his biohub-synthetic-dataset notebook). It contains 1,539 static volumes and 2,174 time sequences. I used only the sequences, since I pretrain the detector and the linker together and the static volumes have no time axis — and in fact only 497 of those 2,174, for 80 epochs, before fine-tuning on the 175 real videos. Yesterday I repeated the pretraining with all 2,174 at the same number of gradient steps, and it came out slightly worse on my hold-out (adj −0.005). So what helps is having some of that data, not having more of it.
> > 
> > About "risky" — do you mean risky for the private LB, or risky to train, as in unstable and sensitive to which checkpoint you keep?

#### ↳ ↳ Ogurtsov (MASTER) — 2026-09-18T11:48:48.723Z

> > Thanks! My primary concern is private LB.
