# 303rd / Bronze: Sub-voxel localization + a learned coordinate head

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744507
- **Topic id**: 744507
- **Author**: Eesh saxena (CONTRIBUTOR)
- **Posted**: 2026-09-30T04:58:48.320396600Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

# 303rd / Bronze: sub-voxel localization, a learned coordinate head, and a lot of dead ends

Bronze, 303 of 4017. Thanks to the organizers, and honestly this is built on other people's public notebooks
more than on anything of mine. I want to be straight about what actually moved the score, because most of what
I tried did nothing, and the failures are probably more useful to you than the two wins.

Short version: refine each detection off the voxel grid, layer a learned coordinate head on top for the
in-plane axes, feed the refined z back into the tracker, tune two knobs in the relinker, and then submit two
versions that place coordinates differently so they don't fail together on unseen embryos. Everything else I
checked, and I checked a lot, either tied or lost on the leaderboard.

## The score progression

| step | public LB | what changed |
|------|-----------|--------------|
| public temporal-UNet3D + ILP | ~0.947 | starting point |
| + sub-voxel soft-argmax (all axes, gain 2) | ~0.951 | refine detections off the grid |
| + learned coordinate head for y/x, refined z to the edge model | 0.953 | the "learned-head" body |
| + relink learned-edge bonus (1.6 to 2.5) | 0.954 | on both bodies |
| + rawlink raw-ILP override (min prob 0.70) | 0.955 | learned-head body, final |

The two selected submissions were the top of each body: the learned-head body at 0.955 and the sub-voxel body
at 0.954.

## The metric, because it drives every design choice

```
score = adjEJ + 0.1 * divJ
adjEJ = sum_i w_i * EJ_i * (1 - 0.1 * rho_i) / sum_i w_i,   w_i = TP_i + FP_i + FN_i
rho_i = (N_pred_i - N_est_i) / N_est_i        (node-count penalty, unclipped)
```

- Edges are matched one to one per frame at a 7 um gate, and only consecutive-frame (dt == 1) edges count. An
  edge between two nodes that were both left unmatched is free.
- `divJ` is a pooled division Jaccard, and in practice it was FN-limited: we were missing divisions, not
  inventing them, so trying to add divisions mostly added false positives.
- The `rho` term punishes over-detection linearly and is not clipped, so pushing node count up to chase recall
  backfires. The metric actually rewards mild under-detection.

Two things kill most offline reasoning, and both are in the data description. The four visible test movies are
placeholders copied from train, so scoring the public split locally predicts nothing. And train and test are
embryo-disjoint: the hidden movies come from embryos no one has seen. So leave-one-embryo-out is the only
sensible proxy, learned components fitted on train transfer worse than they look, and any trick keyed on the
movie prefix is worthless.

## The lever that mattered: sub-voxel localization

The base pipeline emits detections on a downsampled voxel grid, so every coordinate is quantized to the grid.
Instead of taking the argmax voxel as the centre, I do a soft-argmax over the detector logits in a small
window around each peak, with a gain of 2 on the logits and a clip that keeps the shift inside the voxel
(about 0.5), computed separately for the in-plane axes and for z. Concretely: move each detection a fraction of
a voxel toward where the logit mass actually sits, instead of snapping it to the nearest grid point.

The reason this pays on this metric specifically: adjusted edge Jaccard rewards edges whose two endpoints both
fall within the 7 um match gate, so a systematic sub-voxel correction pulls the borderline edges back inside
the gate and converts near-misses into true positives. It moved the all-axis score from about 0.946 to 0.951,
and an ablation that zeroed the in-plane correction and kept only sub-voxel z retained most of that gain, so z
was carrying it.

z is also the weakest part and where I think headroom remains. A 3-point logit soft-argmax overshoots on z
(mean absolute z error is worse than the in-plane axes), and I tried a pile of alternatives (5 and 7 slice
soft-argmax, Gaussian and parabolic fits, lower gains, robust variants) without finding one I trusted more.
A clean z estimator looks like the most obvious remaining gain.

## Two bodies

I ran two versions that share the detector and the tracker but disagree on how coordinates get placed:

- Learned-head body (final 0.955): y/x from the public V1284 learned coordinate head, z from my sub-voxel
  refinement, refined z fed to the edge model, full relinker stack.
- Sub-voxel body (final 0.954): both y/x and z from my sub-voxel soft-argmax at gain 2, same tracker stack.

One detail worth calling out: feeding the refined z back into the edge model, not just writing it to the
submission, was worth about +2 ticks on the learned-head body. Refinement isn't cosmetic. Handing the tracker
the better z changes which edges it decides to trust, and that is where the gain comes from.

I did test blending the two coordinate estimates (half learned head, half sub-voxel for y/x). It scored below
both parents at 0.953, because the two estimates are correlated at roughly 0.8 in y and 0.7 in x, so averaging
them buys almost nothing and just splits the difference.

## Tracker tuning

On top of the public ILP tracker, in the motion relinker:

- Velocity-weighted relink with a tight distance gate. On the 0.955 body the gate sat at 6.5 um; I swept it and
  6.5 was the board optimum, tighter and wider both cost.
- A learned-edge bonus, the reward the relinker gives to high-confidence learned edges. Raising it in steps
  (1.6, then 2.0, then 2.5) each nudged my full-precision standing upward, and the replay agreed with the
  ordering, one of the few times it did.
- rawlink: let raw ILP edges above a probability threshold (0.70) override conflicting final edges, with
  divisions protected. Idea adapted from a public V1057 discussion post. Stacking rawlink on the bonus is what
  took the learned-head body from 0.954 to 0.955.

Final config, for reference: velocity weight 0.25, learned bonus 2.5, relink tight gate 6.5 um, rawlink min
prob 0.70, sub-voxel gain 2.0.

## The part I'd actually pay attention to: validation

Because the public test movies are placeholders, I built a replay harness that reruns the real notebook cells
end to end (real detector outputs, real ILP, the organizers' scorer ported call for call) on GT-rich train
movies, about a minute per movie per config, with paired bootstraps across the two embryos and per-embryo plus
pooled deltas. It let me score a change offline without spending a leaderboard submission.

The single most valuable output was not a ranking of ideas, it was a calibration and a known bias.

- Calibration: board delta was about replay delta minus 0.8 ticks on the sub-voxel body, with a leave-one-out
  RMSE around 0.6. So a change had to clear roughly +0.8 on replay before I'd expect it to even hold serve on
  the board.
- Bias one: the harness runs on trained movies, so it literally cannot see the in-plane localization effect
  that the hidden embryos would feel. The detector is in-sample.
- Bias two: it over-trusts anything that leans harder on the learned edges (rawlink, learned bonus, image
  flow), because the edge model was trained on all 199 train movies. The cross-body check made this obvious:
  my sub-voxel body read about -10 ticks against the learned-head body on replay, and was only about -1 on the
  actual board.

Once I understood those two biases I treated every positive replay number as an upper bound, not a result.
Blunt takeaway: an offline harness on an in-sample detector will call board-transferable localization changes
wrong and over-rate anything that trusts the ILP more. The leaderboard, with its 6 to 8 hour latency and 5
reads a day, was the only instrument that was ever actually right. Plan your submission budget around that.

## What didn't work (the longer and more useful list)

Every one of these looked reasonable, several looked great offline, and all of them died on the board or under
a bias-corrected replay:

- Image-flow / phase-correlation link repair in high-motion windows. My best offline lever all week, about
  +2.9 on replay and positive on LOEO, and it tied or lost on the board every single time. This is what
  taught me the harness's ILP-trust bias.
- Coordinate blending (half learned head, half sub-voxel), 0.953, below both parents, as above.
- ILP division weight 0.4. A public post reported +0.001 from letting the ILP fork more freely. Full-path
  replay showed the extra forks land on noise, net negative, so I did not submit it.
- The public retrained localization head plus z-refiner (coord-inject). Falsified: it makes the >3 um
  displacement tail worse on both embryos and every axis, because it optimizes the board-flat "node within
  2 um" quantity rather than the tail the metric actually pays for.
- Sub-voxel parameter sweeps: temperature grid, gains above 2, per-axis gains, soft-argmax window sizes.
  Around 15 arms, all lost. The production setting is already a board optimum.
- Learned-head gain above 1 (the head under-corrects on one embryo and over-corrects on the other, optimal
  scalar around 0.6, so scaling up hurts), line-fit reweighting, short-track rescue tuning, node-count
  pruning, DET-threshold moves in both directions, MIN_TRACK_LEN, flow-inside-relink, and external models
  (Trackastra, DaXi). All neutral or negative.

If there's one thing to take from this competition, it's that the detector/localization axis and the ILP-trust
axis pull in opposite directions on an in-sample validator, and only the leaderboard resolves which way a
change actually goes.

## Picking the final two

Kaggle keeps the better of your two selected submissions on the private board, so the two picks should fail
independently. On an embryo-disjoint hidden set the failure that scares me is one coordinate scheme
mislocalizing on an unseen embryo, so I picked one submission from each body: the learned-head 0.955 and the
sub-voxel 0.954. Those two differ from each other far more than any same-body pair (a same-body sibling differs
on under 1% of edges and rises and falls with its twin), so they hedge the dominant private-board risk.

A last-day arm (learned bonus 3.0 plus a tighter relink gate) replayed +0.6 but came back 0.954 on the board,
exactly the 0.8-tick calibration gap, so it never threatened the picks.

## Credit

Standing on a lot of public work:

- Detector, tracker and support pack, all from **@pilkwang**:
  `pilkwang/biohub-temporal-unet3d-seed314159-v1`, `pilkwang/biohub-tracking-support-pack-50ep-v1`,
  `pilkwang/biohub-deepcenter-unet3d-center-prior-v1`.
- Learned coordinate head (the "V1284" head): **@anvithpothula**, `anvithpothula/biohub-v1284-head-s075`.
- The rawlink raw-ILP override idea came from a public "V1057" discussion post by **@josephadamski91**.
- The organizers' scorer, which I ported call for call to score offline.

Glad to go deeper on the sub-voxel refinement or the replay-versus-board calibration if it's useful to anyone.

---

## Comments (0)

*(none)*
