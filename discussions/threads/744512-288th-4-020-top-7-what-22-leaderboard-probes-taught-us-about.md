# 288th / 4,020 (top 7%): what 22 leaderboard probes taught us about this metric

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744512
- **Topic id**: 744512
- **Author**: Santanu Banerjee (CONTRIBUTOR)
- **Posted**: 2026-09-30T05:14:37.597105600Z
- **Votes**: 0
- **Comments**: 0

---

## Opening post

**Final: private 0.921 (public 0.955), rank 288 of 4,020.** Our final submission is a public notebook
plus two environment switches, so this is not a new model. It is a set of measurements that may save
someone time: what moved the private score, what only moved the public one, and a few facts about the
metric we only found by reading its source. All code, every submission with its public and private
score, and the day-by-day notes are here: **https://github.com/SanTanBan/biohub-cell-tracking**

### TL;DR

- **The biggest gain was re-basing onto the public ceiling.** Forking `anvithpothula/biohub-x138`
  (public 0.953) was worth +0.007 private, more than two weeks of knob probing on the older 0.940
  notebook combined.
- **Public differences of ≤ 0.002 did not predict private ones**, in either direction (table below).
- **In the public pipeline's ILP preset, the solver can never produce a division.** In `tracksdata`
  the division weight is a *penalty*, and with free track appearances a fork never pays. Fixing it
  cost us 0.013, because the solver's forks were mostly wrong.
- **Local CV on train movies is in-sample for the public models** (they were trained on all 199), so
  it cannot rank changes that shift trust between the learned links and the heuristics.

### Results (public → private)

| submission | public | private |
|---|---|---|
| x138 + re-admit 0.90 + detection threshold 0.955 | 0.955 | **0.921** |
| x138 + re-admit 0.90 + DeepCenter division veto 0.15 | 0.955 | **0.921** |
| x138 + re-admit 0.90 | 0.955 | 0.919 |
| x138 + re-admit 0.85 | 0.954 | 0.919 |
| x138 + flow re-link tight gate 8 µm | 0.951 | 0.919 |
| x138, exact fork | 0.953 | 0.917 |
| public 0.940 + candidate edge threshold 0.25 | 0.938 | 0.913 |
| public 0.940, exact fork | 0.940 | 0.910 |
| public 0.940 + ILP division weight 0.6 | 0.926 | 0.898 |
| our own detector + linker | 0.786 | 0.760 |

Two pairs to notice. Detection 0.955 and the looser division veto tied re-admit 0.90 on public and
beat it by 0.002 on private. The candidate-threshold probe lost 0.002 on public against its base and
won 0.003 on private. With ~29% of the test set on the public board, a single 0.001-0.002 is not evidence.

### 1. Our own stack: 0.786 public / 0.760 private

- **Detector:** a temporal 3D UNet predicting a centre heatmap on a 4× XY-pooled grid (isotropic at
  1.625 µm). It uses a CenterNet-style focal loss with *ignore regions* around unannotated nuclei,
  because sparse GT means "no label" is not "background", and parabolic sub-voxel refinement on the logits.
- **Linker:** Hungarian assignment in physical µm with a *null column per target* and a squared cost,
  velocity extrapolation (coefficient 0.3, the measured autocorrelation of GT displacements), and
  physical NMS at 6 µm.
- **A local official-metric harness** (the `tracking_cellmot` scorer run on cached detections). It
  found six silent bugs, each worth more than any hyper-parameter:
  - a loss with no negatives (`<` vs `<=` on a flat background);
  - centroid sub-voxel refinement that was worse than none;
  - a linker forced to make `min(n, m)` matches, where 116 of 143 missed edges had been stolen by a distractor;
  - checkpoint selection on recall instead of recall², which picked epoch 3 of 22;
  - NMS that never ran in production;
  - a unit double-scaling in the harness itself.

  The last one produced two wrong recommendations before we caught it. Synthetic tests passed
  because they were self-consistent in the wrong units.
- **Decomposition:** with *perfect* detection of the annotated cells our linker reached 0.856
  adjusted edge J. With the public detector it reached 0.825, and with ours 0.673. The bottleneck was
  detector precision, and fixing it needed more GPU than our shared 30 h/week quota allowed. So we
  moved to the public pipeline.

### 2. Probing the public pipeline

One environment switch per submission on the public 0.940 notebook. Each was checked first with the
official scorer on exported ILP graphs of 36 train movies, and on 8 movies before that. The 8-movie
sweeps misled us once (min track length 10), and the 36-movie sweep predicted that loss.

- **Motion re-link off** gained +0.020 adjusted J locally and lost 0.001 on public. This is where the
  in-sample lesson came from: learned links look better than they are on movies the models were trained on.
- **The ILP never divides.** There were zero forks in all 36 raw ILP graphs. `tracksdata` *minimises*
  cost, and at the preset `division_weight=1.2`, `appearance_weight=0`, a second daughter starting a
  new track is free while a division costs more than the extra edge can earn. The preset's comment
  says the value was raised to encourage divisions, which is the opposite of what it does. Every
  division in these submissions comes from a geometric repair stage.
  We lowered the weight to 0.6, and 473 new ILP forks cost 0.013 on public. They displaced correct
  continuation links, and false forks were far more expensive on the hidden set than on sparse local GT.
- **Division headroom:** over 49 local GT divisions, the parent was detected 49 times, both daughters
  36 times, inside the repair stage's geometry 25 times, and recovered 10 times. Loosening the geometric
  gates added 3 true forks for 55 false ones. Candidate edge probabilities never reached the ~11%
  precision a fork needs to pay here.
- **Edge errors** on the public pipeline (its own four validator movies): 95.6% of GT edges recovered, 2.4% fragmented (both ends
  detected, no link), 2.0% lost to detection, and essentially zero wrong associations. The links it
  makes are right; the ones it misses are the problem.
- Detection threshold, gap-close radius, their stack, and the output track length all tied or lost on public.

### 3. Re-basing on x138, and what won

x138 adds four things to the same pipeline: a neighbourhood-flow prior in the motion re-link,
**re-admission** of detector peaks the ILP discarded next to open track ends, a **gap-filler** that
bridges 1-3 frame breaks only through real sub-threshold peaks, and a small learned head that
refines each detection centre by ≤ 2 µm. We read its code for scorer tricks before adopting it and
found none. Our fork reproduced the author's local score exactly.

- **Re-admit threshold 0.965 → 0.90** was the one clear public win (+0.002). Weaker *real* peaks
  close to where a track stops pay off. Reaching further (4 → 6 µm) lost, and going weaker (0.85) lost.
- **Detection threshold 0.965 → 0.955** and a **looser DeepCenter division veto (0.25 → 0.15)** both
  tied on public and gave our best private score. The veto change follows the x138 author's own
  measurement: over 50 movies this pipeline family found 17 true divisions against 19 false and 56 missed.
  That is ~47% precision against an ~11% break-even, so the veto was too strict.
- A wider tight gate in the flow re-link (7 → 8 µm, beyond the ~7.6 µm cell spacing) lost 0.004 on
  public. The loss came entirely from the densest movie, where neighbours stole each other's links.

### 4. Metric notes that may help others

- An edge counts only if its source matched an annotated node with a successor, or its target one
  with a predecessor. **Edges between unannotated cells are free**, however wrong.
- A wrong link is worse than no link: link iff P(correct) > J / (1 + J) ≈ 0.48.
- Adding forks of precision *p* raises division J only if p > TP / (2·TP + FP + FN). That is ~11% for
  a pipeline at 10 TP / 29 FP / 39 FN, but the hidden set punished extra forks far more than that suggests.
- The node-count term `J × (1 − 0.1 × ratio)` has no ceiling. Shedding short tracks for it (min track
  length 6 → 10) lost 0.002, because those tracks still carry scoring edges.

### 5. What we'd do differently

- **Track the public ceiling from day one** (`kaggle kernels list --competition … --sort-by
  scoreDescending`) and re-base the moment it moves.
- **Don't promote on +0.001 public.** Keep mechanism-backed ties in the final stack, and keep one
  robust pick.
- **Spend GPU on models, not post-processing.** The whole public lineage lands at ~0.92 private, while
  the top 20 hold 0.94-0.98. That gap looks like better detection and linking, not better thresholds.
- **Automation:** our daily probe loop was run by an AI coding agent (Claude Code) on a schedule. It
  worked when the machine was on. Four of the last six days it wasn't, which cost 21 submission slots,
  more than any knob was worth. For code-competition submissions, the Kaggle CLI's success text is
  `N submissions remaining today.`, and "did it submit?" is answered by `kaggle competitions submissions`, not a log.

### Acknowledgements

Thanks to **pilkwang** for the public pipeline, models and datasets everything here builds on;
**nusrati** for the 0.940 notebook; **anvithpothula** for x138, the V1284 head and the division-QC
measurements; the **Biohub** team for the data and the `tracking_cellmot` metric; and everyone in
the discussion forum who shared what they measured.

Code, notes and the full public/private table: https://github.com/SanTanBan/biohub-cell-tracking

This writeup is also available at: https://github.com/SanTanBan/biohub-cell-tracking/blob/main/WRITEUP.md

---

## Comments (0)

*(none)*
