# Evidence behind the playbook

Measured outcomes only. Each item says where it comes from, so a rule can be weighed rather
than obeyed. Biohub discussion ids resolve at
`kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/<id>`; Rogii
writeups are under `kaggle.com/competitions/rogii-wellbore-geology-prediction/writeups`.

## Contents

1. Our two competitions
2. Forking has a ceiling; building is what cleared it
3. Validation: in-sample instruments and the public board
4. Diagnosed defects and reframing
5. Scouting and missing dependencies
6. Endgame and operations
7. AI agents among the winners
8. The Biohub top 25, in numbers

## 1. Our two competitions

**Rogii — Wellbore Geology Prediction (Jun–Aug 2026, RMSE, lower is better).**
Our notes are in the `rogii` repo, `chat/memory/`.
- The best submission was 7.182 public, from a notebook forked in full. The ~7.1 public cluster
  was a trap. Everyone forked one notebook, and its score was mostly seed noise: byte-identical
  reruns scored 7.09, 7.13 and 7.09.
- The score was also propped up by a "guarded contact override". It exploited wells that
  overlap train and test: one competitor measured 0.005 ft on overlap wells against 6.47 ft
  without.
- We then built an honest harness: GroupKFold by well, plus leave-spatial-block-out.
  - It measured the fork's leakage-free core at 10.38. Our notes put the gap to the public
    7.18 down to leakage and leaderboard luck. The fork's learned branch wasn't part of that
    core.
  - Recalibrating the reference well ("typewell") and blending improved that to 9.92, with the
    gain holding in all six spatial blocks.
  - The public leaders were at ~5.3, level with the oracle floor our harness measured (5.31).
- The highest-ceiling route, a per-well CNN of our own, was scheduled as "phase 3", after
  improving the fork. The repo's later notebooks (August) are still built on the fork.
- Rogii 1st place finished at 5.639 private.

**Biohub — Cell Tracking During Development (Sep 2026).** Our notes are in the Biohub repo,
`notes/75`–`notes/95`. The score was edge Jaccard, adjusted for node count, plus 0.1 × division
Jaccard.
- Final result: `rd07`, 0.954 public / 0.9178 private. That was 1041st when the competition
  closed, and 1010th of 3,947 after Kaggle's post-close re-ranking.
- **325 teams scored exactly 0.91780**: in effect the same predictions, the mark of a forked
  lineage. On the public board, 704 teams were tied at 0.947 mid-competition.
- **Parameter sweeps moved nothing:** three weeks and ~20 arms of sweeps on the forked
  pipeline's parameters all landed at 0.943–0.946. Among them, a learned relink bonus had five
  offline points of support; it scored 0.946, identical to the base.
- **Both gains came from other people's code**, found by by-author scouting:
  - `flow2`, +0.001;
  - `rd07`, +0.007: 25 added keys and the code behind them, and not one of the 57 values we had
    swept.

## 2. Forking has a ceiling; building is what cleared it

- **Ceiling (Biohub):** forkers put the fork-and-tune ceiling at ~0.954 public / 0.924 private,
  161st of 4,020 (744498). The 288th-place team stalled at 0.921 private after forking.
- **From scratch (Biohub, 744485):** one competitor built a 2.5D ConvNeXt detector, an ILP
  tracker and a small division model, trained on 1.5 embryos. It scored 0.939 private,
  about 22nd. They didn't select it. Synthetic and ZebraHub pretraining gave them nothing.
- **Biohub 3rd (744484):** they reused a backbone from their previous competition and built
  their own detector, flow, matcher and division models. OOF CV 0.978, private 0.967.
- **Rogii 1st:** they built on a public particle-filter baseline but framed the task as 2D
  alignment with a neural network. They credit Tucker Arrants with motivating them "to explore
  neural networks early in the competition".
- **Rogii 22nd:** their writeup's "solo climb" went from 9.91 to 5.83.

## 3. Validation: in-sample instruments and the public board

- **In-sample by construction (Biohub):** the public checkpoint had seen all 199 training
  videos, so all five validators we built on it were in-sample.
  - One of them called a change +0.034; the board scored it −0.001.
  - The local proxy ranked arms 0-for-4.
  - Biohub 12th (744501) made "every number names its ruler" their first rule, after their
    in-sample bench over-read private by 0.03.
- **Biohub test composition (3rd place, from probing):** the public board is one new embryo (60
  videos) and private is another (106 videos), with different cell density. The public board
  was therefore a single group.
- **Public inverted private within one lineage (Biohub, 744548):** korokke3 ran five
  configurations of our exact lineage.
  - The public ranking was almost exactly the reverse of the private one: their worst public
    (0.948) was their best private (0.924).
  - Leave-one-embryo-out CV had picked it. They overrode CV with public tuning and missed
    bronze by 0.001.
- **Rogii, four writeups:**
  - 5th: "Public and private are almost anti-correlated, while CV and private rank-correlate
    very well."
  - 20th private / 3rd public: CV correlated +0.44 with private, public −0.16. "Chasing public
    LB would have actively steered me the wrong way … On the final private LB (150 wells), the
    CV ordering was the right one."
  - 22nd: all 50 of their submissions scored worse on private than public (+1.47 ± 0.95 ft).
    Inside the endgame band, public and private were anti-correlated (Spearman −0.48), while
    773-well CV correlated +0.84 with private. "The submission our public-LB evidence told us to
    reject was our best private score."
  - 24th: "The public LB was actively anti-correlated for us."
- **Score the shippable path (Rogii 7th):** "Every stage is scored the same way: GroupKFold(5,
  seed 42) out-of-fold over all 773 training wells, through the shippable inference path."
- **The CV-to-private gap can be steered by (Biohub 3rd):** OOF 0.978 against private 0.967, a
  stable gap.

## 4. Diagnosed defects and reframing

- **Biohub divisions.** We proved that the public ILP could never form a cell division, and
  that a later relink stage discarded the ILP's edges. We then filed divisions as a closed axis.
  - 18th (744531) started from the same public notebook. Its inherited pipeline recovered 23 of
    151 training divisions, and they rebuilt edge selection and division placement.
  - 3rd place's ablation:
    - detection + Hungarian linking: 0.890, division Jaccard 0.000;
    - + division models, calibration, and joint link/division optimisation: 0.964 (+0.063),
      division Jaccard 0.535.
  - 5th (744549) gave the linker a division head and made the ILP's split cost depend on it.
- **Biohub localisation.** Annotation z-offsets differ by embryo: +0.675 of a plane in one,
  +0.115 in another (12th; korokke3 found +0.17 µm vs +0.84 µm). 3rd place's affine coordinate
  refinement was worth +0.006. None of our arms looked at localisation.
- **Biohub data quirks (12th):** 80 byte-identical pairs of overlapping crops.
- **Biohub metric surrogate (3rd):** they built a surrogate of the metric and solved an LP over
  it, including the probability that a prediction is evaluated at all under sparse labels.
- **Rogii reframing:** 1st place framed the task as 2D alignment, and 23rd as "whole-well
  geological alignment" with a distributional U-Net. Both replaced per-row regression.

## 5. Scouting and missing dependencies

- **Scouting by title vs by author (Biohub).** Scouting by title concluded "nothing runnable
  above the 0.947 plateau". The by-author join found six accounts between 0.948 and 0.957
  publishing working forks. In one afternoon, at zero submission cost, it:
  - closed an axis we were re-running, because another author had already spent four
    submissions on it;
  - produced our first gain;
  - produced the diff that showed what `rd07` adds.
- **Missing dependency (Biohub).** On 09-22 we closed `x138`, a notebook by an author scoring
  0.956, on `RuntimeError: ('my V1284 head mount mismatch', [])`, assuming the head was private.
  - It was public: `anvithpothula/biohub-v1284-head-s075`, updated 09-20. It could also be
    reproduced from the notebook's own capture mode, and a self-trained head scored 0.954
    (744498).
  - A notebook that mounts it became our best submission a week later.
  - `scripts/scout.py find v1284` returns that dataset as its first hit, and the notebook that
    mounts it among the notebook hits.
- **Rogii, in hindsight.** The same join, run on Rogii after the close, lists notebooks last run
  in mid-July by authors who finished at 5.6–5.7 public. Our best that month was the 7.18 fork.
  An author's final score is not necessarily that notebook's score, so this shows where to
  look, not what those notebooks scored.

## 6. Endgame and operations

- **Quota (Biohub).** Graded reruns draw on the weekly GPU quota. With it spent, every
  submission failed for four days in the final week. Earlier we had argued they wouldn't.
- **Runtime (Biohub).** A notebook with a fixed 24-candidate sweep (~3 h, which doesn't shrink
  under grading) timed out three graded reruns in a row. Its inference, meanwhile, scales ~17×
  on the hidden test. The risk had been flagged before the first two.
- **Accelerator (Biohub).** Switching a CUDA-only notebook to TPU removed the GPU, and three
  submissions died on the first line.
- **Format (Biohub, 744093).** Writing float centroids where the sample used integers cost one
  team 0.008 public.
- **Selection (Biohub).**
  - `flow2` was selected manually while `rd07` was still grading, and the second slot was left
    empty.
  - Kaggle filled it with the best public submission, `rd07`: 0.917 private against `flow2`'s
    0.913, about 570 places.
  - The empty slot helped only because a run finished grading after the manual pick.
- **Watchers (Biohub).** On five separate occasions, rate limits, proxy refusals, busy slots
  and a spent quota were reported as results about the work. A 429 once marked a running
  notebook as FAILED.
- **Log arithmetic (Biohub).** Summing per-pass log lines across runs with different numbers of
  passes produced a false node-count gap. Each run's own submission summary had the right
  figure.

## 7. AI agents among the winners (Rogii)

- 1st: "Most of the code was written with Codex using GPT-5.5/5.6."
- 7th: "HMM + UNet (agent is all you need)", built by agents.
- 14th: Codex with GPT-5.6, then Claude Code for the last two days. "I made the calls about
  what to try and what to ship."
- 23rd: "Kaggle Agent Solution".
- 29th: Codex "handled all of the implementation, parameter fine-tuning, pipeline blending,
  and most of the ideation".
- 36th: "codex --yolo continuously for two weeks with access to 2xGPU L4".

What distinguished these agent-built finishes was what the agent was pointed at: building and
validating models on grouped OOF CV.

## 8. The Biohub top 25, in numbers

The full digest is in the `rogii` repo at `winning_writeups/biohub_top25/README.md`. It covers
17 writeups and the private leaderboard as a CSV. These are the figures behind the rules.

- **Divisions were the largest single lever:**
  - 4th: a learned per-node division cost in the ILP, +0.035 on both boards.
  - 5th: new-cell and division heads, +0.024 private.
  - 7th: division costs, private 0.910 → 0.937.
  - 9th: re-adding the daughters the public ILP never produced, +0.013 private.
  - 23rd: a second-daughter classifier, +0.010 private.
- **Count the scarce event at every stage.** The public pipeline recovered 23 of 151 training
  divisions (18th), and its ILP produced zero forks on every movie (9th, 12th, 14th, 23rd).
- **Own models vs in-sample public checkpoints:**
  - 9th: own detectors and linker were worth +0.004 public but **+0.023 private**.
  - 12th: an in-sample bench over-read private by 0.03.
  - Teams with their own models climbed from public 191st / 1087th / 1067th to private
    10th / 11th / 20th.
- **The public board is one embryo:**
  - The private top 25 had public ranks from 1st to 1,087th (median 30th).
  - Among small late changes, public and private correlated at r = 0.09, and CV and private at
    r = 0.34 (18th).
  - A seed change alone moved public 0.006 and private 0.001 (16th).
- **Validation that held:**
  - 17th designed changes on half the movies and confirmed each once on the other half: 7 of 19
    design-half wins were rejected.
  - 18th: gains of +0.005 to +0.012 on 8–16-video checks became −0.001 to −0.003 on all 199
    videos.
- **Post-hoc edits.** 17th labelled candidate edits by their effect on the metric, not by a
  semantic label, and those edits held up on confirmation.
- **The writer.** An int16 cast that truncated instead of rounding cost 9th 0.013 private.
  Integers vs floats: 0.950 vs 0.942 public (12th).
- **Final selection:**
  - 4th's, 6th's, 16th's and 20th's best private submissions were not among their selected
    two.
  - 23rd's second-slot hedge, a z+1 shift of every node, lost 0.003 public, gained 0.005
    private, and was their best.
- **Agents:** 10th (Claude, Codex and a Kaggle agent), 11th, 12th (Claude Code, more than 4,200
  sub-agent runs) and 23rd (Claude Code and Codex, under pre-registered rules).

