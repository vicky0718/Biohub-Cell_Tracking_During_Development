---
name: kaggle-competition-playbook
description: How to run a Kaggle competition so it finishes well — day-one setup, cross-validation, where to spend the weeks, and the endgame — distilled from our own two competitions (Rogii wellbore geology, Biohub cell tracking), where forked public notebooks capped us mid-table, and from the winners' writeups of both. Use it whenever you start, plan, advise on, or work in a Kaggle competition — choosing an approach, forking or judging public notebooks, designing CV, a CV/leaderboard disagreement, a score stuck on a plateau, a notebook failing on a missing dataset or weights, GPU quota or runtime limits, or picking final submissions. Use it even when the user only mentions a competition name, an LB score, "public notebook", "CV", "shake-up" or "final picks".
---

# Kaggle Competition Playbook

## What this is for

In both competitions we ran (Rogii wellbore geology and Biohub cell tracking, 2026) our best
submission was a fork of a public notebook, and the fork's ceiling set our finish: mid-table,
inside a crowd tied on the same score. The winners of both did three things we didn't:

1. **built their own model** for the part of the problem with the most headroom, often after
   reframing the problem;
2. **validated on out-of-fold predictions, grouped by the unit the test is split on**;
3. **chose final submissions on that CV**, not on the public leaderboard.

Several of Rogii's top finishers had AI coding agents do most of the implementation, so none of
this is out of reach. What matters is what the agent's hours are pointed at.

This playbook covers the process failures that sank otherwise good work. **It is not the
modelling.** The score comes from understanding the problem — the domain, the data-generating
process, the metric, and what the public pipeline gets wrong — and every winning move we studied
was domain-specific (re-casting per-row regression as aligning two sequences; rebuilding a
cell-division stage; correcting a per-embryo annotation offset). Use the playbook to protect
time and attention for that thinking, not to replace it.

## When advising the user

- **Lead with their problem.** Start from their domain, data and metric, and give real
  domain substance: preprocessing and artifacts specific to the modality, known failure modes,
  relevant pretrained models or external data, how past Kaggle competitions on similar data were
  won. Then add the points from this playbook they are likely to be missing. If the checklist is
  crowding out domain content, the balance is wrong.
- **Our history is calibration for you, not content for them.** Don't retell our competitions.
  If a past result changes the decision in front of them, one clause is enough.
- **Keep unknowns labelled as unknowns.** How their hidden test is split, and what public versus
  private covers, are assumptions until verified: say which assumption you're making, what
  changes if it's wrong, and how to check it (data page, host posts, adversarial validation).
- **Answer narrow questions narrowly.** "Which two do we pick, two days left" gets the pick and
  the reasoning. Suggest a new experiment only if it can be validated and graded in the time
  left, and keep it clearly optional.
- **Describe the method, not just our command.** The user may not have this skill's scripts at
  hand, so say what to do (e.g. join the notebook list to the leaderboard by author) and offer
  to run `scripts/scout.py` yourself when Kaggle credentials are available.

## Day one

A day or two of work that decides most of the competition. Skip what the user already has.

1. **Get the metric's source and run it verbatim.** The host's metric code, not the prose
   description. Read it for clamps, weights, normalisers, per-group averaging and edge cases,
   and test it on hand-built cases with known answers. Then find which term has the most
   headroom *relative to where the public notebooks sit*: a term weighted 0.1× can hold most of
   the available gain when every public pipeline scores badly on it.
2. **Find what the test is new along, and make it the CV group.** Patients, sites, wells,
   embryos, spatial blocks, time. Use the data page, host posts, ID overlap, and adversarial
   validation (a classifier that tries to tell train from test) when test features are visible.
   Look for quirks that break naive folds: duplicate or overlapping windows and crops, per-group
   annotation offsets, train/test overlap that a public notebook exploits.
3. **Scout public notebooks by author, and submit a floor today.** Join the notebook list to
   the leaderboard on the author's username (this skill's `scripts/scout.py board <competition>`
   does it). Titles and votes lead to the notebook everyone already forked; the join shows who
   is above you publishing working code. Fork the best board-verified runnable one and submit
   it: that is your floor, reached in a day — the floor, not the project. Before building on it,
   check how it validates (random splits over windows or rows of the same group inflate CV), and
   what its public score is made of (leakage or overlap that exists only in the public subset
   won't survive to private).
4. **Read the rules that bite late.** Weekly accelerator quota and whether graded reruns draw on
   it (when they do, a spent quota blocks every submission); the rerun's runtime limit and the
   hidden test's size; external data and pretrained-model rules; the team-merge deadline.
5. **Start your own model line this week** — in parallel with the fork, not after it. Even a
   modest model trained on your grouped folds gives out-of-fold predictions, the instrument
   everything else depends on, and it is the only line that can leave the plateau. In both our
   competitions the own-model line was scheduled for "later", and we finished on a fork.

## The instrument: grouped out-of-fold CV

- **No out-of-fold predictions, no instrument.** A model trained on all the training data makes
  every offline score on that data in-sample, however carefully the validator is built. Before
  trusting an offline number, ask whether the model that produced it saw those rows.
- **Score the shippable path.** CV runs the exact inference and post-processing you will
  submit, on held-out groups.
- **Validate the validator.** CV becomes an instrument once it has predicted the *direction* of
  a few deliberately large leaderboard moves; small moves are lost in public noise.
- **Know the noise floor.** Resample your OOF at the public subset's size (whole groups, if the
  split is by group) and look at the spread of score differences; rerun an identical notebook to
  see seed noise. A difference inside that spread is not evidence.
- **Fit thresholds and blend weights on OOF, nested where you can** (fit on some folds, score
  on the rest). Fold-averaged test predictions are less extreme than single-model OOF, so check
  that a threshold chosen on OOF still sits where you think on the test distribution.
- **Every number names its ruler** — in-sample, OOF (grouped by what), public, or private — in
  notes, logs, and messages to the user.

## Where the score is

- **Error analysis on OOF.** Break the score down by metric term, group, case type and pipeline
  stage. Headroom is usually concentrated in one or two places.
- **A diagnosed defect in the shared baseline is the opportunity.** When you prove the public
  pipeline structurally cannot do something the metric rewards, every fork shares the hole, and
  filling it is how you leave the crowd. So "X is impossible in this pipeline" is the start of a
  plan to replace that stage, never a note that the axis is closed.
- **Reframe before you tune.** Is it really an alignment or sequence-decoding problem, a
  detection-plus-assignment problem, a ranking problem, something to decode jointly under
  physical constraints? Would optimising a surrogate of the metric directly beat optimising a
  proxy loss?
- **Do the domain homework.** Host papers and code, the field's standard preprocessing and known
  artifacts, writeups from past competitions on similar data — then ask which of it the public
  notebooks ignore.

## Tripwires

For when you are operating the competition yourself. Each is a specific way we lost weeks, and
each is cheap to check.

- **No own OOF by the end of week two.** If there are still no out-of-fold predictions from a
  model you trained, building them is the priority over everything else.
- **Plateau.** Dozens of teams on your exact score; three experiments in a row inside the noise
  floor; the forum calling the same knobs "all negative". Stop sweeping and go back to error
  analysis, the structural defect, or a reframing.
- **A notebook fails on a missing dataset, weights file or mount.** That is a search, not a
  verdict: run `scripts/scout.py find <name or /kaggle/input/... path>`, and look through the
  author's own datasets, before closing it.
- **You wrote "closed", "exhausted" or "impossible in this pipeline".** Ask whether the stage
  should be replaced instead.
- **You flagged a runtime or format risk.** Fix it before submitting; flagging is not acting.
- **A watcher can't reach the API, hits a rate limit, finds a busy slot or an empty quota.**
  Those are facts about the plumbing, not results about the model. Report them as such and stop
  rather than wait quietly.
- **Comparing runs.** Read each run's own final summary; figures assembled from logs of runs with
  different control flow (one did more passes) don't compare.
- **Every few days, re-scout.** People above you publish mid-competition.

## Endgame

- **Final week:** start nothing new — finish, integrate and validate what already works.
  **Last two or three days:** freeze; only ship and verify.
- **Quota:** keep at least a third of the final week's accelerator quota for graded reruns.
- **Runtime:** estimate the graded run as fixed costs + (hidden size ÷ visible size) × per-item
  cost and keep it well under the limit (~60–70%). No sweeps inside the submitted notebook.
- **Robustness:** the hidden test can contain what the sample doesn't. Wrap per-item work so one
  odd file falls back to a default prediction instead of failing the whole run.
- **Settings and format:** check the accelerator on the exact version you submit (switching a
  CUDA-only notebook to TPU removes the GPU), and match the sample submission exactly, dtypes
  included.
- **Final selection:**
  - **Slot 1: the best grouped-CV submission.** Among a team's own near-equal candidates, public
    and private ranks are often uncorrelated or inverted; grouped CV tracked private far better
    in every writeup we read that measured it. Audit that CV as you would anyone else's —
    blend weights or thresholds fit on the same OOF they're scored on, or a member trained on
    all the data, inflate it. That is a quick check, not a new experiment.
  - **Slot 2: a deliberate hedge** on the assumption you're least sure of — the public-best if
    it is close on CV, or a variant that holds up across groups. Avoid the plain public fork
    unless CV genuinely prefers it: it locks you into the crowd tied on its score.
  - **An empty slot** is filled with your best *public* submission at the close, so it equals
    picking public-best. It only helps when a run may still finish grading after you choose.

## Carry forward

Start each competition from the last one's kit: the metric harness and its tests, a grouped-CV /
OOF template, `scripts/scout.py`, submit/poll tooling, and backbones worth reusing (one Biohub
top-3 team reused a backbone from their previous competition).

## Evidence

The measured outcomes behind every rule — our two post-mortems and the winners' writeups, with
source ids — are in `references/evidence.md`. Read it when a rule seems not to fit the situation
in front of you and you need to weigh it.
