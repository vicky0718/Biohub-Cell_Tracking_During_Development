# What the winners did, set against what we did

2026-09-30, the day after the close. Nine placement writeups went up (3rd, 5th, 12th, 14th,
18th, 89th, 213th, 288th, 303rd) plus several targeted posts. Read against our own
`notes/75`–`notes/94`, they overturn two of this project's conclusions and confirm the rest.
Threads are saved under `discussions/threads/<id>-*.md`.

## 1. Where the score was

Biohub 3rd place (yu4u, GRANDMASTER; `744484`) published an ablation over their own OOF CV:

```
A0  detection + Hungarian linking             0.890   div_J 0.000
A1  + dense flow                              0.902
A2  + learned matcher                         0.902
A3  + division models, calibration,           0.964   div_J 0.535    <- +0.063
      joint link/division optimisation
A4-A7 gap closing, short-component removal,   0.978
      affine coordinate refinement
private                                       0.967   div_J 0.47
```

**Most of their lead over detection-plus-linking came from the division term** — the term
worth 0.1× that the public pipeline scored at ~0.2. Ours was 0.23–0.31 on every arm we ran.

## 2. Two of our conclusions were wrong

### "Divisions are closed" (`notes/78`, `notes/79`)

We established, correctly, that the public ILP cannot form a division at the weight it
ships, and that the later motion relink is one-to-one and discards the ILP's edges. Four
separate writeups (12th, 14th, 288th, and roger in `notes/89`) found the same fact
independently. Where we differed was the next step. We filed divisions as a closed axis.

- **18th place (`744531`) started from the same public notebook we did** — "Biohub Harmonic
  Fusion" — kept its detection and candidate graph, and rebuilt everything after: which
  edges to keep and where divisions go. The inherited pipeline recovered 23 of 151 training
  divisions; that was their stated reason for focusing there.
- 3rd place built parent / pre / post division models and a joint LP over links and
  divisions.
- 5th place gave the linker transformer a dedicated division head and made the ILP's split
  cost depend on it.

A diagnosed structural defect shared by every fork is the most valuable finding available
in a crowded competition. We had it in `notes/79` on 09-13 with sixteen days left.

### "The board is the only instrument" (`notes/81`, `notes/89`, `MEMORY.md` §5)

We concluded that because our offline validator never predicted the board, the public
leaderboard was the only trustworthy instrument. The writeups say both were bad:

- The test is two **new embryos**: public = `fdad` (60 videos), private = `ea36` (106 videos),
  with substantially different cell density (3rd place, from probing). Public is one embryo.
- korokke3 (`744548`) ran five configurations of our exact `x138`/`rd07` lineage. **The public
  board ranked them in almost exact reverse of private.** Their worst public (0.948) was
  their best private (0.924). Leave-one-embryo-out CV had picked it; they overrode CV with
  public tuning and missed bronze by 0.001.

The right instrument was **leave-one-embryo-out (or embryo-grouped) CV on out-of-fold
models**. We could not build it, because we never trained models — the public checkpoint
had seen all 199 videos (`notes/87` §3). That is the root cause under both failures: without
our own OOF predictions, every offline number was in-sample, and the only remaining ruler was
a single-embryo public board.

## 3. What they confirm

- **Node deletion doesn't transfer.** 18th: "Never delete nodes, and decide on all 199
  videos." Same conclusion as `notes/90`, which zhincez paid four submissions for.
- **Fork-and-tune has a ceiling.** `744498`: "a fork-and-tune line … had a ceiling: ~0.954
  public, 0.924 private, 161/4020." 288th: forking `x138` was their biggest gain, then
  probing stalled at private 0.921. Our `rd07` (the same lineage) landed 0.953 / 0.917.
- **The V1284 head was never private.** `744498`: it was reproducible from `x138`'s own
  capture mode, and a self-trained head scored 0.954. We closed `x138` on that mount on
  09-22 (`notes/94` §4).
- **Integer coordinates.** `744093`: writing float centroids cost a team 0.008 public. Our
  submissions inherited `int(round(...))` from the public notebooks and were unaffected.
- **Coordinate offsets are per-embryo.** 12th: 6bba GT sits +0.675 plane above nucleus
  centres, 44b6 +0.115. korokke3: mean z-offset +0.17 µm vs +0.84 µm. 3rd place's affine
  coordinate refinement was worth +0.006. None of our arms looked at localisation.

## 4. The from-scratch counterfactual

`744485`: one competitor, a 2.5D ConvNeXt detector, an ILP tracker and a small division
model, trained on one embryo plus half the other and judged on held-out videos from both.
**Private 0.939 — about 22nd.** Synthetic pretraining and ZebraHub pretraining gave them
nothing, which matches our `notes/87` §5 correction. They didn't select it and ranked lower,
which is rule 7 of the playbook below in miniature.

We had eight weeks, two GPU sessions and 30 GPU-hours a week. That model was within reach.

## 5. Where this went

The forward-looking version is a Claude Code skill, `kaggle-competition-playbook`. It lives in
this repo at `.claude/skills/kaggle-competition-playbook/`. To use it in any repo, copy the
folder into that repo's `.claude/skills/` or into `~/.claude/skills/`, or save the packaged
`.skill` to your Claude profile.

- `SKILL.md`: day one, the grouped-OOF instrument, where the score is, tripwires for an agent
  running a competition, and the endgame.
- `references/evidence.md`: the measured outcome behind each rule. The sources are this repo,
  our Rogii notes (`rogii/chat/memory/`) and both competitions' winners' writeups.
- `scripts/scout.py`: a standalone version of `tools/scout_notebooks.py`.
  - `board` joins notebooks to the leaderboard by author.
  - `find` searches datasets, models and notebooks for a missing mount.
  - `find v1284` returns `anvithpothula/biohub-v1284-head-s075` as its first result. That is
    the head we closed `x138` over, and it had been public since 09-20.

**Rogii told the same story.** Our best Rogii submission was also a full fork: 7.18 public,
propped up by seed noise and a leakage exploit. Its leakage-free core measured 10.38 on our
honest harness, while the public leaders were near 5.3. Our own CNN line had been scheduled
for "phase 3", and the later notebooks are still built on the fork.

Rogii's winners made the same points:
- At least four top-25 writeups found public and private anti-correlated across their own
  candidates, while grouped CV tracked private. In the 22nd-place writeup, public/private
  Spearman was −0.48 and CV/private +0.84.
- At least six top-40 writeups describe coding agents doing most of the implementation.

**Testing the skill.** The skill was checked blind: answers to four realistic prompts written
with the skill, and without it, scored against 34 assertions by a grader that didn't know which
was which.

- The first draft passed its own checklist, but it retold our history, naming Biohub/Rogii up to
  19 times per answer, and crowded out domain content.
  - Its answers to the EEG and solar prompts had almost no EEG or solar content.
  - Regraded, it passed 19/27 and ranked last every time.
- The revision leads with the user's domain.
  - It passed 34/34 against 30/34 without the skill.
  - It ranked first on three of four prompts for overall usefulness.

The skill adds, over no skill: scouting by author, forking a floor, looking at what
higher-ranked teams publish, and treating a missing weights file as a search.

## 6. The top 25, complete

By 2026-10-06, 17 of the private top 25 had published writeups. They are archived with their
comment threads and images in `discussions/writeups/`, and the README there is a cross-team
digest. `discussions/scrape_writeups.py` regenerates the archive: it reads each team's
`solutionWriteUpUrl` from the private leaderboard, then `WriteUpsService/GetWriteUpBySlug`.

Three of those writeups are about our own lineage:

- **9th** took `x138` and added two things: six detectors and a linker of their own, worth
  +0.023 private, and DIVCARRY, which re-adds the daughters the public ILP never produces
  (+0.013). That came to **0.949 private, gold**. Their own models looked worth only +0.004 on
  public.
- **23rd** kept `x138` frozen and added a division-recovery stack, taking it from 0.917 to
  0.935 private. A z+1 hedge in their second slot made it **0.939, silver**.
- **12th** (a Claude Code agent, the 744501 thread above) built its own stages on the same
  public models, reaching **0.946, gold**. They name in-sample validation as their largest
  error.

Same base, same missing division stage: they rebuilt it, and we filed it as closed
(`notes/78`–`79`). The difference was 0.02–0.03 private, about 1,000 places.
