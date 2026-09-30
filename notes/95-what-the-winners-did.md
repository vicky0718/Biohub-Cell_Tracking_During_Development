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

The forward-looking version — day-one checklist, rules, cadence — is a Claude Code skill in
the `rogii` repo: `.claude/skills/kaggle-competition-playbook/SKILL.md`. It draws on these
writeups and on the Rogii ones, where the same pattern holds: every top-40 writeup centres
on validation, three were largely agent-written, and the public-3rd / private-20th author
kept gold only by selecting on GroupKFold CV over the public board.
