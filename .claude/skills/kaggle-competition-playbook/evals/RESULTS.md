# Eval results

Four prompts, in `evals.json`:
- a day-one plan (EEG);
- a plateau (solar);
- final selection;
- an operating check-in (reef segmentation, added in iteration 2).

Each prompt was answered by a fresh agent with the skill and by one without it. A separate
grader scored every answer blind against the assertions, then ranked the answers for overall
usefulness. There is one run per cell, so the differences are indicative, not statistics.

## Iteration 1: v1 draft

- **Scores:** with the skill 17/17 (100%), without it 14/17 (85%), on the original assertions.
- **What the skill added:** scouting notebooks by author, and forking the best one as a floor.
  Also looking at what higher-ranked teams published.
- **Grader's note:** the v1 answers were process-heavy and anecdote-heavy. They retold Biohub and
  Rogii up to 19 times per answer, and lost the domain content the no-skill answers had:
  - EEG montage harmonisation and line-noise filtering;
  - PV temperature correction, and telling soiling apart from degradation.

## Iteration 2: v2 rewrite

**What changed in the skill:**
- Domain-first framing, plus a "when advising the user" section: lead with their problem, our
  history is calibration and not content, label unknowns, keep narrow questions narrow.
- Evidence moved into `references/evidence.md`.
- A tripwires section.
- `scripts/scout.py`.

**What changed in the evals:** new assertions for domain substance and noise-floor measurement,
the merge deadline, and guards against history narration and unflagged assumptions. A fourth
prompt was added. The v1 answers were regraded blind against the same assertions.

| configuration | assertions passed | usefulness rank (day-one, plateau, final, check-in) |
|---|---|---|
| v2 | 34/34 (100%) | 1, 1, 2, 1 |
| no skill | 30/34 (88%) | 2, 2, 1, 2 |
| v1 | 19/27 (70%) | 3, 3, 3, – |

**Grader's notes on v2:**
- The plateau plan froze 7 of the 12 remaining days.
- The final-selection answer didn't audit whether the CV-best's own CV was inflated. The no-skill
  answer did, which is why it ranked first there.
- Answers assumed the user has `scripts/scout.py`.

## Iteration 3: v3 targeted fixes

- **Final week:** start nothing new. Freeze only in the last two or three days.
- **Slot 1:** audit the CV-best's own CV.
- **Describe the method, not just our command.**

The two affected prompts were rerun with v3.

Results, graded by the skill's author (not blind):

- **Plateau:** 10/10.
  - The plan now runs days 1–5 on new work, days 6–9 on integrating, and freezes only for
    days 10–12.
  - It describes the by-author join before offering to run the script.
  - It audits the slot-1 CV.
- **Final selection:** 7/7.
  - It now audits the CV-best's own CV, re-scoring blend weights nested on the existing OOF
    with no submissions. This was the point where the no-skill answer had ranked higher.
  - Two borderline notes:
    - The audit sits under "before you lock them in", and it could change which pick is the
      main bet.
    - One bolded sentence about private sites comes before its "if split by site" qualifier,
      though the answer does have a dedicated section on that assumption.

## Next time

- Many process assertions pass under every configuration. Sharpen them, or add operating
  scenarios where the no-skill agent tends to fail:
  - a validator that is in-sample by construction;
  - a notebook that dies on a missing mount;
  - a runtime risk that has been flagged but not fixed.
- Run 3 or more samples per cell to get variance.

## Post-evaluation edits (2026-10-06, not re-evaluated)

Three short items were added after reading the Biohub top-25 writeups
(`winning_writeups/biohub_top25/`):
- a tripwire to count the metric's rare event at every pipeline stage;
- a line on auditing how the writer rounds and casts values;
- a clause on labelling post-hoc edits by their effect on the metric.

The evidence file gained a section of figures from those writeups, and our final rank was
corrected after Kaggle's re-ranking. These are small additions in the evaluated style, and the
evals were not re-run for them.
