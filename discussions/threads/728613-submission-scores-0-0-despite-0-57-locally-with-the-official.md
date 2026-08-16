# Submission scores 0.0 despite ~0.57 locally with the official evaluate()

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/728613
- **Topic id**: 728613
- **Author**: Diana Daher (CONTRIBUTOR)
- **Posted**: 2026-07-23T23:36:07.208422200Z
- **Votes**: 3
- **Comments**: 2

---

## Opening post

Hi all,

Two separate submissions scored exactly 0.0, despite validating locally with the official scorer and getting a solid result. Posting in case others hit the same thing.

Local validation (tracking_cellmot.metrics.evaluate on train/ with real ground truth):
Cellpose + Hungarian tracking pipeline, scored directly against ds.tracks: Edge TP/FP/FN = 39/18/11, Jaccard = 0.5735
Full CSV round-trip test (write submission.csv, reload, rebuild graph, re-score): identical result, 0.5735. So the CSV format isn't losing anything as far as I can tell.
Every dataset in test/ appears at least once (explicit guard added). Format matches sample_submission.csv exactly.

Actual leaderboard:
Submission 1 (full-res, no time budget): completed, scored 0.0.
Submission 2 (downsampling x2 + dynamic time-budget/stride + fallbacks for empty/errored files): completed in ~4h, scored 0.0.
Both submitted around the metric patch/rescore. Still 0.0 on submission 2 after "Rescore Complete."

I can't see the hidden-test scoring run's logs, so I can't tell if this is something that only breaks on the hidden set's actual structure, or something in the scoring pipeline itself.

Has anyone else seen a submission score exactly 0.0 despite reasonable local validation? Any pointers (hidden-set naming, node_id scoping, edge t=-1 convention, anything Kaggle-harness-specific vs. the published tracking_cellmot code) would help a lot.

Thanks!

---

## Comments (2)


### Cortex Evolved (CONTRIBUTOR) — 2026-07-31T11:48:23.853Z

from what I gathered from the rules is that they hide the actual score. Confusing! I had a perfect 1 local, did some things for submission, the bot changed my code to some generic AI slop and I got a .580. I just don't know.

#### ↳ Diana Daher (CONTRIBUTOR) — 2026-07-31T23:41:17.217Z

> Following up on my own thread (submission scoring 0.0 despite ~0.57 locally) — found the root cause, sharing in case it helps others who hit an exact 0.0 rather than a partial drop.
> 
> Root cause: edges linking non-consecutive frames.
> 
> My pipeline used a time-budget strategy that subsampled frames across the whole video (process 1 frame every N, spread across the full duration) to stay under the 12h limit on a much larger hidden test set than the visible test/ sample. That produced edges connecting frames like t=0 -> t=5, t=5 -> t=10, etc.
> 
> Ground truth edges only ever connect strictly consecutive timepoints (t -> t+1). An edge between non-consecutive frames can never match a GT edge under the official metric, so every single edge I submitted was structurally unmatchable — TP=0 across the board, regardless of detection quality. No format error was raised (the file was valid CSV, all columns correct, all datasets present), so it silently scored 0.0 with no diagnostic signal.
> 
> Fix: instead of subsampling with a stride, process a *contiguous block* of frames from t=0 for each dataset (sized proportionally to the global time budget). This guarantees every edge connects t and t+1. Went from 0.0 -> 0.366 -> 0.380 after this fix plus some tracking tuning (volume-weighted cost matrix, min_size filter).
> 
> If your submission scores exactly 0.0 (not just lower than expected) and your local validation looks fine, I'd check whether any part of your pipeline could be producing edges across frame gaps — sparse detections, frame skipping for speed, or gap-closing/rescue logic that links across more than one timepoint without it being reflected correctly in the graph structure.
> 
> Hope this saves someone else a few submission attempts.
