# Submission Stuck

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/726526
- **Topic id**: 726526
- **Author**: Samson Enoch (CONTRIBUTOR)
- **Posted**: 2026-07-15T16:47:18.295306400Z
- **Votes**: -2
- **Comments**: 4

---

## Opening post

is it just me or the submission evaluation is going on for a long time, mine personally has been scoring for the past 5 hour ! ANY CLUE ON WHATS HAPPENING ????

---

## Comments (4)


### FasterYouChase FasterIRun (CONTRIBUTOR) — 2026-07-16T07:04:59.320Z — 2 votes

Because this is a standard Kaggle code competition with a 12-hour rerun limit, the visible verification dataset is intentionally kept minimal to allow rapid notebook saving and verification in 15–30 minutes. However, according to the official Kaggle Dataset Overview, the hidden test set swapped in during final evaluation is not a fraction of the training data. It is roughly **the exact same size as the ~88 GB training dataset.** Since the training set covers 199 total Zarr image volumes, our submission pipelines must be optimized to process roughly 200 hidden test volumes within the strict 12-hour notebook execution window.

### OzanM. (GRANDMASTER) — 2026-07-16T12:16:31.433Z

I'm curious about the secret of anyone whose submission time is shorter.
Mine sometimes takes 7 hours, sometimes 5 and a half hours.

### unknown — 2026-07-16T07:28:04.277Z — -1 votes

*(empty)*

### unknown — 2026-07-16T03:11:14.070Z — -1 votes

*(empty)*
