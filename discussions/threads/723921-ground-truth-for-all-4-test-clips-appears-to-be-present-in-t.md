# Ground truth for all 4 test clips appears to be present in the train split

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/723921
- **Topic id**: 723921
- **Author**: LeeWhieldon (CONTRIBUTOR)
- **Posted**: 2026-07-08T19:09:04.758917200Z
- **Votes**: -8
- **Comments**: 10

---

## Opening post

While building a local cross-validation setup, I found that all 4 official test clips presented in the dataset for the competition have byte-identical copies in the `train` split, complete with full ground truth annotations (`.geff`).

Verified directly against the competition zip (pixel comparison at t=0, t=50, t=99 for each clip, not just filename matching):

| Test clip id | Train copy | Pixel-identical | Train `.geff` present |
|---|---|---|---|
| `44b6_0113de3b` | `train/44b6_0113de3b.zarr` | Yes | Yes |
| `44b6_0b24845f` | `train/44b6_0b24845f.zarr` | Yes | Yes |
| `6bba_05b6850b` | `train/6bba_05b6850b.zarr` | Yes | Yes |
| `6bba_05db0fb1` | `train/6bba_05db0fb1.zarr` | Yes | Yes |


Reproduction: read `train/{clip_id}.zarr/0/c/{t}/0/0/0` and `test/{clip_id}.zarr/0/c/{t}/0/0/0` from the competition zip for the same clip id and timepoint, decompress with blosc2, and compare arrays: they match exactly.

This means the ground truth for the hidden test set is currently downloadable by anyone via the train split. Flagging this before it's discovered and used to game the leaderboard. Happy to provide the verification script if useful.

---

## Comments (10)


### Adarsh (MASTER) — 2026-07-09T12:40:38.307Z — 1 votes

Those are just a placeholder

#### ↳ LeeWhieldon (CONTRIBUTOR) — 2026-07-09T13:14:39.053Z

> Placeholder for what? Can you please clarify?

#### ↳ ↳ Thibgolds (CONTRIBUTOR) — 2026-07-10T20:17:58.243Z

> > Its a placeholder to see if your notebook actually produces a csv file, it's just to help participants debug and has nothing to do with the actual scoring or leaderboard. Once you submit the notebook for evaluation, a different test set is used (that you don't have access to).

#### ↳ ↳ LeeWhieldon (CONTRIBUTOR) — 2026-07-12T12:02:37.673Z

> > But I think it could impact the public leaderboard. Could you clarify how it won’t cause a potential gaming of the commit/public leaderboard?

#### ↳ ↳ fnands (MASTER) — 2026-07-12T15:07:04.890Z

> > Those four events won't be used to calculate your leaderboard score. They are only there to test if your solution works mechanically. 
> > 
> > Once you do an actual submission, they will be replaced with the actual test dataset. 
> > So think of them as placeholders. 
> > 
> > Kaggle has two ways of running your notebooks: Normal mode, and submission mode (there is even a environmental variable you can check to see what mode you are in). 
> > 
> > Once you run in submission mode, the true test set it used. That's why internet access is disabled for submission mode: so you can't exfiltrate the real data.

### Timmy Juicehouse (EXPERT) — 2026-07-09T08:06:44.387Z — 2 votes

The real test set is not visible. The four you see are actually dummies, used to help you debug whether the submission can be generated and pass the test by producing submission.csv.

#### ↳ LeeWhieldon (CONTRIBUTOR) — 2026-07-09T12:53:23.820Z

> Hi @sweetyheehee, thanks for flagging this & really useful if true.
> 
> I independently found something that's consistent with what you're describing: the 4 visible test clips (44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1) are byte-identical to train clips of the same id, including full ground truth annotations (see original message). If the real hidden test set gets substituted at grading time, that would explain it tidily: the visible ones are just there so we can validate our submission pipeline runs and produces a well-formed submission.csv, not for actual scoring. Is that safe to say?
> 
> Also, do you know if this testing assumption has been confirmed by the host anywhere, or is it something you inferred from behavior (e.g. discrepancies between public LB scores and what a notebook produces locally)? Would be great to get it nailed down explicitly since it changes how everyone should be interpreting local validation against the visible test folder.

#### ↳ ↳ Timmy Juicehouse (EXPERT) — 2026-07-09T12:58:42.817Z

> > All Kaggle competitions work like this, so don't overthink it.

#### ↳ ↳ LeeWhieldon (CONTRIBUTOR) — 2026-07-09T13:08:26.987Z

> > I've participated in other Kaggle competitions where train & test datasets are mutually exclusive, hence why I am bubbling up my questions. It is typically recommended to perform testing against a dataset that's not part of the training set (as you know, could cause contamination of the model). 
> > 
> > That said, I'll follow your advice and submit rather than attempting to perform heavy CV testing on my end. 
> > 
> > Thanks!

### unknown — 2026-07-08T20:52:32.737Z

*(empty)*
