# Clarification on Using a Self-Trained Model in Notebook Submission

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724502
- **Topic id**: 724502
- **Author**: Giridharan R (CONTRIBUTOR)
- **Posted**: 2026-07-11T14:24:42.701801400Z
- **Votes**: 0
- **Comments**: 1

---

## Opening post

I would like to clarify the notebook submission requirements.

I trained a model entirely by myself using only the official BioHub Cell Tracking During Development competition training data. I would like to use this trained model for inference in my submission notebook instead of retraining it every time.

Is it permitted to:

Upload the trained model as a private Kaggle Dataset (or other supported notebook input), and
Load that model in the submission notebook for inference?

The model was trained exclusively on the competition's official training data, and no external data or pretrained weights were used.

Would this be considered a valid notebook submission under the competition rules?

Thank you!

---

## Comments (1)


### Jordão Bragantini (CONTRIBUTOR) — 2026-07-13T14:54:03.473Z

That's permitted, but you should be able to reproduce it in case you win, see https://www.kaggle.com/WinningModelDocumentationGuidelines
