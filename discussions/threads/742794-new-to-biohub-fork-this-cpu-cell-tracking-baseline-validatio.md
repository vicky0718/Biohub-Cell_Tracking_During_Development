# New to Biohub? Fork this CPU cell-tracking baseline + validation toolkit

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/742794
- **Topic id**: 742794
- **Author**: Yang Kuang Ou (CONTRIBUTOR)
- **Posted**: 2026-09-23T13:53:04.969112Z
- **Votes**: -2
- **Comments**: 0

---

## Opening post

Want a clean place to start building a Biohub cell tracker? This public Notebook is a forkable CPU starter kit: a small baseline plus checks and visualizations to help you understand what your pipeline is doing.

**What you can try**
- CPU baseline: intensity thresholding → connected components and centroids → greedy nearest-neighbor links.
- Embryo-grouped validation helpers for leakage-aware comparisons.
- `.geff` graph inspection, physical-coordinate visualizations, and sparse-label-aware edge diagnostics.

**Get started:** open the [Notebook](https://www.kaggle.com/code/yangkuangou/biohub-public-simple-baseline-validation), fork it privately, attach the competition input, and enable `RUN_BASELINE_ON_TEST` to create a candidate CSV. It does not submit anything.

It is intentionally a simple learning baseline, with no trained detector or division handling. The public run reports no validation or leaderboard score, so treat it as a starting point to extend, not a performance claim. No private model, weights, or post-processing are included. Licensed under Apache 2.0.

What would you add first: motion cues, division handling, or a stronger detector? Feel free to fork and build on it.

---

## Comments (0)

*(none)*
