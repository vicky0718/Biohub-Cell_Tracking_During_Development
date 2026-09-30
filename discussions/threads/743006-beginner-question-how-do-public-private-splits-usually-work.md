# Beginner question: How do Public/Private splits usually work in biological datasets?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/743006
- **Topic id**: 743006
- **Author**: Rishavendra Sharma (CONTRIBUTOR)
- **Posted**: 2026-09-24T15:54:53.727976800Z
- **Votes**: 3
- **Comments**: 5

---

## Opening post

Hi everyone,

I'm relatively new to this type of Kaggle competition, and I have a general question about how the Public/Private Leaderboard split usually works in these tracking/computer vision challenges.

I noticed on the leaderboard page that 71% of the test data is reserved for the private LB. For those of you with a lot of Kaggle experience (or the hosts!), how is this usually handled in biological datasets?

Does a 71% private split typically mean it is drawn from the exact same distribution as the public 29% (e.g., just different timeframes or random spatial crops of the same embryos)? Or is it standard practice in these types of competitions for the private set to evaluate true generalization—meaning we should expect completely unseen imaging conditions, different microscopes, or totally different cell types?

I'm asking because I'm trying to figure out my local validation strategy. I'm not sure if I should be heavily tuning my pipeline to perfectly fit the visual characteristics of the public test movies, or if I should be focusing entirely on heavy augmentations and generalized robustness assuming the private 71% will look completely different.

Any insights into how Kaggle or biological tracking challenges generally structure this would be super helpful for a beginner! Thanks!

---

## Comments (5)


### g john rao (MASTER) — 2026-09-28T12:27:30.183Z — 1 votes

there are two embryos in the hidden test set. the first one is private and the second is public (but it could be a mix split of the two embryos as well)

### Georgy Mamarin (MASTER) — 2026-09-28T05:16:49.117Z — 1 votes

@rishavendrasharma yes, the prefix is the embryo. The [Data tab](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data) spells it out under "Embryo Identity": "The first segment identifies which embryo the sample comes from." The train set has only two prefixes, 44b6 (71 movies) and 6bba (128), which matches the [host's reply](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793#3485884) that there are two training embryos. So splitting by prefix is an embryo-level holdout for whatever you train yourself (as you said, the public checkpoints have seen both).

### Georgy Mamarin (MASTER) — 2026-09-26T08:31:58.167Z — 1 votes

Only the hosts know how the 29/71 cut was made, but they have said what the hidden test set is. It shares no embryos with train and is roughly the size of train ([host reply](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/716793#3485884)). All the data followed one protocol, the same instrument and developmental stage, with each embryo imaged in its own session ([another host reply](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724386#3496285)). So expect new embryos under the same imaging conditions rather than new microscopes or cell types.

The 71% on its own tells you little. On about a thousand past competitions, when a submission moved a team's best public place up, it also placed above the submission it replaced on the private board roughly 8 times in 10, whether the public part was under 10% of the test or over half. In code competitions like this one it was closer to 3 in 4.

How much a board moves differs far more between competitions. On the thirteen cell, tissue and cryo-ET competitions I tabulated in [the shake-up thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352#3525358), between 97% and 36% of the teams in public medal positions still had a medal after the reveal (9% on one more, where the bronze line fell inside a tie). The size of the public part didn't predict which: SenNet showed 67% of its test and kept 36% of its medal zone, CZII showed 26% and kept 96%.

So I wouldn't tune to the public movies. Holding out a whole embryo is the local setup that matches a test of new embryos, and it is what the [CV thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160) converged on. @ramarlina found the public board almost 10% above that hold-out, with single movies scoring anywhere from 0.46 to 0.98. One trap from the same thread: the public checkpoints were trained on all 199 annotated videos, so scoring them on train videos scores data they have already seen.

What are you validating on right now: a held-out embryo, held-out movies, or time windows inside one movie?

#### ↳ Rishavendra Sharma (CONTRIBUTOR) — 2026-09-26T10:41:58.203Z

> Thanks, this is very helpful—especially the distinction between new embryos and completely different imaging conditions.
> Right now, my comparisons use whole training movies, not a verified embryo-level holdout. I’m using public pretrained checkpoints, so those results are diagnostics on previously seen data rather than clean CV. For association-head fine-tuning, I’ve separated training, checkpoint-selection and confirmation movies, but that still doesn’t remove the original checkpoint’s training exposure or establish embryo independence.
> Do you know whether there is an authoritative mapping from movie IDs to source embryos? In particular, do prefixes such as 44b6 and 6bba identify embryos or something else? I’d like to avoid calling a movie-level split an embryo-level holdout without confirming that.
> Your explanation also clears up my main concern: the 71% private fraction alone doesn’t imply different microscopes or cell types. Generalization to new embryos under the same acquisition protocol is the more relevant target.

### Gabriel (CONTRIBUTOR) — 2026-09-25T21:55:15.607Z

I am also curious about the answer to this question.
