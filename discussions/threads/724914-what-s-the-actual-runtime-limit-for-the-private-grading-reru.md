# What's the actual runtime limit for the private grading rerun?

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/724914
- **Topic id**: 724914
- **Author**: LeeWhieldon (CONTRIBUTOR)
- **Posted**: 2026-07-13T15:26:49.618372800Z
- **Votes**: -1
- **Comments**: 0

---

## Opening post

Our submission notebook completes successfully in ~5.6h against the visible test clips, but the actual scored rerun came back as Notebook Timeout with no public score. The debugging docs say to check the Code Requirements page (12h GPU), and that the hidden dataset "can be larger/smaller/different than the public dataset", which would explain a timeout if the real hidden set is bigger or denser than the 4 visible clips.

Wanted to confirm directly rather than guess: is the rerun genuinely held to the same 12h GPU limit as the Code Requirements page states, or is there a separate, shorter limit specific to the scoring rerun (I've seen other competitions mention something like 9h for that step specifically)? Trying to budget compute time correctly rather than keep guessing blind. 

Thanks!

---

## Comments (0)

*(none)*
