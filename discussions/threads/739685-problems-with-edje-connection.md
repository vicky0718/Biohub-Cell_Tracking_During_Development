# Problems with edje connection

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739685
- **Topic id**: 739685
- **Author**: Mark (MASTER)
- **Posted**: 2026-09-05T13:05:03.739057700Z
- **Votes**: 3
- **Comments**: 7

---

## Opening post

Any implementation of a "smarter" motion linker—anything beyond a basic algorithm—results in an increase of exactly 0.001.

The CV may vary by about 0.01–0.03, but the model's performance simply neither worsens nor improves on the LB, which seems strange. 

Has anyone encountered this?

---

## Comments (7)


### Rishabh Roy (EXPERT) — 2026-09-07T09:08:18.460Z — -1 votes

i believe that for me since i am using a point based detector it losing out on lot of features . If i would have used a cell segment based approach it would have al lot of features that we could ave used and identify the tracked cell

### Yubo WANG (EXPERT) — 2026-09-08T04:03:10.897Z

I’m seeing pretty much the same behavior.

My motion linker changes give around +0.012 on 5-fold CV on average, with no fold getting worse, but the LB gain is exactly +0.001. I also compared the public 0.946 LB notebook with my older private 0.942 LB notebook under the same 5-fold CV setup, and surprisingly the public one was about 0.015 worse locally. Given that some people have already pointed out possible test-set overfitting in public notebooks, I’m starting to suspect there may also be a fairly large distribution shift between train, public test, and hidden test...?

https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/739278

#### ↳ hengck23 (GRANDMASTER) — 2026-09-08T12:03:12.330Z — 1 votes

> ask chatgpt "distribution shift between train, public test, and hidden test…" on how to probe or verify.
> You definitely can do some online testing/training and send the results back by "smart encoding your submission"
> 
> also download more external data or augmentation etc to test locally.
> when you evaluate your model on local dataset, don't just look at the numbers. visualise the results and more importantly, explain why (especially statistically). is the error common or will be common?

#### ↳ ↳ Yubo WANG (EXPERT) — 2026-09-08T12:08:33.423Z

> > I'm working on finding the cause you mentioned, which was also a problem I encountered when I hit a bottleneck with my ROGII. Thanks a lot🥲

#### ↳ ↳ hengck23 (GRANDMASTER) — 2026-09-08T12:50:02.010Z — 1 votes

> > Root cause not necessarily means why it happens. It can also means will it reliably happen. Eg you can make some smart error correction to repair local track. But will this error actually happens in hidden test ( not why it happens).
> > 
> > 
> > As a concept example, you find your gain in local lb is due to repair of gap = 5 missing frames. Then you find that of all broken tracks, 80% is gap1, 18% is gap2 … less than one percent is gap5

#### ↳ ↳ Yubo WANG (EXPERT) — 2026-09-08T14:49:59.107Z

> > Truly brilliant probe design! Thank you for guiding a Kaggle newcomer like me.

### Rishabh Roy (EXPERT) — 2026-09-05T14:55:36.760Z

on the same boat
