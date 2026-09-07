# What the top 15 actually say — and the one lead in it we can use

`notes/58` mined one author chosen by reputation (hengck23). This mines by **rank**:
`discussions/extract_by_leaderboard.py` joins `TeamMemberUserNames` from the leaderboard CSV
against every forum author, so the filter is "has solved this problem", not "is famous".

Top 15 teams = 23 usernames. **29 posts**, from 7 of them — including rank 1 (0.970) and
rank 2 (0.966).

> The extractor found zero on its first run and that looked like a finding. It was a bug:
> comment authors carry no `userName` field, only a profile path in `author.url`
> (`/madarshbb`). Fixed before reading anything into the silence.

---

## 1. Rank 5 says our entire current approach is capped

Tang (`hirotetsu`, **0.961**), three separate posts:

> *"I'd recommend retraining the model instead of just using the public ckpt… **the current
> ckpt has kind of hit a wall, it's hard to get more gain from post-processing alone.**"*
>
> *"there's a 'correct' order to work on them: **detection → linking → division**. detection
> should come first, once detection is solid, it's easier to improve others."*
>
> *"in my case, **most of the gains come from model improvements**."*

That is the most directly useful thing in the whole set, and it is bad news. Every arm this
project has run is post-processing on `pilkwang`'s public checkpoint. Our measured results
are exactly the shape he describes: `lb941` 0.941 → `sew20` 0.942, one thousandth, with
`union` at zero and `div15` at −0.003.

**And retraining is closed for us.** `notes/33` §5: ~333 h at 50 min/epoch against ~140 h of
remaining Kaggle quota, and `notes/53` priced the alternatives. So the honest statement is
that the ceiling of our approach is somewhere near where we are, and the top of the board is
reachable only through a door we cannot open.

## 2. Rank 6 says the opposite, and holds 0.957 doing it

TWEAK (`tweakai`, **0.957**):

> *"we are working on a **universal plugin** that the bio cell team can plug into their
> current pipeline with minimal changes. We have tested our plugin with every available
> unique public notebook and model, with **gains ranging from 0.030, 0.040, to 0.050**
> instantly just attaching our plugin. We've seen gains from a single public model reach a
> score of 0.940 untuned."*

A pure post-processing layer, applied to public detections, reaching 0.957. So
post-processing is *not* capped at 0.942 — ours is. The difference is that theirs is an
algorithm and ours is a parameter sweep. Sergio (rank 1) asked him the obvious question —
*"do you mean something that optimizes/refines the tracking graph, or do you take the public
notebook detections and apply your own tracking method on top?"* — and got no answer.

## 3. Rank 2 gives a diagnostic we already own

Soheil Ayati (`soheilayati`, **0.966**):

> *"Break your missed edges into two categories: **missing endpoint nodes** and **incorrect
> associations**. In my case, many 'linking' issues actually originated earlier during node
> selection, so improving the edge model won't necessarily help."*
>
> *"Always validate complete movies using the official scorer and **movie-level OOF
> splits**. Edge-level random CV can be highly misleading."*

The first half is `pipeline/anatomy.py::edge_anatomy` — our `fn_detect` / `fn_mislink` /
`fn_gap` buckets, which `notes/51` already used to find the anatomy had inverted. We have the
tool and have not pointed it at the fork's output.

The second half is a **partial** correction to `notes/66`. Local scoring failed there because
it used the four *placeholder* clips with 52-1,229 sparse annotations. Complete training
movies with movie-level splits is a different and better design. But it does not fully
transfer: `notes/67` §3 read `split_manifest.json` and the public checkpoint trained on
**all 199** movies, so any CV we build on them scores a model on its own training data.
Soheil and Sergio retrain, so their CV is clean; ours cannot be. Worth stating rather than
retrying and being surprised.

Sergio (`sersasj`, **0.970**) confirms he has one that works: *"Every node/edges removal
I've done reflected public lb/cv score."*

## 4. Rank 8 hands over a concrete data property nobody exploits

tom99763 (`tom99763`, **0.954**), thread *"beware of jumps in ground truth track"*:

> *"Videos sharing the schedule | Frames duplicated after index…
> `6bba_05b6850b · 07477033 · 5b28472a` : 4, 12, 27, 42, 52, 57, 59, 62, 66, 76"*

**`6bba_05b6850b` is one of our four verification clips.** If frames are literally duplicated,
the true displacement across those `t → t+1` pairs is exactly zero — which is the same
phenomenon `notes/59` measured from the other end: **8.4% of GT links have exactly zero
displacement** (10,772 of 128,883). Two independent observations of one thing.

An identity link across a duplicated frame pair is correct *by construction*, and nothing in
the public notebook lineage mentions duplicate frames at all. `claude_dupframes` is checking
the pixels rather than the claim: whether the duplicates exist, whether tom99763's indices
match, and how common it is across the training set.

## 5. Closed by someone who spent the time

Sergio (rank 1) on synthetic data:

> *"I've tried the synthetic volume idea for **~2 weeks with no clear improvement** in my
> scores, so I dropped it."*

Two weeks of the rank-1 competitor's time says do not start. He also priced the runtime
early: *"~200 second per volume should fit in 12h"*, consistent with `notes/69` §3.

## 6. A metric claim, raised and withdrawn

antonoof (rank 6) claimed deleting edges did not move the score — *"deleted them more than
1/2, the metric did not change, deleted 2/3, the metric did not change"* — then withdrew it:
*"maybe I was mistaken… I think it's just me."* Sergio contradicted him directly from his own
CV. Recorded so it does not get picked up later as an unexplored exploit; `notes/02` already
measured the edge terms against the official scorer.

---

## What changes because of this

1. **`claude_dupframes` is the new lead** — a data property, from a top-10 competitor,
   corroborated by our own `notes/59`, unexploited by any public notebook. Running.
2. **Point `edge_anatomy` at the fork's output** (Soheil's split). We own the tool and have
   never used it on this pipeline.
3. **Do not start synthetic data.** Rank 1 spent two weeks and dropped it.
4. **Do not expect the parameter sweep to reach the top 100 alone.** Rank 5 says the
   checkpoint has hit a wall and our own numbers agree. The remaining knobs are worth
   thousandths; the gap to rank 100 is one thousandth, so they are still worth finishing —
   but the band above needs the thing TWEAK has and will not describe.
