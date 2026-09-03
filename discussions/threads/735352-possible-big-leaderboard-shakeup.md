# Possible big leaderboard shakeup

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352
- **Topic id**: 735352
- **Author**: mikelou1 (EXPERT)
- **Posted**: 2026-08-15T08:25:14.079757100Z
- **Votes**: 9
- **Comments**: 11

---

## Opening post

Hey guys! I noticed the top of the leaderboard often has difference less than 0.001 points. In areas like division jaccard where theres only a few samples, it's really easy to overfit. I think this might cause a huge shakeup as even parameter tuning could cause 0.001 changes.

Do you guys have any ideas on this?

---

## Comments (11)


### TWEAK (EXPERT) — 2026-08-15T13:23:10.347Z — 7 votes

I can see why it may appear that way. I can't speak as to what others in the top 10 are doing, but we are not focused on 0.0001 or Division J; we are working on a universal plugin that the bio cell team can plug into their current pipeline with minimal changes. We have tested our plugin with every available unique public notebook and model, with gains ranging from 0.030, 0.040, to 0.050 instantly just attaching our plugin. We've seen gains from a single public model reach a score of 0.940 untuned. We are not focused on the 0.0001 or tuning to the hidden.

#### ↳ mikelou1 (EXPERT) — 2026-08-15T14:04:18.200Z

> thanks for the info! I feel like the public notebooks are really overtuned to the lb so i'm afraid to use them

#### ↳ Sergio Alvarez (MASTER) — 2026-08-15T18:12:20.333Z

> Hi @tweakai, by plugin do you mean something that optimizes/refines the tracking graph, or do you mean you take the public notebook detections and apply your own tracking method on top?
> 
> Got curious about it, but no worries if you can’t share more

#### ↳ ↳ Moawiz (CONTRIBUTOR) — 2026-08-16T09:56:04.003Z

> > yeah i believe it is very interesting to know if the calibration was for keeping the track or reassigning

#### ↳ OpPrime (CONTRIBUTOR) — 2026-08-19T08:44:19.630Z

> that is an impressive approach/tool. Thank you for the information.

### Tang (MASTER) — 2026-08-15T11:43:45.287Z — 4 votes

I guess what we can do is:
- Build a reliable CV, and trust it.
- Use external datasets or synthetic data to make our model more robust on division cells prediction.

#### ↳ nusrati (CONTRIBUTOR) — 2026-08-17T20:06:00.097Z — -1 votes

> did you train your very own model uptil now or used the publicly available one and did u take the public notebook detections and apply your own tracking method on top

### Bharath Varma (CONTRIBUTOR) — 2026-08-15T12:49:38.453Z — 1 votes

yea ig so too , because training past ep 400 has diminishing returns , so maybe yea we wait fro the private shake

### nusrati (CONTRIBUTOR) — 2026-08-17T20:04:44.877Z

you dont mind me asking you, did you train your very own model from scratch or rather you fine tuned public model?

#### ↳ mikelou1 (EXPERT) — 2026-08-18T00:00:15.817Z — 1 votes

> I trained mine from scratch since my division score is quite good [0.3] and edge is really bad [its ~0.01 points below public notebooks] so I'm trying to improve that.

#### ↳ Komil Parmar (EXPERT) — 2026-08-20T04:44:28.813Z — 1 votes

> Trained from scratch. And training here isn't very expensive. Its affordable.
