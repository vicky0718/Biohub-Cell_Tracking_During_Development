# Possible big leaderboard shakeup

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/735352
- **Topic id**: 735352
- **Author**: mikelou1 (EXPERT)
- **Posted**: 2026-08-15T08:25:14.079757100Z
- **Votes**: 9
- **Comments**: 12

---

## Opening post

Hey guys! I noticed the top of the leaderboard often has difference less than 0.001 points. In areas like division jaccard where theres only a few samples, it's really easy to overfit. I think this might cause a huge shakeup as even parameter tuning could cause 0.001 changes.

Do you guys have any ideas on this?

---

## Comments (12)


### Georgy Mamarin (MASTER) — 2026-09-17T15:49:49.820Z — 2 votes

I looked up how much boards like this one moved once the private part was scored: every medal-awarding microscopy or histopathology image competition since 2019 with 500 or more teams, a public/private split and two final picks, thirteen boards on cells, tissue and cryo-ET. Eleven were code competitions, Recursion 2019 and HPA 2019 took file submissions, and none is a tracking task. For each: how many of the public top 10 stayed in the top 10, how much of the public medal zone (every rank down to the last bronze, 100 to 216 teams on these boards) still held a medal, and how many teams took no medal while one of their own submissions would have earned one, counted in hindsight, one team at a time with everyone else left where they finished.

| board | top 10 kept | medal zone kept | no medal, one in own submissions |
|---|---|---|---|
| Recursion 2019 | 9 | 97% | 0.1% |
| CZII cryo-ET 2025 | 9 | 96% | 0.2% |
| Sartorius 2021 | 7 | 95% | 0.2% |
| HuBMAP organ 2022 | 7 | 94% | 0.3% |
| HPA 2019 | 6 | 87% | 0.7% |
| HPA single cell 2021 | 7 | 67% | 0.4% |
| UBC-OCEAN 2024 | 5 | 63% | 2.9% |
| BYU motors 2025 | 6 | 60% | 5.5% |
| HuBMAP kidney 2021 | 0 | 50% | 5.4% |
| PANDA 2020 | 3 | 46% | 5.6% |
| HuBMAP vasculature 2023 | 3 | 44% | 18% |
| SenNet vessels 2024 | 4 | 36% | 9.3% |
| Mayo Clinic STRIP AI 2022 | 0 | 9% | 6.1% |

On the eight boards that kept 60% or more of the zone, at least 5 of the public top 10 stayed; on the five that kept half or less, the top ten fell to 0, 3, 3, 4 and 0. On four of those five, 5 to 18% of all teams went home without a medal they had already scored in a submission that did not count, against 0.1 to 5.5% on the other eight, 6 of them under 1%; the plain best-public pick would have caught 21 of those 413 teams. Mayo is the fifth: its bronze line fell inside 68 teams tied on a constant prediction, and 42 of its 99 medals went to that tie, so its row is as much a tie artefact as a reshuffle. The public share does not sort the table: SenNet showed 67% of the test and collapsed, CZII showed 26% and held; this competition shows 29%.

Nothing on the public board says which end of that table this competition lands on. One number that bears on that risk is @ramarlina's ±0.14 movie-to-movie spread on a leave-one-embryo-out hold-out in [the CV thread](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/730160), where the public board runs about 10% above that hold-out.

Numbers from my [dataset of public and private scores for past competitions](https://www.kaggle.com/datasets/georgymamarin/kaggle-submission-histories).

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
