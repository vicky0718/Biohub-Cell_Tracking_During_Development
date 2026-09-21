# The field moved, we did not — and we are now BELOW the public notebook

Leaderboard and forum re-scraped 2026-09-21 (previous scrape 09-07). 8 days to deadline.

## 1. We lost 486 places without our score changing

```
                      2026-09-13        2026-09-21
teams                      3,460             3,779
our score                  0.945             0.945
our rank                    ~648            1,134
top score                  0.961             0.974   (Sergio Alvarez)
rank-100 cutoff            0.948             0.951
```

Where a score lands **now**:

```
0.945  ->  ~1173      0.948  ->  ~241
0.946  ->  ~1116      0.950  ->  ~123
0.947  ->   ~945      0.955  ->   ~52
```

## 2. The headline: **704 teams are tied at 0.947, and we are at 0.945**

```
0.952   16 teams        0.947  704 teams   <- public notebook plateau
0.951   15              0.946  171
0.950   19              0.945   57         <- us
0.949   30              0.944   39
0.948   88
```

**We are 0.002 below the public notebook.** Our `ttasec` is `tta946` (0.944, forked 09-08) plus
our own +0.001. The public stack has since moved to **0.947** and we never re-forked.

Verified against the board rather than titles — `notes/74`'s rule, and it mattered again
(`andnyu`'s "0.948 Reproduction" sits at 0.947):

```
author          rank   score     notebook
haideptry        304   0.947     biohub-sota-0-947-fast-2xt4-22m-divnet   <- 22 min runtime
lingxd           324   0.947     biohub-repro059-public-0947-exact-copy
beraterolelk     522   0.947     0-947-lb-biohub-deepcenter-ilp-tracker   (27 votes)
shienzhang       838   0.947     biohub-public-0947-exact-repro
reyhanksatria    783   0.947     (our tta946 base author — they moved up too)
```

**What changed in the public stack: a HOCT veto.** It is now
*"dual-seed UNet-transformer + DeepCenter + HOCT veto"*. HOCT is the host's open-source
division-aware tracker (`musculer/biohub-hoct-general-v0-official`), run as a veto stage. We
have no HOCT at all. It carries `BIOHUB_HOCT_DEADLINE_H = 10` and **skips any video it cannot
finish in time** — which is also the shape of `ftune`'s graded-rerun failure (`notes/86`).

## 3. Fifteen new threads. Two are from top-50 teams, and neither is technical

```
rank  score  author             thread
   6  0.967  Masha Mikhisor     A few questions about external zebrafish data
  21  0.960  Mark               What is your best solo model?
```

The rank-6 thread is about sourcing **external validation embryos**; Sergio Alvarez (rank 1,
0.974) replies that the host tracks are *"based on their methods (ultrack?/hoct?) + manual
review"*. The rank-21 thread has no content. **The top of the board is not discussing method.**

The substance is in threads from the 0.947 plateau, and it corroborates our own notes almost
line for line:

* **hikaggler** (rank 258): three weeks of model-side changes, one variable at a time on a
  24-video stratified hold-out. Nothing moved — wider U-Net, longer training, strong
  augmentation, wider temporal window, a trained division head. *"Post-processing knobs
  (gap closing, short-track removal, thresholds): saturated"* — our `notes/85`, independently.
  **Local CV r ≈ −0.2 against the LB in the 0.93–0.95 band** — our `notes/81`, independently.
  *"node-count calibration predicted my LB movement far better than missed detections"* — our
  `adj = J·(1 − 0.1·ratio)` analysis, independently.
* **Justin CH123** (rank 484): ten single-knob board tests, all lost, *"the loss tracked the
  change in predicted node count, not my offline validation score"*. Also identifies the
  Hungarian motion relink replacing the ILP edge set — our `notes/79`, independently.
* **Eric** (rank 82): the public weights' `split_manifest.json` lists **all 199 training
  videos**, so any hold-out drawn from train is in-sample. That is `notes/81`'s contamination
  finding, confirmed from the manifest rather than inferred from a failed prediction.
* **A trained division head**: local `division_jaccard` 0.07 → 0.19, **±0.001 on the LB**.
  Our division work reached the same place from the other direction.

**The one lever anyone reports working**: pretraining on the public synthetic dataset,
**+0.012 to +0.018** (hikaggler), the largest single number in any thread. The dataset is free
— 18.5 GB, 165k labelled divisions (thread 732103, 47 votes).

## 4. What this changes

The cheapest real gain available is no longer anything we invent: it is **catching up to the
public notebook**. Fork a board-verified 0.947, then re-apply the one patch of ours that ever
scored (secondary-TTA, +0.001, and disjoint from a HOCT veto). 0.947 alone is rank ~945;
0.948 is ~241.

Against it: additivity has failed three times here (`notes/85`), so +0.001 on top of 0.947 is
a hope, not a plan. And `haideptry`'s 22-minute variant matters more than its score — our
graded reruns are running at ~11 h of a 12 h box, which is what killed `ftune`.

## 5. Correction: the "+0.012 to +0.018 synthetic pretraining" number does not mean what I said

I reported hikaggler's synthetic-pretraining gain as *"the one lever anyone reports working"*
and the largest number in any thread. Checked against the board, that framing is wrong.

```
author                  rank   score   what they published
hikaggler                258   0.947   reported +0.012-0.018 from synthetic pretraining
josefreitasalvesneto     968   0.946   CREATED the synthetic dataset
daifanhao                271   0.947   published synthetic-detector-pretrain-public-v1
hitoshisaito             155   0.948   published synthetic16-cc0-subset
bhpepper                 118   0.950   published synthetic-5fold-ensemble-v1
```

**Nobody using synthetic pretraining is demonstrably above the public plateau.** The dataset's
own creator sits at 0.946 — *below* 0.947. And hikaggler says so plainly in their own thread,
in a reply I read and did not weigh:

> "the 0.947 next to my name is a public notebook score, not mine. My own pipeline, run end to
> end by me, is at 0.939."

So the +0.012-0.018 took **their own pipeline from ~0.92 to 0.939**. It is a real gain inside a
weaker stack, and it still lands **0.008 below** the public notebook. It is not a lever over
the public frontier, and I presented it as one.

The only name in that group above the plateau is `bhpepper` at 0.950, whose dataset is
`synthetic-5fold-**ensemble**-v1` — the ensembling, not the synthetic data, is the part that is
not already in the public stack.

**Practical consequence.** I was one step from proposing an 8-day, 18 GB training build on the
strength of that number. It would have been the `norelink` mistake again in a more expensive
form: a figure measured in one context, carried into another where it does not hold. The
build is off.
