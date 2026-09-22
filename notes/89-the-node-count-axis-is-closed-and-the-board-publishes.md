# The node-count axis is closed by someone else's four submissions, and the people above the plateau publish

2026-09-22, 7 days to deadline. Best 0.946, rank ~1,116 of 3,791.

## 1. Two corrections first, because three of yesterday's numbers were artifacts

`notes/88` reported arm node counts by summing every `FINAL: <n> nodes` line in a kernel log.
That count is not the submission's node count — it is the sum over **however many times the
post-processing sweep re-ran the pipeline**, and arms differ in that number. Recounted from
the per-dataset summary the notebook prints for the four test clips it actually writes:

```
arm          FINAL lines   sum of them   TEST-CLIP nodes   submission rows
pub947bera        68        1,633,336        122,794          241,306
lb15              68        1,633,291        122,801          241,325
lb20              72        1,756,246        122,789          241,298
lb20x             12          311,679        122,821          241,362
lb30              68        1,633,397        122,845          241,417
slm50             68        1,636,394        123,002          241,716
gdg08             68        1,633,336        122,794          241,306
```

* **`lb20` did not move node count +7.5%.** It moved it **−5 nodes**. Its sweep evaluated one
  extra candidate, so it has 72 `FINAL` lines instead of 68, and those four extra lines are
  the entire "+122,910".
* **`lb20x` did not collapse node count by 81%.** It has `NO_SWEEP`, so it ran the pipeline
  once instead of six times. Its submission is 241,362 rows — *larger* than the base's.
* **`notes/88` §3 concluded "the bonus curve is not monotonic" from those two figures.** It
  is monotonic, in both node count and Jaccard. That section was wrong.

Rule that follows: **never compare a derived sum across logs with different control flow.**
The submission's own per-dataset summary was in every log the whole time.

## 2. The learned-bonus curve, measured properly

```
bonus   test-clip nodes   Δ vs base   proxy adj   div_J    PROXY
1.0          122,794           —        0.9280   0.2308   0.9511
1.5          122,801          +7        0.9282   0.2308   0.9512
2.0          122,821         +27        0.9285   0.1667   0.9451
3.0          122,845         +51        0.9295   0.2500   0.9545
```

`adj` rises monotonically and node count is flat to **+0.04%**. The node-ratio multiplier
therefore contributes ~4e-5 of the +0.0015; the rest is edge Jaccard. That distinction is
the whole of §3 below, and it is the reason `lb30` survives it.

`div_J` swinging 0.25 → 0.17 → 0.25 is one division event out of twelve. Noise, as `notes/77`
established.

## 3. The node-count exploit is real, already tried, and cost its author four submissions

Deriving it independently this morning: the score is
`adj = edge_J × (1 − 0.1 × (N_pred − N_est)/N_est)`, unclamped above and signed, so deleting
nodes multiplies the Jaccard up. `notes/77` §7 recorded it and declined to chase it. I went
back at it, on the grounds that §7's stated reason was backwards — `det955` and `det960` both
moved the threshold **down**, and all three board points (0.955→0.941, 0.960→0.942,
0.965→0.944) say fewer nodes is better. I wrote five arms (`det97`, `det975`, `det98`,
`mtl12`, `mtl20`) and a pre-registered acceptance rule, and launched two of them.

Then a public-notebook sweep turned up **`zhincez/the-metric-pays-you-to-delete-nodes`**,
16 votes, published 2026-09-08, by an author at 0.952:

```
                                        their validator     their public LB
baseline fork                               0.9368               0.938
min track length 10 + det 0.98 + per-type   0.9495               0.934
their own pipeline, same policy             0.9523               0.865
```

Offline said +0.013. The board said −0.004, and −0.087 on their own stack. They tested
**exactly** my five arms: min track length 8/10/12/15/20 and detection threshold 0.98.

Their mechanism is right and it is the one I missed. Training labels cover 0.2–1.3% of nodes
on one embryo type and 13–15% on the other. Delete a track and the validator only notices if
that track was labelled. **The labels can confirm the reward. They cannot confirm the price.**

And it defeats the acceptance rule I had just pre-registered — "accept if mean `adj` gain is
positive and mean `edge_J` loss is under 0.010" — because their `edge_J` was *flat* and the
entire gain was multiplier. My rule would have waved it through. Their replacement rule is
better and is now ours:

> Split every offline gain into the Jaccard half and the multiplier half. If more than ~70%
> of it came from the multiplier, treat it as unproven until the board says otherwise.

`det975` and `mtl12` were already on Kaggle when this turned up; they will finish and will not
be submitted. **The axis is closed, at a cost of two GPU hours and no submission slots.**

By the same rule, `lb30` passes: 100% of its move is Jaccard, node count flat.

## 4. The sweep that found it: people above the plateau publish, and titles were hiding it

`notes/87` scraped the forum and read notebook *titles*. This sweep cross-referenced all
**700** public notebooks against the board by author, which is a different instrument, and it
turns up authors well above the plateau publishing working code:

```
rank  score  author           what they publish
  34  0.957  mige551          DualSeed logit blends; "Clean 122 det 0.9690"
  41  0.956  anvithpothula    "biohub x138", pushed 09-21
  66  0.953  thtennant        ten "frontier947 <change> v1" notebooks, 09-17..19
  74  0.953  rogerrogerroger3r ~25 notebooks + three writeups
  77  0.952  zhincez          the node-deletion post above; a runnable 0.947
 135  0.949  humblehumbert    "chen learned bonus5 parentledger"
```

Diffing `thtennant`'s and `anvithpothula`'s env blocks against our 0.947 base, both add the
same four groups of keys that **do not exist anywhere in the 0.947 plateau**:

```
MOTION_RELINK_FLOW_*   17 keys   a seeded local-flow model feeding the Hungarian relink
GAPFILL_*               9 keys   image-space gap filling with a peak search
READMIT_*               2 keys   re-admitting detections below the main threshold
LOWDET_THRESHOLD        1 key    0.3 — a second, much looser detection pass
```

plus ~35k characters of code. Their mounts are the same three pilkwang datasets we already
use, so they run here unmodified. `claude-arm-flow2` (the FLOW half alone, on our exact base)
and `claude-arm-x138` are built.

Tempered by what x138's own comments say about itself: two of its monkey-patches anchor on
the same line, one silently disabled the other for a whole version, and its metadata lists a
fourth data source that does not resolve. High ceiling, low reliability.

## 5. `MOTION_RELINK_TIGHT_UM` 5.5 → 6.0, from three independent directions

* `zhincez` (0.952) sets **6.0** in their runnable 0.947.
* `thtennant` (0.953) publishes `frontier947-fast-tight60-v1`, whose only change is **6.0**.
* `beraterolelk` — **the author of the notebook we forked** — pushed v5 on Sep 22 which
  *deletes* the `BIOHUB_MOTION_RELINK_TIGHT_UM = "5.5"` line so the code default **6.0**
  applies, and adds a `tight60` sweep candidate. Their commit calls it the "bronze push".

We have never moved it. `claude-arm-tight60` is built against the **cached v3 source** — the
exact program that scored our 0.946 — so it is one variable against a known board point, with
`NO_SWEEP` because v3's own candidate list contains a `tight55` entry that would override the
arm straight back to 5.5, the way `lb20` was overridden to bonus 1.25.

### And we were forking a stale version the whole time

Every 0.947-base arm we have run — `pub947bera`, `pub947w10`, `lb15`, `lb20`, `lb20x`, `lb30`,
`slm50`, `gdg08` — was built from a **v3** provenance cache. The notebook is at **v5**. The
only functional difference is the tight radius above, so this does not explain the 0.946/0.947
gap on its own (v5 postdates the author's 0.947), but the cache was silently a month stale and
nothing in the builder said so. `--refresh` exists; nothing was making anyone use it.

## 6. What is queued

```
running   det975, mtl12        closed axis, finishing, will not be submitted
queued    tight60              v3 + tight radius 6.0, three-way corroborated
queued    lb50                 learned bonus 5.0 — curve still rising at 3.0, and
                               humblehumbert (0.949) ships exactly 5.0
built     flow2, x138          the subsystems the plateau does not have
CPU       claude-probe         is a node budget shipped on the TEST side at all
```

`lb30` remains the one complete, submission-ready arm, and it now has a second reason to
believe it beyond the proxy: it is pure Jaccard by §3's test.
