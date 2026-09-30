# Final: what actually worked, and what three weeks of tuning did not

Competition closed 2026-09-29. 50 submissions.

## 1. The result

```
submission                          public   private
claude-arm-rd07      2026-09-29     0.953    0.917    <- best on both boards
claude-arm-flow2     2026-09-29     0.947    0.913
claude-arm-lb50      2026-09-27     0.946    0.912
pub947bera/w10       2026-09-21     0.946    0.912
everything else, ~20 arms           0.943 - 0.946
```

`0.946` ranked ~970 of 3,791 on the last board snapshot we hold (09-21); `0.947` ranked ~251.
The plateau was that crowded, so the last two steps are worth far more than their size.

**Selection, resolved.** `flow2` was selected manually while `rd07` was still grading, and the
second slot was deliberately left empty — so Kaggle's automatic selection filled it with the
best-scoring submission, which was `rd07`. **The private score that counts is 0.917.**

Worth keeping as an operational note: leaving a selection slot open is not an omission, it is a
hedge. It costs nothing and it captures a submission that scores after you have stopped
watching — which is exactly what happened here, with the best result of the whole competition
landing seven hours before the close and grading after the manual choice had been made.

## 2. Every gain came from forking code we did not have. None came from tuning.

Three weeks and roughly twenty arms went into post-processing parameters on the public base.
Recounted from the board, not from the validator:

```
detection threshold      0.955 / 0.960          0.941 / 0.942     lost
min track length         4 / 8 / 12             0.943 / 0.945     lost or null
gap close, density gain, retention, SEW, DET    0.943 - 0.945     lost or null
division gates, ILP division weight             0.943 - 0.945     lost or null
learned relink bonus     1.5 -> 5.0             0.946             NULL
```

The learned-bonus work is the sharpest case. Five offline points, a mechanism that predicted
its own saturation point and was proved right by two independent sweeps, +0.0027 adjusted edge
Jaccard with **0% of it from the node-count multiplier** — the one regime two authors above the
plateau said the offline validator could be trusted in. **Board: 0.946, identical to the base.**

The two things that worked were both somebody else's code:

```
flow2   thtennant (0.953)   our base + MOTION_RELINK_FLOW_* only          0.946 -> 0.947
rd07    raunakdey07         our base + FLOW + GAPFILL + READMIT + LOWDET   0.946 -> 0.953
                            + the V1284 coordinate-refinement head
```

`rd07` changes **not one** of the 57 configuration values we spent three weeks sweeping. It
adds 25 keys and the code behind them. That is the finding of this project.

## 3. The one instrument that paid for itself

`tools/scout_notebooks.py` — join every public notebook to the leaderboard **by author**, not
by title. `notes/87` had scouted by title and concluded the frontier was a 0.947 plateau with
nothing runnable above it. The join found six accounts between 0.948 and 0.957 publishing
working forks, and in one afternoon it:

* closed the node-count axis for free, because `zhincez` (0.952) had already spent four
  submissions on exactly the five arms I had just built (`notes/89` §3);
* produced `flow2`, the first thing to beat 0.946;
* and produced the diff that showed what `rd07` adds.

Against that, the offline validator produced twenty arms and no gains.

## 4. The miss worth recording

**I had `rd07`'s stack and threw it away.** `x138` (`anvithpothula`, 0.956) is the same
subsystem set, built 09-22, and it died on

```
RuntimeError: ('my V1284 head mount mismatch', [])
```

I wrote it off as depending on a private dataset and closed the arm. I never checked whether
anyone had **published** that head — and `anvithpothula/biohub-v1284-head-s075` was public the
whole time. The user found it seven hours before the deadline by pointing at a notebook that
mounts it, and it turned out to be worth +0.006.

The rule that would have caught it: **a missing mount is a question, not a verdict.** One
dataset search would have turned it up, and the same author who broke the arm had published
the fix.

## 5. Two other things that cost real time

* **`geofus` timed out three times.** Its 24-candidate sweep costs ~3 h that does not shrink
  under grading while inference scales ~17x, so it never fit the 12 h box. I flagged the risk
  before the first submission and again before the second, then submitted anyway. Flagging a
  risk is not the same as acting on it.
* **A TPU switch cost three submissions.** The pipeline is CUDA-only — the notebook's own first
  cell says so — and selecting TPU removed the accelerator instead of changing it. Then,
  separately, an exhausted weekly GPU quota turned out to block graded reruns too
  (`MEMORY.md`), which I had argued it would not.

## 6. What the score means

`0.953` public on a metric whose maximum is 1.1. The top of the board was 0.974 in the last
snapshot we hold. We finished by adopting, verified, the best published work we could find and
adding nothing of our own that survived measurement — which is an honest result, and a smaller
one than the amount of machinery in this repository suggests.
