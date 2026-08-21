# First leaderboard score — 0.752, rank 1969

`09_submission.ipynb`, submitted 2026-08-21. **Public LB 0.752, rank 1969 of ~2,400.**

The submission path works. That was the point of this run, and it is the single biggest
risk retired in the project: notebook-only rerun, internet off, wheelhouse install,
runtime globbing of `test/`, budget regression refit at runtime, CSV format — all of it
survived contact with the real thing.

---

## 1. ⭐ CV under-predicts LB by +0.045 — the first calibration point we have ever had

| | score |
|---|---|
| CV, `adaptive_predicted`, 60-dataset subset, leave-one-embryo-out | 0.7070 |
| **public LB** | **0.752** |
| difference | **+0.045** |

**This corrects `notes/08` §1.** There I withdrew the forum's "LB runs ~10 % above CV"
claim on the grounds that the rule-based author had published a scatter showing CV ≈ LB at
1:1, and concluded "**CV is the estimate**". Our own measurement says otherwise: LB came in
6.4 % above CV. The withdrawal went too far — the original claim was directionally right,
just overstated at 10 %.

Two caveats on how far to trust +0.045:

- Our CV is a **60-dataset subset**, not the full 199, and the public LB is 29 % of the
  hidden test. Different samples of different data.
- A plausible mechanism is the budget multiplier. On CV our node ratio was **+0.040**
  (slightly over budget, so a small penalty, ×0.9919). If the regression runs slightly
  *under* the true budgets on the hidden embryos, the multiplier flips to a bonus — worth
  up to ×1.1. Some of the +0.045 may be that rather than better tracking.

Either way the practical consequence is the same and it is useful: **our gate is
conservative.** A change worth +0.01 on CV is worth at least that on the board.

## 2. Where 0.752 actually sits, and why the rank is worse than the score suggests

Against the archived 2,402-team snapshot (`discussions/raw/leaderboard.json`):

| target | rank | gain needed from 0.752 |
|---|---|---|
| **0.752 (us, today)** | **1969** | — |
| 0.800 | ~1,723 | +0.048 |
| 0.850 — the published classical ceiling | ~1,521 | +0.098 |
| 0.890 — median | ~1,197 | +0.138 |
| 0.913 | ~491 | +0.161 |
| 0.915 | ~139 | +0.163 |
| 0.935 — gold | ~13 | +0.183 |

The snapshot would have placed 0.752 at rank ~1,797; we are at 1,969. **The board drifts
upward** — roughly 170 places of drift since 2026-08-16 — so a fixed score loses rank over
time. Anything we bank now decays.

Read the middle of that table carefully. **Doubling our remaining classical headroom buys
almost nothing in rank.** +0.048 moves us ~250 places. Even reaching the *best published
classical pipeline* — +0.098, more than we have gained in eight experiments combined —
leaves us at ~1,521, still bottom third. Then +0.063 more, from 0.850 to 0.913, is worth
**1,030 places**.

The leaderboard is not a gradient. It is flat where we are and a cliff at 0.913.

## 3. 🚨 What the cliff is made of

I pulled the source of the public learned pipeline
(`pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery`, scriptVersionId 333208869)
via `/kernels/scriptcontent/{id}/download`. It settles what the 0.913–0.916 cluster is.

**The notebook does not train anything. It is inference over shipped weights.**

```
METHOD           = "unet_transformer"
WEIGHTS_RELATIVE = "weights/unet_transformer/split_0/edge_predictor_best.pth"
TARGET_ARTIFACT  = "/kaggle/input/datasets/pilkwang/biohub-tracking-support-pack-50ep-v1"
EXPERIMENT_TAG   = "candidate_20_200ep_recall_clean_repair"
DET_THRESHOLD    = 0.985      # UNet probability, not intensity
USE_ILP          = 1          # ILP linking, with a division weight of 1.0
```

Training happened offline; the `.pth` is published as a Kaggle Dataset alongside offline
wheels. So **514 teams sit between 0.913 and 0.916 because they are running the same
weights.** Forking it would put us *in* that cluster, around rank 500–1,000 — a real jump,
but on someone else's model, and one that stops improving the moment we land.

Three details worth keeping:

- **They install offline with `pip install --no-index --no-deps --find-links=<dir>`** from
  a wheels directory inside the artifact. Independent confirmation that the wheelhouse
  pattern in `10_wheelhouse.ipynb` is the right shape.
- **The weights ship as a Dataset, separate from the notebook.** That is the architecture
  we need too: train in one notebook on GPU, save weights as a Dataset, have the
  submission notebook load them. Training does not have to fit in the submission's 12 h.
- **`ILP_DIVISION_WEIGHT = 1.0`** — they model divisions. Our division Jaccard is 0.000 in
  every arm we have ever run, and the term is worth 0.1 of the 1.1 maximum.

## 4. Why over-detection has been nearly free, and what that says about the ceiling

Worth stating explicitly because it shapes what to optimise next. A predicted edge scores
only if **at least one endpoint matched a GT node** (`pred_valid = out_valid(source) OR
in_valid(target)`, `notes/02`). Edges between two unmatched predictions are dropped
silently, not counted as false positives.

That is why we can predict ~1.2 M nodes against ~40 k annotated ones and still hold edge
Jaccard at 0.71: the vast majority of our predictions are invisible to the edge term. The
**only** brake on over-detection is the node-budget multiplier — which is exactly why
three quarters of every gain since `04` came from that multiplier rather than from
tracking (`notes/14` §3).

The corollary: with the budget term now nearly exhausted (×0.9919 of a ×1.0 ceiling,
+0.0058 left), further score has to come from **matching more annotated GT nodes and
linking them correctly**. Our node recall is 0.866. That is the axis, and it is a detector
problem, not a density problem.

---

## What to do next

**The classical line is done.** Not because it cannot improve, but because its improvements
no longer buy rank. Remaining known headroom is roughly +0.06 (budget term +0.006,
divisions maybe +0.01, adaptive sigma unknown, linking unknown), which lands us near 0.81
and rank ~1,700. That is not worth five weeks.

1. **⭐ Build a learned detector.** The cliff is a UNet, and every classical pipeline in the
   public set is below it. Phase 1 target is our own weights, trained on the competition's
   own 199 train datasets, submitted through the `09` machinery which now demonstrably
   works. Train in a GPU notebook, ship weights as a Dataset, load at inference — the
   pattern §3 shows is standard here.
2. **Then pretrain, which is the actual differentiator.** 514 teams share one set of
   weights. Nothing a fork can do gets above that cluster, because they are all the same
   model. External pretraining is the one lever the cluster does not have — which is the
   Zebrahub direction, and it looks strategically sound for exactly this reason. Note the
   ordering: pretraining is worth nothing until we have a training pipeline to fine-tune
   *from*.
3. **Zebrahub acquisition is still blocked here.** `zebrahub.org` and
   `public.czbiohub.org` are refused by this environment's egress proxy (403 on CONNECT).
   It has to happen in a Kaggle notebook with internet on, same as the wheelhouse.
4. Keep `09` as the fallback submission. 0.752 is banked, and the board drifting upward
   means it will quietly lose rank — but it is a floor, and it is reproducible.
