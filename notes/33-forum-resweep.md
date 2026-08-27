# The board moved, my recorded picture was stale, and 0.883 is below the median

Re-scraped the forum while `claude_ilp_sweep2` runs: **79 threads (8 new), 30 with new
comments, 257 comments total.** The leaderboard came back with it, and it says something
this project's notes had wrong.

```
2,792 teams        top 0.962        median 0.894
GOLD    ranks   1-15    cutoff 0.944
SILVER  ranks  16-139   cutoff 0.927
BRONZE  ranks 140-278   cutoff 0.926

>>> 0.883 ranks ~1671 / 2792 — top 59.8%, below the median, no medal
    to bronze  +0.043      to gold  +0.061
```

Every note since `notes/24` has closed with "Cluster **0.913–0.916**." That number was
never wrong, but the framing built on it was: I was treating it as the frontier. It is not
— it is the **public-notebook lineage**, and 947 of 2,792 teams sit in the 0.91–0.92 band
alone. The frontier is 0.944+.

---

## 1. 🚨 The public notebooks score 0.923–0.927 and use two models I do not attach

This is the finding. `evgendvorkin/biohub-0-927-lb` (59 votes) declares three dataset
sources; I use one of them:

```
pilkwang/biohub-tracking-support-pack-50ep-v1        <- the only one I attach
pilkwang/biohub-deepcenter-unet3d-center-prior-v1    <- not attached
pilkwang/biohub-temporal-unet3d-seed314159-v1        <- not attached
```

Both missing datasets are public, from the same author as the pack, attachable exactly like
it, and require **no training**. `rockerritesh/0-926-biohub-divsub` uses the same three.

So the ~0.04 between my 0.883 and the public 0.927 is not tuning I have failed to find. **I
have been carefully post-processing a single-model base while the public field runs three
models.** Every gain this project has measured — repair +0.013, ILP asymmetry +0.003 — was
banked on a base weaker than what is freely downloadable.

Read from the notebook's own configuration, the three models are not an ensemble average.
Each has a distinct job:

| model | role | mechanism |
|---|---|---|
| support pack (mine) | primary detector + linker | as I use it |
| **deepcenter** | **add-only veto gate** | every gap-closed node and every geometric division candidate must score above a threshold on a second detector's heatmap (`GAP_THRESHOLD 0.25`, `SAFE_DIV_THRESHOLD 0.12`) before it is accepted |
| **temporal (seed 314159)** | secondary linker | edge logits mean/std-calibrated onto the primary's scale, then blended with an adaptive weight from each model's top-2 margin (`SECONDARY_EDGE_WEIGHT 0.15`, `LOW_MARGIN_MAX 0.35`) |

**The deepcenter veto is the direct fix for a cost I measured and accepted.** `notes/27` §1
and `notes/31` §3 both recorded that repair and asymmetry pay a detection tax — `fn_detect`
+36 at the best arm — because `close_gaps` inserts synthetic nodes blind. They insert the
same nodes and then ask a second detector whether anything is actually there.

## 2. Bidirectional harmonic linking — no new weights needed

```
BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT = 0.25
BIOHUB_BIDIRECTIONAL_FUSION_MODE = harmonic_probability
```

Run the linker forward *and* reverse in time, calibrate the reverse logits onto the
forward's scale, convert both to probabilities, and take a weighted **harmonic** mean. The
harmonic mean is the point: it collapses when *either* direction assigns a candidate low
probability, so a link only survives if it is plausible read both ways.

This runs on the model I already have. It is the cheapest item on this list.

## 3. Two of my own findings were independently replicated, and one was contradicted

**Replicated — the division weight is a wash.** Their config carries the comment
`DIVISION_WEIGHT = 1.0  # reverted after testing -- 0.3/1.0/2.0/3.0 all scored 0.915`.
That is `notes/31` §1 from the outside: the knob saturates and prices to zero. Note they
sampled 0.3/1.0/2.0/3.0 and so, like v1, **never sampled (0.5, 1.0)** — the gap
`claude_ilp_sweep2` is sitting in right now.

**Replicated — the gap-closing radius.** They use `GAP_CLOSE_UM = 5.8`; `notes/27` swept to
**5.75**. Independently derived, 0.9 % apart.

**Replicated, and more strongly than I expected — the disappearance asymmetry.** They run
`APPEARANCE_WEIGHT = 0.0`, `DISAPPEARANCE_WEIGHT = 1.5`. I submitted 0.1 / 0.5 for
+0.003 and `notes/31` §3 warned the optimum might lie past the grid boundary. It does, and
by a lot. **`claude_ilp_sweep2` sweeps disappearance to 2.0 — it brackets their setting**,
which was luck, but it means prediction 4 is being asked in the right range.

**Contradicted — motion relink.** They run `MOTION_RELINK_LEARNED_BONUS = 1.0`. `notes/29`
and `notes/30` closed local relink as dead from both ends. The distinction I can defend:
mine was a *post-hoc override* of a finished ILP solve, theirs runs inside the pack's own
pipeline before the graph is final. That is a real difference, not a save — but "relink is
dead" was too broad and should read "post-hoc relink over a finished solve is dead."

## 4. What the forum says about the layer the gains live in

- **Tang (MASTER)**, on where to work: *"detection -> linking -> division. detection should
  come first, once detection is solid, it's easier to improve others."* And separately:
  *"the current ckpt has kind of hit a wall, it's hard to get more gain from
  post-processing alone."* Which is Track A, named.
- **Soheil Ayati** (12 votes): *"Break your missed edges into two categories: missing
  endpoint nodes and incorrect associations. In my case, many 'linking' issues actually
  originated earlier during node selection."* That is `pipeline/anatomy.py`'s
  `fn_detect` / `fn_mislink` split, arrived at independently. At my best arm the ratio is
  254 : 382.
- **Mendrika Ramarlina (MASTER)**: *"the public LB is more optimistic by almost 10%…
  movie-to-movie variability is large: ±0.14, 18% CV. Worst movie scores 0.460, best
  0.984."* Leave-one-embryo-out for model selection.
- **Divisions are not dead for everyone.** mikelou1 (from scratch) reports `div_J` **0.3**;
  kevin park reports **0.12**. At the 0.1 weight that is +0.03 and +0.012 of score. I score
  **0.000** on the submitted arm. That is most of my gap to bronze, on a term I closed
  twice and reopened once.

## 5. Training from scratch: the 725 h figure was for the wrong configuration

`MEMORY.md` closed Track B on 725 h for the pack's 402-epoch recipe. The forum's measured
numbers:

```
Davit Khantadze, Kaggle T4   2 h/epoch, then 50 min/epoch after fixing precision
mikelou1, own GPUs           4 h for 50 epochs   (~5 min/epoch — a real GPU is ~10x a T4)
Bharath Varma                "training past ep 400 has diminishing returns"
Komil Parmar (EXPERT)        "training here isn't very expensive. Its affordable."
```

At 50 min/epoch on a T4, 400 epochs is **~333 h**. Kaggle's quota is 30 h/week and the
deadline (2026-09-29) is ~4.7 weeks out — **~140 h of quota against 333 h of need.** So the
number was too high but the conclusion survives: *on Kaggle quota alone*, the full recipe
does not fit. 50 epochs (~42 h) fits and would only reproduce the checkpoint I already have.

**This is the one open question I cannot resolve from here: is there a GPU available
outside Kaggle?** mikelou1's 4 h/50 epochs makes the whole thing tractable on one decent
card. On Kaggle alone it is not.

## 6. Next, in the order the evidence supports

Nothing here needs training or a submission slot to *build*; items 1–3 are attach-and-run.

1. **Attach the two missing models.** Largest single measurable gap, zero training. Build
   `claude_submit_trimodel` and measure each addition separately on train before submitting
   — the point is to learn which of the two carries the gain, not to reach 0.927 blind.
2. **Bidirectional harmonic linking.** Runs on the model I already have.
3. **Gate `close_gaps` and division insertion on the second detector**, which prices out the
   `fn_detect` tax `notes/27` and `notes/31` both measured and accepted.
4. **Extend the disappearance sweep past 2.0** once `claude_ilp_sweep2` reports — their 1.5
   with appearance 0.0 is outside anything swept so far on the appearance axis.
5. `notes/32`'s free check still stands: read the leaderboard to more than 3 decimals and
   both transfer-ratio intervals collapse.

Banked floor **0.752**. Best scored **0.883** (rank ~1671/2792, below median).
Bronze **0.926**, gold **0.944**.
