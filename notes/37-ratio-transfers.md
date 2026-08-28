# 0.883 → 0.897. Biggest jump yet, and transfer is now settled: it is not a constant.

`claude_submit_ratio`, leaderboard **0.897**. One functional line changed from the 0.883
submission — `appearance` 0.1 → 0.4, `disappearance` 0.5 → 2.0.

```
0.752  classical champion
0.843  pack, ILP bypassed
0.867  pack, ILP running
0.880  + gap-close + linefit-smooth
0.883  + disappearance 0.1 -> 0.5
0.897  + the located ILP optimum (appear 0.4 / disappear 2.0)   <- here
0.926  bronze   0.927  silver   0.944  gold
```

**Rank ~1336 / 2792, top 47.9 %** — above the 0.894 median for the first time. Bronze is
**+0.029** away, gold **+0.047**.

---

## 1. Transfer is definitively not a constant, and the intervals now prove it

Three measurements, with leaderboard rounding propagated so the comparison is honest:

```
                 train      LB delta          implied ratio
repair chain    +0.0115   (0.0120, 0.0140)   (1.04, 1.22)
ILP asymmetry   +0.0037   (0.0020, 0.0040)   (0.54, 1.08)
ratio0.4_2.0    +0.0221   (0.0130, 0.0150)   (0.59, 0.68)
```

`notes/32` §2 said the first two "probably differ, and the evidence that they differ is
weak," because their intervals overlapped on a sliver. **This one settles it**: (1.04, 1.22)
and (0.59, 0.68) do not overlap at all, and this interval is tight precisely because the
effect is large enough that 3-decimal rounding barely blurs it.

So: a train-measured gain is **not** convertible to a leaderboard gain by any fixed factor.
The repair chain over-delivered; this arm delivered under two thirds. What survives is the
weaker and more useful claim — *direction* has transferred every single time, four for four.

## 2. Which half fell short is not resolvable, and I am not going to pretend otherwise

`notes/35` §3 decomposed the train gain and flagged the risk in advance:

```
divisions   +0.0115    div_J 0.0000 -> 0.1154
node budget +0.0106    ~10% over budget -> ~2% under
raw edges   -0.0003
```

and said "the budget half is a calibration… the part most at risk on test." The outcome is
consistent with that, but **so is the alternative**, and both fit inside the measured band:

```
divisions transfer fully, budget at ~0.25:  0.0115 + 0.0027 = 0.0142   inside (0.0130, 0.0150)
both transfer uniformly at 0.63:            0.0221 x 0.63   = 0.0139   inside (0.0130, 0.0150)
```

`division_jaccard` on test needs the hidden ground truth, so the split cannot be measured
from here. The pre-registered *direction* of the concern was right; its *attribution* stays
open. What is known: the node-shedding mechanism did fire on test at almost exactly the
train magnitude (−10.0 % nodes vs ~−11 %), so the shortfall is not the mechanism failing to
engage.

## 3. What this closes

**The ILP's weights are done.** Three sweeps, 69 settings, a located interior optimum
(`notes/36`), and now a leaderboard confirmation worth +0.014. There is no more score here.

The remaining gap is entirely the **edge term**:

```
ours:              raw edge_J 0.9047      div_J 0.1154   (competitive — notes/33: field reports 0.12)
public notebooks:  0.923–0.927 total
```

Divisions are no longer where we lose. We lose on edges, and the lever for edges is the one
still untouched.

## 4. Next: the second missing model

`notes/33` §1 found the 0.927 public notebook attaches three model datasets where we attach
one. The deepcenter veto was the first and it closed at a ~0.002 ceiling (`notes/34`). The
second is the one that acts on edge probabilities:

- **`pilkwang/biohub-temporal-unet3d-seed314159-v1`** — a secondary linker whose edge logits
  are mean/std-calibrated onto the primary's scale and blended with a margin-adaptive weight
  (`SECONDARY_EDGE_WEIGHT 0.15`, `LOW_MARGIN_MAX 0.35`).
- **Bidirectional harmonic linking** (`notes/33` §2) — run the linker forward *and* reverse
  in time and take a weighted harmonic mean, so a link survives only if it is plausible read
  both ways. **Needs no new weights at all.**

Unlike the last three runs this needs a real prediction pass rather than cached ILP
instances, so it is the first expensive build since the veto.

Banked floor **0.752**. Best scored **0.897** (rank ~1336/2792).
Bronze **0.926**, gold **0.944**.
