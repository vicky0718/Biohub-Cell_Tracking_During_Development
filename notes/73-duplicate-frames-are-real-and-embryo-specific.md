# Duplicate frames are real, embryo-specific, and worth about half a thousandth

tom99763 (rank 8, LB 0.954) posted frame indices on 2026-07-11 and nobody followed up.
`claude_dupframes` checked the pixels.

## 1. Exact confirmation

```
6bba_05b6850b   frames identical to their predecessor:  5, 13, 28, 43, 53, 58, 60, 63, 67, 77
tom99763's "duplicated after index":                    4, 12, 27, 42, 52, 57, 59, 62, 66, 76
ours minus one:                                         4, 12, 27, 42, 52, 57, 59, 62, 66, 76
```

Every index, exactly. **10 of 99 frame pairs in that clip are byte-identical.**

## 2. It is an embryo property, not a clip quirk

40 movies sampled, whole-array comparison:

```
6bba_*   25 of 25 movies have duplicates, 4 to 14 each
44b6_*    0 of 14 movies have any
TOTAL    237 duplicated pairs of 3,960 (5.98%)
```

The one apparent exception, `6bba_d2b9fc0c`, has none — so it is near-universal in `6bba`
rather than absolute. And tom99763's "videos sharing the schedule" is visible in the data:
`6bba_4aa14c89` and `6bba_6479435d` duplicate at the *identical* indices
`[4, 6, 13, 26, 31, 55, 56, 57]`, as do `6bba_1ebfb80d`/`6bba_55c70843` and
`6bba_05db0fb1`/`6bba_09961292`. Crops from one original movie inherit its duplication
schedule.

**This matters more than 6% suggests.** Recon §b measured that the `6bba` datasets carry
~95% of the score weight, because `metrics.summarise` weight-averages per-dataset
`adj_edge_jaccard` by `TP+FP+FN` and the `44b6` clips have ~50 edges each. So the duplicates
sit almost entirely inside the part of the score that counts.

## 3. What it is worth, measured on `lb941`'s own output

A cell cannot divide between two byte-identical images. Counting divisions whose parent
frame `t` has `t+1` duplicated:

```
dataset            forks   at a duplicated boundary
44b6_0113de3b         51                          0
44b6_0b24845f         10                          0
6bba_05b6850b          6                          0
6bba_05db0fb1         27                          6
TOTAL                 94                          6
```

**6 of 94 divisions (6.4%) are provably false** — not "unlikely", impossible by construction.
And **5,802 of 115,009 edges (5.0%) cross a duplicated pair**, where the correct association
is the identity map.

Sizing the division veto honestly: `division_jaccard = TP/(TP+FP+FN)`, so removing 6 false
positives from the denominator gains roughly `div_J × 6/(D−6)`. At the fork's reported
`div_J ≈ 0.0625` that is about **+0.0004** on the score, plus a few spurious edges removed.
**Half a thousandth.** Real, certain, and small — the gap to rank 100 is one thousandth.

## 4. The larger version, which is speculation until measured

The 5% of edges crossing duplicated pairs are *not* obviously wrong: with identical images
the detector's candidates coincide and a zero-distance match beats any alternative, so a sane
linker should already get them right.

The place a real gain could hide is **detection**, not linking. The detector is a
`TemporalUNet3D` with `window_size=2` — it sees a window, so frames `t` and `t+1` have
*different* inputs even when the two images are identical, and its outputs at those two
frames can disagree. Any cell found at `t` and missed at `t+1` on identical pixels is pure
inconsistency, and copying detections across the duplicated pair would recover it for free.
`notes/04` measured detection as "essentially the whole contest".

That is worth testing and is not tested. It also cuts against `union`'s result (+3,333 nodes
bought exactly nothing), so it is not a safe bet — it is a *different* kind of node addition,
targeted at frames where the answer is known rather than a threshold loosened everywhere.

## 5. Where this leaves the idea

Two arms, in order:

1. **`dupdiv`** — veto divisions whose parent sits at a duplicated boundary. Cheap, certain,
   ~+0.0004. Implemented as a post-processing step over the written `submission.csv`, since
   duplication is detectable from the mounted test zarrs at inference time.
2. **`dupdet`** — symmetrise detections across duplicated pairs. Larger potential, unmeasured,
   and in tension with the `union` result.

Neither is a substitute for the `sew` sweep, which is the axis that has actually produced a
gain. Both are additive to it: nothing here touches `SECONDARY_EDGE_WEIGHT`.
