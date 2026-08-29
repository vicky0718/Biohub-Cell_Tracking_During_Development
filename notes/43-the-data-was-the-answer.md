# One external embryo has 10x the labelled nodes of the entire competition training set

Two results from `claude_divdata`. The first closes a direction I proposed one note ago.
The second is the strongest evidence this project has produced for training on more data,
and it is not the reason anyone expected.

---

## 1. The division classifier is dead, and `notes/42` §3 needs correcting

```
199 datasets    nodes 133,318    divisions 151   (0.11% of nodes)
per dataset:    min 0   median 0   max 5   mean 0.8
```

**151 division events in the whole training set.** Prediction 1 wanted 10,000 and it was
written as a kill switch precisely so this would cost one CPU notebook instead of a GPU
week. A per-node division classifier cannot be trained on 151 positives with a median of
zero per dataset. Closed.

### What survives of the headroom argument, and what does not

`notes/42` §3 said the headroom was in divisions and that is where the effort should go.
The arithmetic was right and the mechanism was wrong.

`division_jaccard = TP / (FP + D)` with `D` fixed by ground truth. On the 12 measured
datasets `D ≈ 9`. We score `div_J = 0.0645`, so `TP/(FP + 9) = 0.0645`. The term is worth
up to 0.1 of the total and we collect 0.0065, so the 0.094 is real arithmetic — but it
rests on **nine events across twelve movies**, and the way to move it is not to find more.
It is to **stop emitting false ones**:

```
FP ≈ 50, TP = 2   ->  div_J 0.03
FP = 20, TP = 2   ->  div_J 0.07     (roughly where we are)
FP =  0, TP = 2   ->  div_J 0.22     +0.016 on the score
FP =  0, TP = 9   ->  div_J 1.00     +0.094, and requires perfection
```

`notes/39` counted forks: 53 at appearance 0.13, 2,477 at 2.0; 51 down to 20 as
disappearance goes 0.5 → 2.0. We emit tens of forks against nine real divisions. Cutting
false forks toward zero is worth **+0.016** without any model at all, and it is an ILP
weight question — cheap, and never once asked with `div_J` as the thing being read.

So: right target, wrong mechanism, and no training data involved. Corrected here rather
than left standing.

## 2. The competition's ground truth is sparse. Astonishingly sparse.

```
133,318 GT nodes / 199 datasets / 100 frames  =  ~6.7 annotated cells per frame
our pipeline predicts                            ~220 cells per frame
```

The published model was trained on **all 199 datasets** — and all 199 datasets amount to
133 thousand labelled nodes. It reports `test_recall 0.9755` on that, then runs on frames
holding thirty times more cells than it was ever supervised on.

## 3. `kkunizaw/biohub-zh001r`, measured rather than assumed

```
zh001r_iso.npy   (72, 20, 64, 64, 64) uint8    72 clips x 20 frames, isotropic 64^3
zh001r_tgt.npy   (72, 20, 64, 64, 64) uint8    detection target, same shape
zh001r_nodes.npz 1440 per-frame arrays, (N, 4) float32 = (t_in_clip, z, y, x)
                 551 / 926 / 1502 nodes per frame (min / median / max)
```

```
competition GT nodes, ALL 199 datasets :   133,318
zh001r nodes, ONE embryo               : 1,357,051      10.2x
```

**One external embryo carries ten times the labelled nodes of the entire competition
training set.** Not because it is more embryos — it is one — but because it is *densely*
annotated: every cell in every frame, 926 per frame against the competition's 6.7.

The geometry lines up exactly. Competition volumes are `(100, 64, 256, 256)` at scale
`(1.625, 0.40625, 0.40625)` µm/voxel — a physical field of `104 × 104 × 104` µm. zh001r is
64³ isotropic, which at 1.625 µm/voxel is `104 × 104 × 104` µm. **Same field of view,
resampled to isotropic** — and 64³ isotropic is precisely the geometry `notes/16` §5
derived and `pipeline/unet.py` was built for, before anyone had looked at this file.

Coordinates are already in that voxel frame, and the target heatmap ships with it. This is
a **detector** training set: volume, target, node positions. There are no track edges, so
it cannot train an edge model.

## 4. Why this bears on the failure that killed every prior learned arm

`notes/21` diagnosed it, and the diagnosis was not label scarcity: it was **temporal
incoherence**. A per-frame detector finds different-but-plausible cells frame to frame, so
edges between them fail to match ground truth even where node recall is high, and it lands
*below* the independence bound — active mispairing — while a fixed DoG filter sits above it
for free, because the same filter responds identically to the same structure.

`notes/23` §2c measured the cost: a learned detector's CV→LB offset is 0.061 worse than the
classical champion's, so a learned arm needs **CV ≈ 0.813** to be expected to match the
0.752 classical floor. Five from-scratch runs topped out at **0.649**.

Every one of those was trained on the competition's sparse labels. And sparse supervision
is a plausible *cause* of exactly that degeneracy: with ~7 labelled cells per frame out of
~900 present, almost any self-consistent subset satisfies the loss. The model is never told
which cells to commit to, so it does not learn to commit to the same ones twice. Dense
supervision removes that freedom — every cell is labelled, so there is one right answer per
frame and coherence is no longer optional.

This is a **hypothesis with a mechanism**, not a hope that more data helps. It is also
falsifiable cheaply: `notes/21` already built `paired_recall` for precisely this, scoring
position against the independence↔perfect scale, so a pilot either moves off the per-frame
baseline in a few epochs or it does not.

What it is not: a claim that this reaches the bar. Nothing built so far is within 0.16 CV
of 0.813, and one embryo of Ultrack-generated labels — algorithm-produced, not hand-curated,
confirmed on the forum — is a domain shift as well as a density gain. `biohub-zmnscrops`,
the 12.9 GB companion, now returns **403**, so this is the whole external option.

## 5. Where this leaves things

```
0.901  submitted    0.926 bronze    0.944 gold
```

Closed here: the division classifier (151 events).
Corrected here: `notes/42` §3 — divisions are worth ~+0.016 by suppressing false forks, an
ILP question with no model in it.
Open, and the first learned direction with a mechanism that addresses the recorded failure:
a detector trained on 1.36M dense labels instead of 133k sparse ones.

`claude_widecv` still running — n=60, paired grading, settles whether the config axes were
ever real.
