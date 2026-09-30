# The 0.953 stack's "private" V1284 head was reproducible from the notebook itself

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744498
- **Topic id**: 744498
- **Author**: Nikolce  (CONTRIBUTOR)
- **Posted**: 2026-09-30T02:39:29.229372Z
- **Votes**: 0
- **Comments**: 1

---

## Opening post

The public 0.953 notebook (anvithpothula/biohub-x138) hard-failed without a private dataset holding v1284_head.pt. It turned out that head was fully reproducible from material the notebook already publishes, and a head we trained ourselves scored slightly higher. Posting the method and, more usefully, the negative results.

WHAT THE HEAD IS
A 7.3k-parameter coordinate refiner: Linear(224,32) -> SiLU -> Linear(32,3), output bounded to 2 um. The 224 features are the UNet's 32-channel map sampled at a detection centre and its 6 face neighbours, encoded as [centre] + [neighbour - centre]. It nudges detection centres closer to truth before association.

WHY IT WAS REPRODUCIBLE
The consuming module is written inline by the notebook, and it documents three modes: candidate, zero, and capture. Capture mode dumps exactly the (coords, features) pairs needed to train it. So no reverse engineering was required - only running the pipeline in capture mode over training movies, matching each detection to its nearest GT node, and fitting the head.

RESULT
Our head (40 movies, 17,975 detection-GT pairs) scored 0.954 public vs the author's 0.953, on an otherwise byte-identical pipeline - the two runs differ by one 34 KB file.

THE MORE USEFUL NEGATIVES
1. More data made it worse on the LB. Scaling to all 199 train movies improved the held-out residual a lot (-28.7% vs -22.7%) but dropped the LB to 0.951. There is an interior optimum. The downstream gates (relink radii, safe-division distances, gap-close steps) are tuned against the original centre-error distribution; a large calibration change makes them mis-fire.
2. The held-out residual does not rank heads. Four heads: -10.5% -> 0.953, -22.7% -> 0.954, -26.0% -> 0.952, -28.7% -> 0.951.
3. Head-to-head spread is ~+/-0.002 against a 0.001 LB quantum. Two independent 40-movie draws gave 0.954 and 0.952, so a single head's score is mostly noise.
4. Re-enabling the runtime PPSWEEP (x138 ships it off) changed nothing: its best candidate was +0.0005 proxy, under the notebook's own 0.001 margin, so it selected base and produced a byte-identical submission.

UNIT GOTCHA
Captured coords index the detector's FEATURE grid, which is the native grid downsampled by (1,4,4). GT geff coords are native. Compared raw, the nearest GT node sits 346 um away; with the downsample applied, 1.62 um. Training on the raw version gives a confident, useless head.

PERSPECTIVE
This was a fork-and-tune line and it had a ceiling: ~0.954 public, 0.924 private, 161/4020. The winners were far beyond it, so treat the above as a note on one mechanism rather than a solution writeup. Thanks to anvithpothula for the original work.

---

## Comments (1)


### Navneet (CONTRIBUTOR) — 2026-09-30T04:54:39.710Z

Thank you for the private V1284 Notebook @nikolce
