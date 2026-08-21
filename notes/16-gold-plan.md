# The plan for gold — what 0.935 actually requires

Decided 2026-08-21. Goal: **build our own detector properly and land in the gold range.**
Deadline 2026-09-29, so **5.5 weeks**.

## 1. The target, from the leaderboard's own medal field

| medal | teams | threshold |
|---|---|---|
| **GOLD** | **14** | **≥ 0.9350** |
| silver | 105 | ~0.916 |
| bronze | 120 | ~0.9150 |
| — us | — | 0.752, rank 1969 |

Top of board 0.9500. Rank 10 = 0.9390, rank 25 = 0.9300, rank 100 = 0.9160.

**Gold is +0.183 from here, and +0.020 above the public fork cluster.** That second number
is the one that matters. 514 teams sit at 0.913–0.916 running the same shipped `.pth`
(`notes/15` §3), and that cluster lands at rank ~100–240 — bronze at best. So forking is
not a path to gold even as a stepping stone; gold requires beating that model, not
matching it.

## 2. What the score is made of, and where the headroom is

`score = adj_edge_jaccard + 0.1 × division_jaccard`, and
`adj_edge_jaccard = edge_Jaccard × budget_multiplier`, where the multiplier **exceeds 1
when under budget**, up to ×1.1.

`notes/04` §7 fed ground-truth nodes back in as detections and scored the whole train set:

| linker | edge Jaccard | adj edge Jaccard |
|---|---|---|
| Hungarian | 0.9915 | **1.0825** |
| greedy NN | 0.9847 | 1.0750 |

**With perfect detection and zero divisions the score is 1.0825** — comfortably past gold,
and past the 0.9500 at the top of the board. So the leaders are at ~88 % of the
perfect-detection ceiling, and every point between us and them is detection.

Three consequences, all of which cut against copying the public architecture:

1. **Linking is worth ≤ 0.015.** Optimal assignment beats naive nearest-neighbour by
   +0.0068, and is itself within 0.9 % of perfect. The public pipeline's transformer +
   ILP + gap repair are competing for that sliver. **We keep Hungarian and spend nothing
   here.**
2. **Divisions are worth ≤ 0.1 × J_div**, and only 0.117 % of annotated nodes divide
   (`notes/04` §5b). Ours is 0.000. Real but second-order; revisit after the detector.
3. **The budget multiplier is a lever, not just a tax.** We have been treating it as
   something to avoid being punished by. A *precise* detector can go deliberately under
   budget and be paid for it — which is the only way the arithmetic reaches 0.935.

## 3. The detector profile gold requires

Today: **1.21 M detections covering 86.6 % of GT**, node ratio +0.04, multiplier 0.992,
edge Jaccard 0.713, score 0.752.

A plausible gold profile: recall ~0.95 at roughly **half** the node budget → multiplier
~1.05, edge Jaccard ~0.90, score ~0.945.

That is **~2.2× better recall-per-detection than we have now**. Is that a realistic step?
The precedent says maybe: `notes/08` §3 measured that at a *matched* detection budget, DoG
recovers 0.67 of dim nuclei where raw intensity thresholding recovers 0.17 — roughly 4× on
the hard cases, and that swap was worth +0.12 on CV. DoG → learned is the same move again,
one rung up. Not guaranteed. Not fantasy either.

**The quantity to optimise, and to report in every experiment from here, is recall at a
matched node count.** Not recall. `notes/09` §2 is the cautionary tale: we reached 97.6 %
recall and *lost* 0.234 of score, spending 571 spurious detections per extra GT node.

## 4. Why the sparse annotation is not the obstacle it looks like

~1/28 of cells are annotated, so a naive "annotated = positive, everything else = negative"
target labels 27 of every 28 real cells as background.

But `notes/04` §5b measured that the annotated subset is a **uniform random spatial
sample** — clonal clumping NOT FOUND, observed-vs-uniform NN ratios 0.9–1.5, only 5 of
~120 datasets clumped. Two things follow:

- **Which cells are annotated is unlearnable, and we should not try.** The target is all
  cells; annotations are a random probe of that population.
- **A model trained on the annotated 1/28 generalises to the other 27/28 for free**,
  because they are drawn from the same distribution. The annotated sample is ~40 k
  positives in our 60-dataset subset — ample.

The only real hazard is the *contradictory gradient*: an unannotated real cell pushed
towards zero. The fix is not to label it at all. Three candidate losses, to be measured,
not assumed:

| variant | positives | negatives | masked |
|---|---|---|---|
| A — naive | annotations | everything else | nothing |
| B — masked | annotations | clearly-empty voxels only | the ambiguous middle |
| C — PU | annotations | everything else, reweighted by the known prior | nothing |

C is attractive because **we know the prior exactly, per dataset**:
`n_annotated / estimated_number_of_nodes`. B's "clearly empty" is definable without
circularity: low normalised intensity **and** low DoG response **and** > 7 µm from any
annotation. Note B must *not* mask everything DoG likes, or the model only learns to
reproduce DoG.

## 5. Shape of the system

Input geometry is convenient. At `downsample=(1,4,4)` — the stride we already use — a frame
is **64 × 64 × 64 voxels, isotropic at 1.625 µm**. That is a natural 3D UNet input, and
199 × 100 = 19,900 of them is a normal-sized training set. Cell spacing is ~8 µm ≈ 5
voxels, nuclei 4–8 µm ≈ 2.5–5 voxels, and the metric's match radius is 7 µm ≈ 4.3 voxels,
so ±1 voxel of localisation error is acceptable.

- **Train** in a GPU notebook, internet on, `pip install` allowed.
- **Ship weights as a Kaggle Dataset**, which is what the public pipeline does and what
  makes training time independent of the submission's 12 h.
- **Infer** in `09`'s machinery, which already works end to end. A small 3D UNet over
  20,000 volumes of 64³ is minutes of GPU, so inference is not a runtime risk.
- **Keep Hungarian linking, pruning, and the predicted-budget calibration.** All three are
  measured, gated, and independent of the detector.

## 6. Phases, and what has to go right

| phase | question | gate to proceed |
|---|---|---|
| **0** | which loss survives 1/28 annotation? | a UNet beats DoG on recall at matched node count, held-out embryo |
| **1** | does that become score? | LB > 0.80, i.e. clearly past the classical ceiling |
| **2** | can it reach the cluster? | CV/LB ≈ 0.91 |
| **3** | can it beat the cluster? | ≥ 0.935 |

Phase 3 is where the external-data idea earns its place. Every team in the cluster shares
one model, so the differentiator has to be something they do not have — more data being
the obvious one, and the Zebrahub direction the obvious source. **But it is phase 3, not
phase 0**: pretraining is worthless without a training pipeline to fine-tune from, and
`zebrahub.org` / `public.czbiohub.org` are refused by this environment's egress proxy
(403 on CONNECT), so acquisition has to happen in a Kaggle notebook with internet on.

**Honest risk statement.** 5.5 weeks is enough for phases 0–2 with discipline. Phase 3 is
not comfortably inside it. Landing at the cluster and missing gold is the most likely
outcome of an honest attempt; landing below the cluster is possible if phase 0 goes badly.
The floor is 0.752, already banked and reproducible.

## 7. What is deliberately not being done

- Forking the public notebook. It caps at bronze and teaches nothing.
- ILP / transformer linking. ≤ 0.015 available, measured.
- Further classical detector tuning. ~+0.06 available, ~250 places.
- Ultrack, or any third-party tracker. Same sliver as the ILP.
