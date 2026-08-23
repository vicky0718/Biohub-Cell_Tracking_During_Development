# The public pipeline runs — and its CV number cannot be trusted

`claude_cluster_probe` v4 + `claude_cluster_repro` v5, 2026-08-23.

| arm | SCORE | edge_J | node recall | s/dataset |
|---|---|---|---|---|
| pack, `det_threshold=0.99` | **0.9588** | 0.9566 | **0.996** | 30.6 |
| pack, `det_threshold=0.985` | 0.9572 | 0.9563 | 0.997 | 38.4 |
| our champion | 0.7070 | 0.7128 | 0.866 | ~30 |
| our best learned arm | 0.6490 | 0.6556 | 0.885 | — |

**Do not read 0.9588 as a CV estimate.** §2 explains why. What this run establishes is that
the pipeline *runs*, at a cost that fits a scored rerun.

---

## 1. It runs, and it is affordable

**34.5 s/dataset on a P100.** A 60-dataset CV pass is 0.58 h; a ~200-dataset scored rerun
is **1.92 h against a 12 h ceiling**. Runtime was the obvious way this could have been
impossible, and it isn't.

Their `predict_video()` returns coords `(N,4)` as `(t,z,y,x)` in original voxel space and
edges as `(src,tgt,prob,dist)` — exactly what our `build_graph` consumes — so their model
drops into our `Harness` with no file I/O and none of their splits machinery.

Three things had to be fixed to get there, each a real defect rather than a nuisance:

- **numpy split.** Their wheels upgrade numpy 2.0.2 → 2.4.6, and the notebook process has
  already imported 2.0.2's compiled extensions by then. In-process that is a hybrid and
  fails as `AttributeError: _blas_supports_fpe`. Everything now runs in a **subprocess**,
  which is how their CLI is meant to be invoked anyway. Installing from the pack's own
  wheel set rather than PyPI also matters: PyPI produced a different numpy mismatch
  (`_center`), and the pack's set is coherent by construction because its author ran it.
- **The official scorer was needed at all.** Their model **predicts divisions** — 564
  forking nodes in a single dataset — and our `purescore`'s division term is exact only for
  fork-free graphs, so `Harness` correctly refused to score it. Divisions are worth 0.1 of
  the 1.1 maximum and every arm we have ever run scored **0.000** on that term.
- **A latent bug in our own harness**, exposed by the first forking prediction ever passed
  through it. `score_graph`'s official branch converted to tracksdata twice — once for
  `evaluate`, again for `node_recall` — but `evaluate` writes its matching onto the graphs
  it is handed and `node_recall` reads it back, so the second pair had nothing to read.
  Fixed, and covered by `tests/test_official_scorer.py`. It had never fired because every
  previous arm was fork-free and took the `purescore` path.

## 2. 🚨 Why the score is not a CV number

Their model was trained on **some subset of the same 199 competition training datasets**.
`predict()` reads that membership from `data_dir/dataset_splits.json`, and:

- the file is **not in the pack**;
- it is **not in the competition data** (`train/`, `test/`, `sample_submission.csv` only);
- `train_unet_transformer.py` **reads** it rather than generating it, so there is no seed
  or rule to reconstruct.

So there is no way to know which datasets `split_0` held out, and the six scored here were
drawn at random from all 199. A model scores well on data it was fitted to. **0.9588 is
most likely a training-set score**, and the gap to the 0.913–0.916 the cluster actually
reports on the leaderboard is consistent with exactly that.

This is not a small caveat. It means **we cannot produce an honest cross-validation number
for this model at all**, on any subset of the competition's training data, because every
dataset in it is potentially contaminated.

## 3. What that does to the plan

The plan was: reach the cluster, then beat it by +0.02 with our per-dataset budget
calibration against their single global `det_threshold`. §2 does not kill that, but it
changes how it can be measured.

- **Absolute level: only the leaderboard can measure it.** One submission gives one honest
  number for the reproduction. That is the only clean measurement available.
- **The differentiator can still be measured as a *delta*.** Contamination inflates "their
  pipeline" and "their pipeline + our budget calibration" *equally*, so the difference
  between them on the same datasets is still informative. A negative delta is decisive; a
  positive delta is suggestive and needs the leaderboard to confirm. Stated limit: a model
  that has memorised a dataset may respond differently to a density change than it would on
  unseen data, so the delta is a screen, not a measurement.
- **Budget: ≤5 submissions/day.** Tuning a threshold against the leaderboard is not
  feasible at that rate, which is precisely why the delta screen is worth having.

## 4. What is confirmed about the differentiator

Read from their source, not inferred: density is controlled by `det_threshold` — a fixed
sigmoid probability — plus a `pool_kernel_um` max-pool NMS. **One constant for every dataset
in the corpus.** Their own default is 0.99; the public notebook ran 0.985. Across the two,
node recall barely moves (0.997 vs 0.996) while the score moves +0.0016, so the operating
point is flat where they sit.

We have a per-dataset budget regression at 10.7 % median error, reproduced exactly in four
separate runs. The asymmetry is real and it aims at the node-budget multiplier, which
`notes/16` §2.3 argued is the only term whose arithmetic reaches 0.935.

---

## What to do next

1. **Blocked on the Competition-Specific Rules.** §6.b permits external models "unless
   specifically prohibited by the Host" and §6.a is satisfied by a CC0 public dataset, but
   §7.a references a Competition-Specific Rules section that has not been read. Everything
   below assumes it permits this; if it does not, all of it is void.
2. **Submit the reproduction once.** It is the only honest measurement of the absolute
   level, and it converts an unmeasurable CV into a real number.
3. **Then screen the budget calibration as a delta** on fixed datasets, and submit it only
   if the delta is positive.

Runtime is not a risk (1.92 h of a 12 h ceiling). The banked floor remains **0.752**.

---

# Addendum, same day — both tracks measured

## Track A: the offline submission works

`claude_submit_pack`, internet OFF, on a P100:

```
numpy 2.4.6 | torch 2.5.1+cu121 | gpu_ok True
model 2,076,706 params, window_size=2, downsample=(1, 4, 4)
254,146 rows | 4 datasets | 135,083 nodes | 119,063 edges
no malformed-graph problems
```

Three blockers had to be cleared, and each was measured rather than assumed:

- **The free GPU is a P100 (sm_60)** and the image torch 2.10+cu128 builds sm_70+ only —
  confirmed directly as `CUDA error: no kernel image is available`. Also confirmed:
  `machineShape="nvidiaTeslaT4"` is **accepted by `kernels/push` and silently ignored**, so
  the accelerator cannot be selected through the API. Fixed by `claude_torch_wheelhouse`
  (23 wheels, 2.84 GB, torch 2.5.1+cu121), verified to resolve `--no-index` *before*
  anything depended on it.
- **A numpy split**: the pack's wheels upgrade 2.0.2 → 2.4.6 under a process that has
  already imported the old one. All work runs in a subprocess.
- **`write_submission` hardcoded `allow_divisions=False`**, which would have reported every
  one of the model's hundreds of divisions per dataset as an out-degree violation.

## Track B: training this architecture ourselves is not affordable

The batch-size search answered it in one run.

| batch | result |
|---|---|
| 4 | `CUDA error: invalid configuration argument` |
| 2 | same |
| **1** | **works — 1,014 s/epoch on 20 datasets** |

`_TemporalAttention` reshapes to `(B·S, T, C)`, so the attention batch is
batch-size × spatial-voxels; at B=2 on a 32³ stage that is 65,536, right at CUDA's 65,535
grid limit. Only B=1 fits on a P100.

Scaled to a full 128-dataset fold at **~6,492 s/epoch**:

| epochs | wall time | |
|---|---|---|
| 50 | **90 h** | exceeds a 12 h session |
| 100 | 180 h | |
| **402** (their checkpoint) | **725 h ≈ 30 days** | |

Kaggle gives roughly 30 GPU-hours per week and the deadline is ~5 weeks out, so 402 epochs
is ~24 weeks of quota. **Not reachable.** Their trainer's own message — *"For 2 GPUs set the
Kaggle accelerator to 'GPU T4 x2'"* — says this model was sized for more hardware than a
free P100.

T4 ×2 would help on two axes at once (sm_75 may lift the grid limit, allowing a larger
batch, and their data-parallel path uses both cards), plausibly 10–15× combined. That puts
50 epochs near 7 h — viable — but 402 still is not.

## What this reprioritises

Track B was justified by giving an honest CV and our own weights. At this cost it would
produce a model **worse** than the pack's (fewer epochs from scratch) for many GPU-hours.

Meanwhile the differentiator — our per-dataset budget calibration against their single
global `det_threshold` — needs **no retraining at all**. It applies to their weights
directly, and §3 above already establishes it can be screened as a delta.

So the order should be: submission first, **differentiator second**, and our own training
only if something specifically requires our own weights.
