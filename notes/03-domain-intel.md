# Domain intel — where the data comes from, and what the reference method does

Compiled 2026-08-14 from literature and source-code research. The Kaggle site itself is
unreachable from our environment (SPA + API 401), so **none of this comes from the
competition's own pages** — no discussion threads, no public notebooks, no leaderboard.

Confidence is tagged throughout, and the tags are load-bearing:

- **[PRIMARY]** — read directly from source code or a repo file.
- **[SEARCH]** — from a search engine's server-side summary of a page that could not be
  opened. Substance likely right, exact numbers carry paraphrase risk.
- **[INFERENCE]** — reasoning on top of the above. Not stated by any source.

---

## 1. Why the ground truth is sparse — and what that implies

Ultrack's Nature Methods paper introduces **dual-channel sparse labeling**: ubiquitous
fluorescence for imaging, plus *"sparse, random labeling at a distinct wavelength"* used to
generate ground truth. **[SEARCH]** — <https://www.nature.com/articles/s41592-025-02778-0>

Corroborated in code **[PRIMARY]**: the baseline's local data fallback is a directory
literally named **`./data/dense_channel`** (`scripts/dataspec.py`).

So: **we are given the dense channel; the ground truth was derived from a second, sparse
channel we never see.** That reframes the sparsity — it is not an annotation-budget
shortcut, it is a measurement modality. Two consequences we can and must test on train
(implemented in `notebooks/01_recon.ipynb` §5b):

1. **Clonal clustering.** A mosaic label is inherited by both daughters, so annotated cells
   should arrive in **clumps**, and **divisions should be over-represented** versus a
   uniform sample. **[INFERENCE]** If true, our nearest-neighbour linking statistics —
   computed among annotated cells only — understate how dense and confusable the real
   detector's neighbourhood is.
2. **Depth bias.** A cell only enters the GT if it was detectable in the *sparse* channel,
   which suffers the same depth attenuation as the dense one. Deep cells may be
   systematically missing. **[INFERENCE]** If true, local validation is **optimistic
   exactly where the imaging is worst**.

Divisions specifically were **human-proofread** in the Ultrack GUI **[SEARCH]** — consistent
with the metric scoring them separately.

Annotations are *"approximate cell centers"*, not segmentation masks. **[PRIMARY]**

## 2. Geometry: the anisotropy is exactly 4:1

`1.625 / 0.40625 = 4.000` exactly. **[INFERENCE, arithmetic]** So **downsampling XY by 4
yields a perfectly isotropic 1.625 µm grid** — which is precisely what the official baseline
does (`--downsample 1,4,4`, then `voxel_size = scale × downsample`). **[PRIMARY]**

### The Z error budget is the tight one

DaXi-class axial resolution is ~2 µm vs ~450 nm lateral **[SEARCH]**, and the Z step is
1.625 µm, so **a nucleus spans only ~4–6 Z-slices**. The 7 µm match cutoff is an *isotropic
physical* distance, so:

| error | in µm | fraction of the 7 µm budget |
|---|---|---|
| 2 Z-slices | 3.25 | **46 %** |
| 2 XY-pixels | 0.81 | 12 % |
| 8 XY-pixels | 3.25 | 46 % |

**Z centroid accuracy is worth roughly 4× XY accuracy**, and Z is the axis most likely to
carry *systematic* bias — e.g. a detector that snaps to the brightest slice rather than the
intensity centroid. **[INFERENCE]** Worth checking explicitly before tuning anything else.

## 3. Ultrack's ILP, exactly — read from source, not the paper

`ultrack/core/solve/solver/mip_solver.py`, `python-mip` over Gurobi or CBC. **[PRIMARY]**

**Variables** — per candidate node *i*: `x_i` selected, `a_i` appears, `d_i` disappears,
`v_i` divides. Per candidate link *e*: `y_e`.

**Objective (maximise):**

```
Σ_e φ(w_e)·y_e  +  Σ_i division_weight·v_i  +  Σ_i appear_weight·a_i
                +  Σ_i disappear_weight·d_i  +  Σ_i φ(p_i)·x_i
```

with link function **`φ(w) = w^η + b`, default `η = 4`** — raising IoU ∈ [0,1] to the 4th
power sharply suppresses marginal links. Easy detail to miss, large effect.

**Constraints**, per node *i*:

```
(1) Σ_{e ∈ in(i)} y_e + a_i == x_i                 # one parent only — forbids merges
(2) x_i + v_i == Σ_{e ∈ out(i)} y_e + d_i          # flow conservation; ≤2 children iff dividing
(3) x_i ≥ v_i                                      # divide only if selected
(4) x_p + x_q ≤ 1  for every ancestor/descendant   # nested hypotheses are mutually exclusive
    pair in the same watershed hierarchy
```

Constraint (4) is the whole idea: many nested segmentation hypotheses per cell, and the ILP
picks a consistent set **jointly with the linking**.

Appear/disappear penalties are **masked to zero** in the first/last frame and within
`image_border_size` — entering or leaving through the volume boundary is free.

**The lab's own zebrafish settings** (`examples/zebrahub/config.toml`) **[PRIMARY]**:

```toml
[linking]      max_distance = 5.0    max_neighbors = 5    distance_weight = 0.0
[segmentation] min_area = 500   max_area = 10_000   threshold = 0.5   ws_hierarchy = "area"
[tracking]     appear_weight = -0.002   disappear_weight = -0.01   division_weight = -0.001
               power = 4   window_size = 100   overlap_size = 5
```

`max_distance = 5.0` is **5 µm** (the call passes `scale=voxel_size`). Note
`disappear_weight` is 5× `appear_weight` — deliberately asymmetric, discouraging track
termination more than initiation.

**Flow fields.** `ultrack/imgproc/flow.py` estimates a multi-scale TV-regularised optical
flow and shifts candidates *before* the KD-tree query **[PRIMARY]** — subtract bulk tissue
deformation, then link on the residual. Directly transferable, and the way to shrink an
effective search radius.

**Stated failure mode** **[SEARCH]**: Ultrack tolerates *occasional* segmentation errors but
can converge on wrong solutions under **systematic errors that persist over time**.
**[INFERENCE]** That is exactly what depth-dependent SNR loss produces — a consistently
under-segmented deep region gets *consistently* mis-tracked, and the temporal-consistency
prior reinforces the error instead of correcting it.

## 4. Linking radius — set it from motion, not from the metric

| quantity | value |
|---|---|
| GT node-match cutoff | **7 µm** (= 4.31 Z-voxels = 17.2 XY-px) |
| Royer lab's own zebrafish linking `max_distance` | **5 µm** **[PRIMARY]** |
| Ultrack library default | 15 µm |
| baseline's min detection separation (`--pool-kernel-um`) | ~3–5 µm **[PRIMARY]** |
| nucleus diameter | **~6–17 µm, centre ~10 µm** **[INFERENCE]** from Ultrack's `min_area`/`max_area` as spheres |
| cell speed, somitogenesis PSM | ~0.83 µm/min **[SEARCH]** |
| cell speed, peak epiboly | up to ~3.3 µm/min **[SEARCH]** |
| frame interval | **UNKNOWN** — 30 s in the Ultrack paper's *different* acquisition **[SEARCH]** |

**[INFERENCE]** At a 30 s interval, per-frame displacement is ~0.4–1.7 µm, so **7 µm is a
generous tolerance — 4–17× the expected motion.** It means "matched to the right cell", not
"matched to the right position". Setting the *linking* radius to 7 µm to "match the metric"
is a category error: it buys ambiguity, not recall. The binding constraint is **nucleus
spacing (~6–10 µm), not speed.**

**The frame interval is the single most consequential unknown**, and everything above scales
linearly with it. `01_recon.ipynb` §5c infers it from the GT displacement distribution and
cross-checks it against the observed division rate.

## 5. Competing methods worth knowing

- **Ultrack** — same lab, built for dense 3D embryos, has an explicit anisotropy knob.
  Usable as a library (BSD-3, `pip install ultrack`). CBC solver works but is slow; Gurobi
  needs a licence, so **CBC is the realistic option in a Kaggle kernel**. **[PRIMARY]**
- **Trackastra** — transformer over a temporal window, self+cross attention, **explicitly
  models division**, greedy linking. Architecturally close to the official baseline.
  3D performance not documented in reachable sources. <https://arxiv.org/abs/2405.15700>
- **ELEPHANT** — deep learning **on sparse annotations** for light-sheet embryos.
  Conceptually the closest match to our supervision regime, though it wants ellipsoid
  pseudo-segmentations rather than points. <https://elifesciences.org/articles/69380>
- **Fluo-N3DH-CE** (*C. elegans*, Cell Tracking Challenge) — the closest public analogue:
  real 3D, anisotropic, high cell density. Best place to look for transferable tricks.

**[INFERENCE]** The strongest prior on what wins here is the baseline's own shape — learned
detection + learned pairwise association — because it trains *through* the sparse
supervision, whereas Ultrack's ILP is unsupervised w.r.t. our GT and would need its
hyperparameters tuned against sparse labels rather than learned. The obvious synthesis is a
hybrid: **learned affinities as ILP edge weights**, which Ultrack's `nodes_prob` / edge-weight
interface already supports.

## 6. Other facts

- **$60,000 prize pool**; launched 29 June 2026. **[SEARCH]**
- Real dataset name seen in a repo issue: `2024_03_22_dorado_0002_0198_0184_0605`.
  **[INFERENCE]** the trailing triplet looks like a crop origin or size, implying the
  released videos are **sub-volume crops** of larger acquisitions — so cells cross the
  boundary, and appearance/disappearance there is an artifact, not biology. Tested in
  `01_recon.ipynb` §5b.4. The acquisition date 2024-03-22 is solid and post-dates Zebrahub.
- Zarr attrs carry `image_statistics.quantiles`, so normalisation needs no data pass. **[PRIMARY]**
- **[INFERENCE]** "Early development" most likely means **somitogenesis (~10–24 hpf), not
  gastrulation** — the lab's mounting protocols specify tricaine "after 15 hpf". Matters
  because it implies weaker coherent flow and more diffusive motion.

## 7. Still unknown — and only Kaggle can answer

Number of videos (train/test), timepoints per video, **frame interval**, imaging duration,
developmental stage, cell counts over time, volume dimensions, what fraction of cells are
annotated, and the entire competitive picture (leaderboard spread, top score, what the
public notebooks do). **No dedicated dataset paper appears to exist** — the data went
straight to Kaggle without a preprint.

Most of the data-side gaps are answered by running `01_recon.ipynb`. The competitive
picture needs Kaggle access.
