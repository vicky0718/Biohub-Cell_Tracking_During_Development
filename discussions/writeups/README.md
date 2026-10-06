# Biohub — Cell Tracking During Development: how the top 25 did it

These are the solution writeups of the teams that finished in the top 25 of the private
leaderboard. 17 of the 25 teams published one. They were scraped on 2026-10-06, each with
its comment thread and its images.

- [`index.md`](index.md) lists every top-25 team with its scores and a link to its writeup.
- `raw/` holds the untouched API responses.
- `raw/leaderboard_private.csv` has every team's private and public rank and score.

This page is the digest: what the 17 writeups say, across teams, with the numbers they report.
It cites each team by its private rank ("4th", "12th"), and every figure comes from that team's
own writeup.

`python discussions/scrape_writeups.py` regenerates the archive in place. It writes `index.md`
and leaves this page alone.

**The task.**
- **Metric:** `adjusted edge Jaccard + 0.1 × division Jaccard`. Nodes are matched within 7 µm.
  The edge term is scaled by `1 − 0.1·(N_pred − N_est)/N_est`.
- **Training data:** 199 movies from two embryos, `44b6` and `6bba`. About 2.8% of nuclei are
  annotated, and there are only 151 annotated divisions.
- **Test data:** two new embryos. Public is `fdad` (60 movies); private is `ea36` (106 movies).
  According to 3rd place's probing, the private embryo is sparser than either training embryo.

**Where we finished, for reference.**
- Our selected submission `rd07`, a fork of the public x138 lineage, scored 0.9539 public and
  0.9178 private: rank **1010 of 3,947**.
- **325 teams have exactly 0.91780 private** (ranks 689–1013). That is the cohort of teams that
  submitted, in effect, the same predictions.

---

## The short version

Five things separate the top 25 from the 325-team block at 0.9178:

1. **They won the division term.** The public pipelines' ILP could never form a division. Their
   divisions came only from geometric grafts, and they recovered about 23 of 151. Most top
   teams put their largest single effort there and got their largest single gain from it.
2. **They trained their own models**, or added their own models to the public ones. The public
   checkpoints had seen every training video. Models that hadn't were worth up to several times
   more on the private embryo than the public board showed.
3. **They validated the way the test differs.** Usually that meant across embryos. Those who
   couldn't said so, and stopped trusting their bench.
4. **They read the public board as what it was: one embryo.** Large differences there were real;
   differences of 0.001–0.005 between near-equal variants were noise or worse.
5. **They fixed the unglamorous details.** Integer coordinates, rounding, the per-embryo z offset,
   smoothing along tracks, stage drift, duplicated frames, and the node count.

---

## 1. Divisions were the lever

**The public pipelines could not divide.** With the public ILP weights (appearance 0, division
1.2, edge −1), a second outgoing edge gains 1.0 and costs 1.2, so the ILP never forms a fork.
In those pipelines every division came from a geometric "safe division" graft afterwards.
Four teams found this independently: 9th, 12th, 14th and 23rd. We had found it too
(Biohub `notes/78`–`79`). 18th measured the inherited pipeline on all 199 training videos:
**23 of 151 divisions recovered** (TP/FP/FN 23/102/128).

What fixing it was worth:

| Team | Change | Gain |
|---|---|---|
| 1st | a per-cell "Soon Net" (is this cell about to divide, where was it, where does it go) | public ~0.925 → ~0.963 "with improvements in divisions alone" |
| 3rd | division models + calibration + joint link/division LP | CV +0.063 (0.902 → 0.964) |
| 4th | a learned per-node division cost inside the ILP, `6 − 16·s` | **+0.035 on public and private** (0.913 → 0.948), their largest single gain |
| 5th | "new cell" and division heads on the linker | +0.020 public, **+0.024 private** |
| 7th | division costs, then daughter-pair models + two-stage ILP | private 0.910 → 0.937, then 0.942 → 0.956 |
| 9th | DIVCARRY: re-solve the ILP with fork-friendly costs, add back the missing daughters | +0.011 public, +0.013 private |
| 23rd | a learned second-daughter classifier on top of x138 | +0.007 public, +0.010 private |

What the division models looked like:
- **An image model of the mother.** 1st (3-frame crops, CNN + transformer); 4th (5-frame CNN);
  7th (3D CNN + BiGRU on motion-aligned difference images); 10th (ConvNeXt + transformer);
  23rd (small CNN).
- **A separate model to pick the two daughters.** 4th (pair model on 174 image features);
  7th (ExtraTrees on 73 geometric features); 17th (triple classifier); 18th (CatBoost on 303
  features).
- **Reframing instead of classifying.** 14th regressed, for every voxel, where its parent was;
  a division is two instances converging in parent space while staying apart in image space.
  18th used external divisions (Linajea, 70 events) and a "time direction" model, which asks
  whether a clip is played forwards or backwards.

Lessons the teams drew:
- **Labels.** Labels from annotated tracks only. 4th: hard-negative mining "hurt badly",
  because the hardest negatives were real, unannotated divisions.
- **Value of one division.** About 74 edges (12th); +0.0005 to +0.0011 per division (17th,
  23rd). Removing a false fork is worth far less than adding a true one (17th, 23rd).
- **Judge divisions on the finished graph.** 18th re-checks every division after daughter-track
  completion, on the graph it actually submits.

## 2. Own models transferred; public checkpoints were in-sample

The public detection and linking checkpoints were trained on all 199 training videos, so every
offline score built on them was in-sample.

**What happened to teams that built on them:**
- **12th:** "Our full-chain bench was in-sample and over-read the private by about 0.03." Their
  first lesson is "Own your base models, or build the OOF world on day 1."
- **23rd:** a late leave-one-embryo-out check moved two of their numbers sharply. Their
  classifier's local gain went from +0.006 to +0.0004. Flip-only TTA went from +0.020 to
  −0.001.
- **9th:** their own detectors and linker, blended 1:1 into x138, were worth **+0.004 public but
  +0.023 private**. "If public weights were trained on all of train, train-based CV over-rates
  them."

**Teams that trained their own models climbed hardest on private:**
- **10th** used their own detectors, linker and division gate, validated leave-one-embryo-out.
  Their selected submission was 1838th on public (their best public was 191st), and finished
  **10th**.
- **11th** trained one end-to-end model (detection, motion and division heads), with embryo
  transfer checks and ILP costs tuned after training. Public 1087th, private **11th**, a
  public–private gap of 0.0016.
- **20th** used one detector of their own and no public checkpoints at all. Public 1067th,
  private **20th**. Their rule of thumb: "parts that learned how images look do not transfer
  across embryos, while parts based on geometry transfer reasonably well."

**Most of the top 7 trained their own detectors.**
- 1st, 2nd, 3rd, 4th, 6th and 7th trained their own detectors.
- 5th retrained the baseline architecture with ZebraHub pretraining.

The public-lineage teams that placed still added their own models:
- **9th:** six detectors and a linker.
- **12th:** a mitosis specialist, an identity link and learned recentring.
- **16th:** self-trained detectors under the public tracking stack.
- **18th:** about ten learned heads.
- **23rd:** a division classifier stack.

A nuance worth keeping: out-of-fold stacking can *under*-value strong models. 4th found
their full-data transformer worth +0.005 on the board while its fold versions looked worthless
locally. 14th: "Out-of-fold can understate a shipped model."

## 3. Validation

**Cross-embryo validation:**
- 10th: 2-fold leave-one-embryo-out, macro-averaged.
- 17th: every learned component trained on one embryo and scored on the other.
- 18th: embryo-wise holdout, plus "decide on all 199 videos".
- 4th: cross-embryo checks for detector changes, because "same-embryo validation over-rated
  changes that fit the seen embryos better".
- 11th: compared the training embryo against the held-out one when choosing architectures,
  synthetic data and augmentation.

**Movie-level 5-fold CV with full OOF:** 1st, 2nd, 3rd and 5th. 5th stratified by embryo and by
the number of divisions per clip. All four stated that this mixes embryos. 3rd's OOF CV
over-read private by 0.011, which was still a stable enough gap to steer by.

Discipline that held up:
- **Pre-registration.** 17th designed each change on half the movies, froze it with a pass rule,
  then confirmed it once on the other half. **7 of 19 design-half wins were rejected**, and
  hand-tuned rules kept a median of about 40% of their design gain. 12th and 23rd also
  pre-registered their decision rules.
- **Score everything on all 199 videos.** 18th: gains of +0.005 to +0.012 on 8–16-video checks
  "repeatedly turned into −0.001 to −0.003 on all 199 videos".
- **One ablation ladder on one fixed OOF world**, scored by the official metric. 3rd did this;
  12th names it as a lesson.
- **Label post-hoc edits by their effect on the metric** (17th), not by a semantic label. Their
  duplicate-merge model with "is a duplicate" labels lost score; the same features with
  metric-effect labels gained +0.0099 on confirmation.
- **Every number names its ruler** (12th): in-sample, OOF, public or private.

## 4. Public vs private: one embryo each

From `raw/leaderboard_private.csv`:
- **The private top 25 came from everywhere on public.** Their public ranks run from 1st to
  1,087th, with a median of 30th.
- **The public top 10 held up best.** 7 of them finished in the private top 25, the worst 52nd.
- **The public top 25 held up less well.** 11 of them finished in the private top 25 (median
  private rank 33rd, worst 354th).
- **Across 2,446 teams with public > 0.90 and private > 0.85**, public and private correlate at
  r = 0.88. The board is right about big differences.
- **Of the 1,040 teams at public ≥ 0.953**, the median private score is 0.918, and two-thirds
  finished at 0.920 or below.

Where it is wrong is small differences between near-equal variants:
- **18th:** among 34 late submissions differing by ≤ 0.005, public and private correlated at
  r = 0.09. CV and private correlated at r = 0.34.
- **12th:** 17 of 104 parent–child submission pairs moved in opposite directions on the two
  boards.
- **16th:** changing only the random seed moved public by 0.006 and private by 0.001, "as large
  as many of the improvements I had been chasing".
- **7th:** synthetic pretraining took public from 0.960 to 0.966 and private from 0.956 to
  **0.947**.
- **2nd:** a stronger synthetic detector gained +0.0013 CV but lost 0.007 public.
- **9th:** the public-model-only x138 variants lost 0.037–0.039 from public to private. Their
  own-model submissions lost a median of 0.013.

Final selection, as it played out:
- **The best private submission was often not selected.** 4th's best private (0.964) tied on
  public and was passed over. The same happened to 6th's 0.955 "best" checkpoints, 16th's
  MAE-pretrained 0.946, and 20th's two-seed average, which was dropped over −0.003 public.
- **Hedges that worked.**
  - 23rd put a one-z-plane shift of every node in the second slot. It lost 0.003 public, gained
    0.005 private, and was their best submission. They chose it with an explicit model of rank
    uncertainty.
  - 9th paired its best submission with a different post-processing family. They estimated the
    selection noise by bootstrapping 29%/71% splits of their holdout.

## 5. Sparse labels, pseudo-labels and external data

**Nobody treated unannotated voxels as background.**
- 3rd masked 6 µm around low-threshold DoG candidates.
- 4th, 6th and 11th used three-tier losses: annotated, verified background, unknown.
- 6th added a count prior that pulls the summed heatmap toward the host's per-movie
  cell-count estimate.
- 9th tried background-as-negative: +0.002 public, **−0.019 private**.

**Pseudo-labels, as several rounds of training a new model on the previous one's labels:**
- **1st:** 1,800 hand-annotated cells, plus pseudo-labels filtered by track length.
- **4th:** used the pseudo-label model's nodes only where the ground-truth model agrees
  (+0.004).
- **6th:** used the whole pipeline (ILP topology, raw positions) as teacher, giving 5 M pseudo
  cells; the first round gained +0.008.
- **10th:** three rounds.
- **16th:** self-training from its own tracking output.

**External data and pretraining:**
- 5th: ZebraHub pretraining, +0.016 private.
- 10th: a DSB-2018 nuclei encoder.
- 12th and 18th: Linajea divisions.
- 11th and 16th: FOCUS-3D, as synthetic statistics and background references respectively.
- 1st: MAE pretraining.
- 16th: Kinetics-pretrained 3D encoders.

**Synthetic data cut both ways.**
- It helped 2nd (synthetic faint cells), 10th and 11th, who trained on mixes and then fine-tuned
  on real data.
- It hurt 7th on private (section 4).

## 6. Linking and global optimisation

- **Motion before matching.**
  - 3rd: a dense flow model.
  - 7th: backward displacement predicted by the detector.
  - 11th: a Linajea-style motion head.
  - 6th, 10th and 14th: drift-corrected distances. 6th's was worth +0.009 on validation:
    "identity swaps concentrate on" stage-drift frames.
- **Learned linkers.**
  - 1st: a LightGlue-style transformer, with a "fork head" that decides whether a cell divides
    separately from which daughters.
  - 5th: a transformer with link, identity, new-cell and division heads.
  - 6th: sparse graph attention with an explicit "new cell" class.
  - 10th: a HOCT-style edge transformer on self-supervised DINO embeddings.
  - 4th, 17th, 18th and 20th: gradient-boosted trees on geometric features.
- **One global solve.**
  - 3rd: an LP/MILP over a linearised surrogate of the metric, with "probability of being
    evaluated × probability correct" to handle sparse labels.
  - 6th: a conservation-tracking ILP with every cost taken from network probabilities, so "no
    repair heuristics are needed". LP-first with HiGHS ran 38–166× faster than SCIP.
  - 7th: a two-stage ILP that re-solves globally after re-scoring divisions.
  - 11th: ILP costs tuned to the metric after training ("my score started improving as soon as
    I began tuning the ILP costs").
  - 1st (greedy decode on learned scores), 17th (augmented LAP) and 18th (capacitated bipartite
    matching) did well without a global ILP.

## 7. Details that moved the score

| Detail | Evidence |
|---|---|
| Integer, rounded coordinates | 9th: an int16 cast truncated instead of rounding, and fixing it gave +0.009 public / +0.013 private. 12th: integers 0.950 vs floats 0.942 public |
| Per-embryo annotation z offset | 6bba ground truth sits about +0.675 plane above nucleus centres, 44b6 about +0.115 (12th). 23rd's z+1 hedge gave +0.005 private. 14th's constant shift after all graph decisions gave +0.002 |
| Coordinate refinement | 3rd: affine motion refinement, +0.006 CV. 12th: learned recentring, +0.003 private. 18th: per-axis LightGBM correction |
| Line-fit smoothing along tracks | 14th: −0.019 when off, their largest mechanism. 9th: −0.005 / −0.007 off. 6th: drift-compensated smoothing, +0.005 |
| Duplicated frames and stage jumps | 947 byte-identical frames in 114 6bba videos (12th). After a duplicate, the edge false-negative rate was 8% vs 2.9% (23rd). Jump registration: 17th, 23rd |
| Node count | 4th: changes that alter the node count "transferred worst". 18th: deleting 10% of tracks cost −0.015 public and **−0.06 private**. 17th: nothing that added or removed nodes survived validation |
| TTA | 6th: going from 1 to 4 views gave +0.05 on a single model. 7th: 8-view detection TTA beat adding detectors, but averaging *motion* vectors hurt |
| Runtime and safety | 6th: a deadline guard with graded fallbacks, so "every movie gets a prediction". 12th: a valid CSV written at t+0, sha256 pins, fail-open stages, and submitting exactly the verified version (an earlier mismatch had scored 0.000). 9th: a local runner that reproduces the Kaggle output byte for byte |

## 8. AI coding agents in the top 25

Four writeups describe agents doing most of the implementation:
- **10th, "Grandmaster Powered Agentic Approach":** "The work was split between Claude, Codex,
  our Kaggle Agent, all steered by our Kaggling experience." They went from 1838th public on
  their selected submission to 10th private.
- **11th:** "AI coding agents, keeping relevant papers, the official implementation, and
  discussion copies in the repository for them to consult."
- **12th:** a Claude Code main agent ("ATHENA") with skills and permission limits, over 4,200
  sub-agent runs, adversarial verifiers and a judge. The human set direction and approved every
  submission. Their own verdict: validation was their largest error.
- **23rd:** "Most implementation and measurement was done with AI coding agents, Claude Code and
  Codex, under pre-registered decision rules."

## 9. The public lineage we were on, and how far it went

| Team | Started from | Added | Private |
|---|---|---|---|
| us (`rd07`) | x138 lineage + V1284 head | flow / gapfill / readmit keys from a public fork | 0.918 (1010th) |
| 23rd | x138 (0.953 / 0.917) | mitosis gate, jump relink, daughter veto, learned second-daughter classifier, z+1 hedge | 0.939 |
| 18th | Harmonic Fusion v30 (detection and candidate graph) | everything after: LightGBM edges, matching, coordinate correction, a division model stack, daughter completion, re-check | 0.942 |
| 16th | public tracking stack | own self-trained detectors (ResEnc-L, ResNet3D-18), edge head fine-tuned | 0.943 |
| 12th | pilkwang's public models + prvsiyan's chain | consolidation, mitosis specialist, identity link, learned recentring, safe writer | 0.946 |
| 9th | x138 | six own detectors + own linker (+0.023 private), DIVCARRY (+0.013) | 0.949 |

Even without leaving the public lineage, the teams that rebuilt the division stage or brought
their own models gained 0.02–0.05 on private over the public chain they started from. That was
the gap between 1010th and the top 25.

---

## 10. Team by team

Ranks are private / public. Each heading links to the full writeup.

**[1st — Sergio Alvarez](01-1st-place-solution.md)** (0.978 / 3rd, solo)
- **Detector:** a 3D U-Net with a Cellpose-style flow head and light MAE pretraining. It was
  fine-tuned on 1,800 cells he annotated by hand plus stamps on the ground-truth nodes, then on
  pseudo-labels.
- **Soon Net:** 3-frame crops around each cell go through a small CNN tokenizer, a transformer
  and occupancy maps. It predicts whether the cell is about to divide or has just divided,
  where it was, and where it goes. It mined 515 divisions from the 151 annotated.
- **Linker:** LightGlue-style, with a pair head and a "kind × who" fork head, followed by a
  greedy decode.
- **CV:** 5-fold. 0.9575 on the original ground truth.
- **Approach:** looked at the cells for dozens of hours, and turned "blind" hand-made features
  into ones a model reads from the image.

**[2nd — Soheil Ayati](02-2nd-place-solution.md)** (0.970 / 28th, solo)
- **Detection:** a residual 3D U-Net on isotropic 64³ volumes over 3–5 frames, predicting a
  heatmap, offsets, backward motion and descriptors.
- **Training data:** synthetic faint cells (real tracks dimmed and pasted within the same movie).
- **Tracking:** a learned count estimator, LightGBM specialists, link and division scorers, and
  auxiliary detector banks used as evidence.
- **Recovery:** faint tracks are recovered from unused candidates, checked against the image,
  and only in the "acq_01" image-condition group (found by K-means clustering).
- **Practice:** inspected raw edge and division counts, not just the score.

**[3rd — yu4u](03-3rd-place-solution.md)** (0.967 / 1st)
- **Detection:** 2.5D EfficientNet U-Nets (reused from their CZII solution) plus a 3D SegResNet,
  trained with DoG-masked sparse supervision.
- **Linking:** a dense flow model and a learned matcher.
- **Divisions:** parent, pre and post division models on raw and flow-aligned frames.
- **Optimisation:** calibration with "probability of being evaluated", then an LP/MILP over a
  surrogate of the metric.
- **Post-processing:** gap closing, short-component removal, affine coordinate refinement.
- **Scores:** OOF CV 0.978. Probing identified the test embryos and their densities.

**[4th — Barry](04-4th-place-solution.md)** (0.962 / 20th)
- **"Model the biology" on modest compute.** The largest network has 18 M parameters, and
  gradient-boosted trees on CPU combine the evidence.
- **Detection:** a Cellpose-style 3D net with tiered sparse loss. A DoG density router sets the
  threshold per movie, and pseudo-label nodes are kept only where an independent model agrees.
- **Divisions:** a per-node division cost in the ILP (+0.035) and a fork verifier that traces
  both daughters in the raw images.
- **Lessons:** validate the way the test differs; be careful with changes to node count.

**[5th — Tang](05-5th-place-3d-u-net-transformer-linker-multi-s.md)** (0.955 / 8th)
- **Models:** the baseline temporal 3D U-Net, plus a transformer linker with link, identity,
  new-cell and division heads. ZebraHub pretraining, a 5-fold ensemble and 8-view TTA.
- **Tracking:** a multi-stage ILP. It solves with confident cells first, then more cells, keeps
  only the first solve's divisions, then re-links and repairs.
- **Gains:** a gain table for every component, on both boards.
- **Data quirks:** found duplicated and drift frames in training.

**[6th — Cyrus](06-6th-place-solution.md)** (0.954 / 94th)
- **One network detects and links**, at native resolution with temporal fusion. It outputs
  probabilities a conservation ILP uses directly, including an explicit "new cell" class.
- **Pseudo-labels:** whole-pipeline teacher → student rounds.
- **Ensemble:** six models on different splits, each with its own LightGBM re-scorer.
- **Drift:** compensated in the solver, the smoother and the augmentation.
- **Runtime:** engineered to fit 12 hours on two T4s. Code and models are public.

**[7th — tatsutaka](07-7th-place-solution.md)** (0.953 / 7th)
- **Detection:** a single nnU-Net ResEncM predicting heatmap and backward motion, with
  frame-interval augmentation to handle freezes.
- **Divisions:** a dedicated pipeline. A CNN + BiGRU reads motion-aligned difference images, an
  ExtraTrees model ranks daughter pairs, and an MLP checks the division against an
  ordinary-continuation explanation. A two-stage ILP follows.
- **Public vs private:** a period-by-period table. Early division work gained on both boards;
  later public gains (synthetic pretraining) did not transfer.

**[9th — yuto083](09-9th-place-solution-own-detectors-public-tracker.md)** (0.950 / 30th, solo)
- **Base:** x138 (our lineage), plus six of their own sub-voxel detectors and their own linker.
- **Fixes:** the coordinate rounding fix, and DIVCARRY.
- **Hedge:** a different family in the second slot, sized by a bootstrap of the selection noise.
- **Lessons:** audit the writer; count divisions at every stage.

**[10th — Kaggle Gentlemen](10-10th-place-solution-grandmaster-powered-agentic-a.md)** (0.949 / 191st best, 1838th selected)
- **Team and setup:** five Grandmasters working with Claude, Codex and a Kaggle agent.
  Leave-one-embryo-out CV.
- **Detection:** two 3D detectors (synthetic, then real, plus 3 pseudo-label rounds), and a
  2.5D detector from a DSB-2018 nuclei encoder that adds missing tracks.
- **Linking:** self-supervised DINO appearance embeddings and a HOCT-style edge transformer.
- **Divisions:** a ConvNeXt classifier that gates and recovers them.

**[11th — tanbo](11-11th-place-solution.md)** (0.947 / 1087th, solo)
- **Model:** one two-frame temporal U-Net with detection, Linajea-style motion and division
  heads, trained end to end on dense features.
- **Training data:** synthetic videos built from FOCUS-3D statistics, then real-only
  fine-tuning.
- **Tuning:** ILP costs and thresholds tuned to the metric on a calibration split.
- **Agents:** AI coding agents with papers kept in the repository.

**[12th — Corwin](12-12th-place-solution.md)** (0.947 / 14th)
- **Base:** pilkwang's public models, tuned but not retrained, plus their own stages:
  consolidation, mitosis specialist, identity link, learned recentring and a safe writer.
- **The most detailed data audit in the set:** copied frames, frame jumps, overlapping crops,
  the z convention, and an exact decomposition of the metric's deficit.
- **Agent:** Claude Code. Validation was their biggest error, and they say so.

**[14th — Vibes & Edges Trade-Off](14-14th-place-solution-from-vibes-and-edges-trade-off.md)** (0.945 / 43rd)
- **Two pipelines, grafted per video** with a circuit-breaker fallback. One is FlowSeg (a
  Cellpose-style field) under a consensus and veto detector scheme; the other is a dual-seed
  TemporalUNet3D.
- **Linking:** an ILP and a GNN linker, gated on residual motion.
- **Divisions:** "divflow" regresses parent positions, so a division is geometry, not a
  classifier. A red–blue–red intensity prior and a ViT add missed ones.
- **Smoothing:** line-fit smoothing was worth 0.019.

**[16th — r3takahashi](16-16th-place-solution.md)** (0.943 / 75th, first Kaggle competition)
- **Detection:** self-trained ResEnc-L and ResNet3D-18 detectors, Gold + Silver rounds from
  their own tracking output, on the public tracking stack.
- **Resolution:** 2× instead of 4× in-plane pooling, +0.004 private.
- **Analysis:** about 14% of annotated cells matched a detection 2.5 µm or more away, at a
  similar rate in every model tested. Two-thirds of those large offsets were shared by all four
  models: localisation headroom no ensemble removed.

**[17th — ibyyue](17-17th-place-solution.md)** (0.942 / 40th, solo)
- **Built from scratch** on competition data only.
- **Detection:** a 3D U-Net pair with adaptive scale for large nuclei.
- **Linking:** a gradient-boosted linker solved as an augmented LAP.
- **Divisions:** a triple classifier plus "sister valley" rules.
- **Graph edits:** a chain of metric-labelled edits.
- **Validation:** cross-embryo CV and a design/confirmation protocol. 23% of CV gain transferred.

**[18th — ymg_aq](18-18th-place-solution-lineage-graph-refinement.md)** (0.942 / 9th)
- **Base:** kept Harmonic Fusion v30's detection and candidates, and rebuilt everything after.
- **Edges and coordinates:** a 341-feature LightGBM, then matching and coordinate correction.
- **Divisions:** a CatBoost triplet model, an external-data division CNN, a time-direction
  model, and TabPFN stacking. Divisions are adopted against competing edges and re-checked on
  the finished graph.
- **Principles:** never delete nodes; decide on all 199 videos.
- **CV vs public:** CV predicted their private better than public did.

**[20th — Yurinchi](20-20th-place-solution.md)** (0.941 / 1067th, solo, first image competition)
- **Pipeline:** one own detector (the host network retrained), a learned coordinate
  refinement, and a GBM linker on 14 geometric features with a global ILP.
- **No public checkpoints.** They took ideas from public notebooks, never their weights.
- **Result:** a public-to-private drop held to about 0.010, and a 1,047-place climb.

**[23rd — taiseiu](23-23rd-place-solution-division-recovery-on-a-public.md)** (0.939 / 4th)
- **Base:** x138 kept frozen.
- **Additions:** a mitosis gate, drift-aware relinking, a daughter veto and a learned
  second-daughter classifier. Together they took x138 from 0.917 to 0.935 private, and the z+1
  hedge reached 0.939.
- **Validation caveat:** they audited the checkpoints and found them in-sample, then ran a late
  leave-one-embryo-out check that cut several local gains.
- **Agents:** Claude Code and Codex under pre-registered rules.

## 11. No writeup

8th (Amin), 13th (The Boys), 15th (Yapay Hücre), 19th (slime), 21st (Lime1123),
22nd (3D Research), 24th (0.948+ pls) and 25th (zhuo wamg + AibePC) had published no solution
writeup, forum post or public competition notebook as of 2026-10-06. The only exception: the
members of 21st had posted four questions in the forum (see [`index.md`](index.md)).
