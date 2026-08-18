# Competitor intel — what the public notebooks say

Gathered 2026-08-16. **Sources: five public Kaggle notebooks**, pulled as source via
`https://www.kaggle.com/kernels/scriptcontent/{scriptVersionId}/download`, which serves
public notebook source without authentication. The version id is in the `oembed` link in
each `/code/<user>/<slug>` page's HTML head.

The discussion forum itself is still unreachable: `api/v1` returns `401 Unauthenticated`,
the web pages are JavaScript shells with no server-rendered content, and `forum.image.sc`
is blocked by this environment's egress proxy. So none of this comes from the discussion
threads — it comes from what competitors published in notebooks.

| notebook | what it is |
|---|---|
| `anhadmahajan06/biohub-track-your-cells-development` | fork of the learned pipeline, reports **public LB 0.915** |
| `pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery` | the learned pipeline itself — UNet + transformer + ILP + repair |
| `pilkwang/biohub-cell-tracking-data-model-eda-baseline` | its EDA/model-description sibling |
| `xiaoleilian/biohub-cell-tracking-classical-baseline` | a classical pipeline very close to ours, with LB numbers |
| `dalloliogm/biohub-suspicious-tracking-event-review` | a submission-auditing tool, no model |

---

## 1. ⭐ The scoreboard we have been missing

This is the first time we have known what any score means.

| approach | reported score |
|---|---|
| official **nearest-neighbour benchmark** | **0.143** |
| classical pipelines — "the public cluster" | **~0.6** |
| a classical config the author names | **0.720** |
| same, with a heavier DoG detector at ~700 cells/frame | **0.59–0.66** (worse) |
| learned: TemporalUNet3D + transformer + ILP + repair | **0.915** |
| same, with divisions accidentally zeroed | 0.906 |

Our best measured config scores **0.5552 on 199 train datasets** — not leaderboard, and
not comparable directly. But it places us in the classical cluster, well above the
nearest-neighbour benchmark and below the classical ceiling of ~0.72.

**The frontier is a learned detector.** No classical pipeline in the public set exceeds
~0.72, and the learned one reaches 0.915. Recon §7's ceiling of ~0.99 with perfect
detections says the whole gap is detection quality.

## 2. ❗ A correction to `notes/05-first-sweep.md` §1

`notes/05` said the official baseline's `--det-threshold 0.99` "costs about 0.48 of
score". **That comparison was invalid and the claim is withdrawn.**

The two thresholds are not the same variable:

- **Ours** is a cut on the *quantile-normalised image intensity*. At 0.99 almost nothing
  survives — hence 0.049.
- **The baseline's** is a cut on the *TemporalUNet3D's predicted centre probability*. The
  competitors run it at **0.96875** and **0.985** and score 0.9+.

A high threshold on a calibrated probability map is a completely different operation from
a high threshold on raw intensity. What `notes/05` actually measured is that **our own
detector's threshold belongs near 0.15**, which remains true and is still the largest
single gain we have. The claim about the organisers' default was a category error.

## 3. ✅ Over-detection *is* penalised — and it brackets our budget-cap result

`xiaoleilian` reports directly: swapping in a Difference-of-Gaussians detector producing
**~700 cells/frame dropped the leaderboard from 0.720 to 0.59–0.66**, and concludes "the
host metric penalises over-detection". Their recommendation is to hold **≈200–320
detections per frame**, the density that scored 0.720.

This does **not** contradict `notes/05` §2, where our per-dataset budget cap measured
worse. It brackets it. Our threshold-0.15 detector emits **~210 detections/frame** and
runs at a pooled budget ratio of **−0.111** — inside their recommended band and *under*
budget, which is exactly the regime where a cap can only cut real detections. Their 700/
frame is far over budget, which is where the multiplier starts charging.

Both results together: **the node budget has a real optimum, we are already sitting near
it, and the danger is on the high side, not the low side.** Any change that pushes
detection counts up has to re-check the ratio.

## 4. Techniques worth stealing, in order of expected value

> **⚠️ Superseded in part by `notes/08` §2 (2026-08-16).** The rule-based author's
> published ladder measures **gap closing as roughly neutral** (CV +0.016, LB −0.002) and
> **division edges as harmful** (LB 0.784 → 0.778). Item 1 below is downgraded from "the
> highest-value repair available" to "implemented, off, low expected value". Item 6 on
> divisions is downgraded further. The ranking that survives is: **detector first**.

1. **Gap closing by inserting a synthetic node.** Both learned notebooks do it, with
   separate one-frame and two-frame passes (`gap_close_max_added_frac ≈ 0.045`,
   `gap2_max_links_frac ≈ 0.0026–0.0045`). This is the direct answer to our
   missed-detection problem: the scorer *drops* a `t → t+2` edge, but a synthetic node at
   `t+1` turns one unusable bridge into **two scoring edges**. With 15.5 % of GT nodes
   missing and edge recall going as recall², this is the highest-value repair available.
   Note both authors cap it tightly and one is explicitly testing "whether the remaining
   loss is over-repair" — so it is a knob with a real optimum, not a free win.
2. **Prune isolated nodes.** `xiaoleilian` prunes degree-0 detections. They cannot
   contribute an edge, but they do count in `N_pred`, so removing them improves the budget
   multiplier. Not provably free — a pruned node may currently be winning a bipartite
   match that would otherwise go to a linked node — so it must be measured, but it is
   cheap and the mechanism is sound.
3. **Two-pass linking, tight then full.** Match confident short-range pairs first, then
   run a second pass over what is left. Attacks identity swaps, which is precisely our
   ~42k wrong links.
4. **Motion-aware linking.** Predict position from the previous displacement rather than
   matching raw positions. Recon §3 has p50 displacement 1.82 µm — small, but in a field
   with 8 µm spacing a velocity prior is exactly what separates the right successor from
   its neighbour.
5. **Boundary-track rescue** at `t = 0` and `t = T_max` — recon §1 found `44b6_` tracks end
   at the volume border ~100 % of the time, so this is aimed at our worse embryo.
6. **Divisions are worth ~0.009**, measured: zeroing them took the learned pipeline from
   0.915 to 0.906. Recon §6 said "ignore divisions" — that stands as a *priority* call, but
   it is not zero, and both learned notebooks run "conservative division recovery" with
   hard fraction caps (`safe_div_global_frac_cap ≈ 0.0036`). Last, not never.

## 5. Their parameter values against ours

| parameter | theirs | ours | note |
|---|---|---|---|
| link distance | ~8 µm ("benchmark used 15 — too loose") | 9.0 µm | close; our grid tests 4–9 |
| NMS / separation radius | ~4 µm | 6.0 µm | **our grid tests 3.5/4.5 — their value is near the low end** |
| detections per frame | 200–320 | ~210 | in band |
| divisions | off by default, or tightly capped | off | agrees |
| validation | grouped by embryo | grouped by embryo | agrees |

`xiaoleilian` also states "edge recall ≈ node recall²" independently — the same
relationship `notes/05` §3 used to size the linking gap.

## 6. What this does not tell us

- **Nothing about whether the visible test set is the scored set.** One notebook mentions
  "the hidden-test submission table", which is suggestive of a hidden split, and another
  refers to "307 biological cell divisions across embryo test datasets" — but the four
  visible test datasets carry only 3 GT divisions between them, so that number is almost
  certainly their *predicted* divisions, not ground truth. The leak flagged in `notes/05`
  §0 stands unresolved and still needs reporting.
- **No discussion-thread content**, so no organiser answers, no frame interval, and no
  confirmation of the dataset construction story in `notes/03-domain-intel.md`.
