# Metric findings — what actually scores points

Measured 2026-08-13 by running the **official scorer**
(`tracking_cellmot.metrics.evaluate`, from `royerlab/kaggle-cell-tracking-competition`)
on hand-built graphs. Everything below is a reproduced number, not a reading of the docs.
Reproduce with `probes/metric_probe.py` and `metric_probe2.py`.

> **Scope caveat.** These are toy graphs, chosen to isolate one metric behaviour each.
> Every *direction* here follows from a code path in `metrics.py` and is therefore
> structural. Every *magnitude* on real data still has to be measured — that is what the
> recon notebook is for. Do not trade on these numbers until the recon confirms the
> quantities they depend on (annotation density, `estimated_number_of_nodes`, division rate).

---

## 1. False positives are almost free, because the ground truth is sparse

Perfect prediction on 3 annotated tracks, then bolt on entirely fictional cells that
match no annotation:

| prediction | edge TP/FP/FN | J | n_pred |
|---|---|---|---|
| the 3 annotated tracks, perfectly | 15/0/0 | 1.0000 | 18 |
| + 3 unannotated tracks | 15/0/0 | 1.0000 | 36 |
| + 30 unannotated tracks | 15/0/0 | 1.0000 | 198 |
| + 300 unannotated tracks | 15/0/0 | **1.0000** | 1818 |
| + duplicate detections 0.8 µm from real cells | 15/0/0 | **1.0000** | 30 |

100× more predicted nodes, **identical score**. An edge is only ever counted — as TP *or*
FP — if at least one endpoint matched an annotated GT node. Edges between cells nobody
annotated are invisible to the metric. Even duplicate detections *inside* the 7 µm match
radius are free: bipartite matching takes one, the rest are ignored.

**The only thing that punishes over-detection is the node-count term**, below.

## 2. The node budget is a weak, two-sided multiplier

`adj_J = max(0, J · (1 − 0.1 · (N_pred − N_total)/N_total))`, `N_total` = the GEFF's
`estimated_number_of_nodes`:

| N_pred vs N_total | ratio | multiplier |
|---|---|---|
| 2× budget | +1.000 | **0.900** |
| exactly budget | 0.000 | 1.000 |
| 0.5× budget | −0.500 | **1.050** |
| 0.2× budget | −0.802 | **1.080** |

Two things fall out. Doubling your node count costs only **10 %** relative. And there is
**no upper cap** — under-predicting *multiplies your Jaccard above 1*, approaching ×1.1 in
the limit. You would have to over-predict by **11×** to zero the score.

## 3. A missed detection is unrecoverable — you cannot bridge a gap

GT track over 4 frames; prediction is missing the `t=2` node:

| prediction | edge TP/FP/FN | J |
|---|---|---|
| bridge the gap with a `t=1 → t=3` edge | 1/0/2 | 0.3333 |
| don't bridge it | 1/0/2 | 0.3333 |

**Identical.** The scorer drops every edge that does not span exactly `t → t+1`, so a
long-range link is not merely unrewarded, it does not exist. One missed detection
destroys the edge coming in *and* the edge going out — **2 FN, permanently**.

## 4. A wrong link costs exactly twice a missing link

Same GT track, varying only the last link:

| prediction | edge TP/FP/FN | J | cost |
|---|---|---|---|
| all 3 links correct | 3/0/0 | 1.0000 | — |
| last link omitted | 2/0/1 | 0.6667 | 1 FN |
| last link → wrong (unannotated) cell | 2/1/1 | 0.5000 | 1 FN **+ 1 FP** |
| correct link **plus** a spurious second link | 3/1/0 | 0.7500 | 1 FP + 1 division FP |

Note the third row: linking a matched cell to an *unannotated* distractor still scores an
FP. Once a predicted node matches an annotated cell, its outgoing edges are under scrutiny.

### The consequence is a decision rule, and it is counter-intuitive

"Wrong costs double" reads like an argument for cautious linking. It is not, because
**declining to link forfeits the GT edge anyway**. Writing `J = N/D` for the global
micro-averaged Jaccard and `p` for your belief that a candidate link is correct:

- don't link → `N / (D+1)`
- link → `(N+p) / (D+2−p)`

Linking wins when **`p > N/(N+D+1) ≈ J/(1+J)`**:

| current J | link if p > |
|---|---|
| 0.3 | 0.23 |
| 0.5 | 0.33 |
| 0.7 | 0.41 |
| 0.9 | 0.47 |

The threshold is **always below 0.5** and only approaches it as the score approaches
perfection. At a realistic J ≈ 0.5 you should emit any link you believe **more than ~⅓**
likely. Aggressive linking, not cautious linking, is correct — and the better you get, the
more selective you should become.

### Corollary worth testing: hedge instead of guessing

Out-degree is capped at 2, not 1, and the second edge only costs a division FP (weight
0.1). For a genuinely 50/50 link, emitting **both** candidates gives a guaranteed TP plus a
guaranteed FP `(N+1, D+2)`, versus a coin flip's `(N+0.5, D+1.5)`. At `N=50, D=100` that is
**0.5000 vs 0.4975** — hedging wins. The bounded downside is the division term: flooding
fake divisions can cost at most the whole `0.1 · division_jaccard`. Worth an A/B once we
know the real division count. *Hypothesis, not yet a finding.*

## 5. Divisions: do not force them

GT with one real division, 5 GT edges:

| prediction | edge TP/FP/FN | edge J | div TP/FP/FN |
|---|---|---|---|
| division predicted correctly | 5/0/0 | 1.0000 | 1/0/0 |
| division missed (one daughter dropped) | 4/0/1 | 0.8000 | 0/0/1 |
| division predicted **one frame late** | 3/1/2 | **0.5000** | 1/0/0 |

The division metric tolerates ±1 frame, so the late call still scored a division TP — but
it cost **0.30 of edge Jaccard** to earn at most `0.1 × div_jaccard`. Missing a division is
strictly better than mistiming one. Divisions are worth 1/10th of the edge term; chase
them last, and only at high confidence.

---

## Strategy that follows

1. **Detection recall on annotated cells is the hard ceiling.** It is the one quantity you
   cannot buy back later (§3). Everything else is recoverable.
2. **Detection precision is nearly free** (§1, §2). The official baseline's default
   `--det-threshold 0.99` — justified in its own CLI help as keeping precision up — is
   pushing on the axis that barely matters while sacrificing the one that is fatal.
   **Sweeping that threshold down is the first experiment**, and my prior is that it is
   worth a lot. The counter-pressures are real but weak: the ×(1−0.1r) node penalty, and
   more distractors making linking harder.
3. **Link aggressively**, at roughly `p > J/(1+J)` ≈ ⅓ (§4), and consider hedging ambiguous
   links across both allowed children.
4. **Divisions last**, high-confidence only (§5).
5. **Learn the node budget.** `estimated_number_of_nodes` is per-dataset GEFF metadata we
   can read on train but never on test. We need the train-side relationship between it,
   the annotated node count, and the cells a detector actually finds, so we can predict our
   test-side ratio instead of discovering it on the leaderboard.

## What must be measured before any of this is actionable

Open questions the recon notebook has to answer, all on train data:

- What fraction of real cells are annotated? (`n_annotated / estimated_number_of_nodes`)
- What is the inter-frame displacement distribution in µm? It sets the linking radius, and
  compared against the 7 µm match cutoff it tells us how confusable neighbouring cells are.
- Cell density: nearest-neighbour distance distribution. If typical spacing is below ~7 µm,
  detections will steal each other's matches and identity errors get expensive.
- How many divisions are there really — is the 0.1 term even worth a line of code?
- How many datasets, how many frames each, and what do the official 5-fold splits look like
  (`dataset_splits.json`)?
- **Linking-only ceiling**: given *perfect* detection (feed the GT nodes back in), what
  edge Jaccard does plain nearest-neighbour linking reach? That single number splits the
  problem into "how much is detection" vs "how much is linking" and tells us where to spend
  the next six weeks.
