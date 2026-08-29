# The node budget is worth up to ×1.1 and we collect none of it

Enumerating the public datasets — something this project had never done — turned up
`altervation/biohub-r35-spotiflow`: a **complete competition solution under MIT licence**,
from a team plausibly at rank 35. Its README is three lines long and every one of them
describes something we do not do.

```
Detector:    spotiflow_domain_r35 @ prob_thresh 0.3   (all-199 balanced FT from r22)
Tracker:     Phase-C · H2 min_track_len=5 on 44b6*
Node budget: Pivot I dens × 0.85 on 44b6* only
```

Then its density model's configuration:

```python
min_cells_per_frame: int = 1      sparse_min_cap: int = 3     # 44b6
max_cells_per_frame: int = 25     sparse_max_cap: int = 8
                                  dense_min_cap:  int = 6     # 6bba
```

**They predict 1–25 cells per frame. We predict about 220.**

---

## 1. The scorer, read rather than assumed

```python
ADJUSTMENT_ALPHA = 0.1
J_adj = max(0, J · (1 − ADJUSTMENT_ALPHA · (N_pred − N_total) / N_total))
```

`N_pred → 0` gives a multiplier of **1.1**. Ours is **1.0012** — we predict almost exactly
`N_total`, which is the estimated *true* cell count, and collect essentially none of it.

```
N_pred    ratio    multiplier   J_adj at unchanged edge_J
24,000    +0.000     1.0000            0.9350      <- us
12,000    −0.500     1.0500            0.9818
 6,000    −0.750     1.0750            1.0051
 1,200    −0.950     1.0950            1.0238
   670    −0.972     1.0972            1.0259
```

That right-hand column is an upper bound — edge Jaccard will not survive unchanged. But
the term is worth up to **+0.09**, against a 0.025 gap to bronze and 0.043 to gold, and we
have never touched it.

## 2. And over-prediction is otherwise free, which is why it went unnoticed

The edge term ignores predictions that match nothing. From `metrics.py`:

```python
edge_valid_pred = int(edge_attrs["pred_valid"].sum())
edge_fp = edge_valid_pred - edge_tp
# pred_valid = out_valid | in_valid, and both come from the GT node a prediction
# matched to, fill_null(False) for unmatched predictions
```

A predicted edge is counted — as TP *or* FP — only if an endpoint matched a **tracked GT
node**. Everything else is excluded, not penalised. That is why we can emit 22,000 edges
against ~600 ground-truth ones and still read `edge_J = 0.935`.

So of our ~24,000 predicted nodes, roughly **670 match ground truth and ~23,300 are pure
budget cost with zero edge benefit.** Deleting a node that matches nothing cannot lower
`edge_J` and can only raise the multiplier. There is no trade at all for those nodes; we
have simply been paying for them.

## 3. This explains why the config surface was flat

`notes/44` found every configuration axis flat inside the measurement's resolution and
concluded the direction was exhausted. It was — but the reason is sharper than "we found
the optimum". **Every sweep this project ran lived inside the predict-everything regime,
where the multiplier is pinned near 1.0 by construction.** Detection thresholds from 0.965
to 0.99 moved the node count by 5.6%; the lever that matters runs from 100% to 3%. We were
measuring a plateau at high resolution and never stepped off it.

## 4. It also retires `claude_zhpilot` as designed, on its own logic

The augmentation fix worked — val loss kept setting new minima to epoch 9 (1.546) instead
of bottoming at epoch 1, so limited image diversity was the right diagnosis. Then it
crashed on `detect_frame_dog` returning a tuple: **the same guess-the-return-shape mistake
as `peaks_from_prob`, twice in one file, in our own repo.**

But the run should not be fixed and relaunched, because its objective is wrong. zh001r's
labels are **dense** — every cell in every frame — so a detector trained on them learns to
find *all* cells. The metric rewards finding *only the annotated ones* and charges for
everything else. Dense supervision is the opposite of what this scorer wants.

Which is exactly what r35 did instead: Spotiflow **fine-tuned on GEFF spots** — the
competition's own sparse annotations — with the docstring noting zero-shot precision of
~0.002 before that fine-tune. Sparse-selective detection is the objective, and
`pipeline/unet.py`'s `masked_loss` and `pu_loss` were built for precisely that setting.
`notes/43` argued the positive-unlabelled machinery was an artefact of a limitation. It is
not; it is aimed at the actual target.

## 5. What to do, in order

1. **Measure the budget trade curve.** Keep the top-K nodes per frame by detection
   confidence, sweep K down through 100%, 50%, 25%, 10%, 5%, and read `adj_edge_jaccard`,
   `edge_jaccard`, `node_recall` and the multiplier separately. This is mostly
   post-processing on graphs we already build, and it either uncovers something worth
   +0.05 or shows the detector cannot rank annotated cells above unannotated ones — which
   is itself the answer to what to build next.
2. **If ranking is the bottleneck**, train detection against the GEFF annotations with the
   PU/masked losses, which is r35's move and what our own code was written for.
3. **Do not fix zhpilot as designed.** Dense labels train the wrong objective.

The bar from `notes/44` still stands — >0.0015 to be measurable, >0.01 to be worth a
submission slot. Unlike everything measured since `notes/34`, this candidate is nominally
worth several times the second figure.

Banked floor **0.752**. Best scored **0.901**. Bronze **0.926**, gold **0.944**.
Honest train anchor **0.9352** at n=60.

*`altervation/biohub-r35-spotiflow` is MIT-licensed; any use of its code or weights carries
the licence and the attribution with it.*
