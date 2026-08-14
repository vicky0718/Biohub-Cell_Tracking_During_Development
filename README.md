# Biohub — Cell Tracking During Development

Workspace for the [Kaggle contest](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
run by the Royer Group at CZ Biohub: detect and track cells in 3D+time light-sheet videos
of zebrafish embryos. Started 2026-08-13.

```
├── MEMORY.md                     state of play, condensed — read this first
├── notes/
│   ├── 01-competition-brief.md   the terrain: data, submission format, scoring, baseline
│   └── 02-metric-findings.md     what actually scores points — measured, not assumed
├── harness/
│   ├── scorer.py                 adapter onto the OFFICIAL scorer (never reimplemented)
│   ├── harness.py                fold-based scoring, caching, and the promote/reject gate
│   └── submission.py             build submission.csv + catch what the scorer fixes silently
├── probes/
│   ├── metric_probe.py           free FPs, the node budget, wrong-vs-missing links
│   ├── metric_probe2.py          gap bridging, division timing
│   └── test_recon_logic.py       offline dry-run of the recon notebook's linking code
├── tests/
│   └── test_harness.py           15 tests, runnable without competition data
└── notebooks/
    └── 01_recon.ipynb            run this on Kaggle first — answers the open questions
```

## The harness

Nothing gets promoted on a pooled number alone. `gate(baseline, arm)` requires the
change to improve the pooled score **and** regress no fold — a pooled gain paid for by
a fold regression is the shape of overfitting, and it is rejected:

```python
from harness import Harness, gate

h = Harness(data_dir=TRAIN)                    # official splits auto-loaded
base = h.evaluate(predict_v1, arm="baseline")  # results cached per (arm, dataset)
arm  = h.evaluate(predict_v2, arm="lower_threshold")
print(gate(base, arm))                         # PROMOTE / REJECT, with per-fold deltas
```

Scores come from the official `tracking_cellmot` scorer, never a reimplementation, so a
local number is the leaderboard's number. Run the tests with:

```bash
CELLMOT_REPO=/path/to/kaggle-cell-tracking-competition python tests/test_harness.py
```

## Where things stand

Compute is **Kaggle GPU notebooks**; the data is mounted there, so nothing needs
downloading. This container has no GPU and no Kaggle credentials — it is for analysis,
harness code and notebook authoring. Every number touching real data comes back from Kaggle.

Done so far: the metric has been taken apart and its behaviour verified against the
official scorer. Read [`notes/02-metric-findings.md`](notes/02-metric-findings.md) before
writing any modelling code — several of its conclusions run against the obvious instinct,
including the official baseline's own default detection threshold.

The headline: **ground truth is sparse, so false positives are nearly free, while a missed
detection is unrecoverable.** Detect aggressively, link aggressively, ignore divisions
until late.

## Next step

Run `notebooks/01_recon.ipynb` on Kaggle (CPU is enough — it never loads an image) and
commit `recon_summary.json` back here. It answers the questions the strategy hangs on:
annotation density, the node budget, inter-frame motion, cell spacing, division counts,
and the linking-only ceiling that splits the score into "detection problem" vs
"linking problem".

## Running the probes locally

```bash
git clone https://github.com/royerlab/kaggle-cell-tracking-competition
pip install tracksdata polars scipy
CELLMOT_REPO=./kaggle-cell-tracking-competition python probes/metric_probe.py
```

## Method

Carried over from the ROGII contest, where it worked: measure honestly against a fixed
harness, pre-commit to what counts as a win before running the A/B, and only promote a
change when the gain holds across every fold rather than in aggregate. Public-leaderboard
movement is not evidence. The discipline this is modelled on is written up in the ROGII
repo at `chat/memory/rogii-validation-harness.md`.
