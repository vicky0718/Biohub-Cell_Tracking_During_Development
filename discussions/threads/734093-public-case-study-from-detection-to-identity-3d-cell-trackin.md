# Public case study: From Detection to Identity — 3D cell tracking reasoning and aggregate evidence

- **URL**: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/734093
- **Topic id**: 734093
- **Author**: Luis Rosar (CONTRIBUTOR)
- **Posted**: 2026-08-09T20:23:41.311777100Z
- **Votes**: 1
- **Comments**: 1

---

## Opening post

I’m sharing a public technical case study developed in connection with the Biohub Cell Tracking During Development competition:

[](https://github.com/luisrosa/from-detection-to-identity)

The repository is available publicly to all competition participants.

The project documents how I decomposed the cell-lineage inference problem into distinct questions about evidence, identity, persistence, representation, validation, and global compatibility, and how those questions affected engineering decisions and experiments.

The public artifact includes:

* the problem formulation and question-to-decision ledger;
* aggregate training and held-out validation records;
* reproducible checkpoint-selection and continuation-promotion analysis;
* a comparison against a zero-motion physical baseline;
* aggregate uncertainty/calibration analysis;
* synthetic demonstrations of unknown-versus-negative supervision;
* public-safe analysis notebooks, utilities, tests, figures, and provenance checks;
* explicit limitations and rejected experiments.

For example, on 7,368 unique held-out ordinary-persistence edges, the selected motion field achieved approximately 0.570 µm error per coordinate versus 1.035 µm for a zero-motion baseline, a 44.9% reduction. The repository also documents why a later continuation state was rejected despite improving some point metrics: held-out motion likelihood deteriorated enough to worsen the declared validation objective.

The repository is intentionally **not** the complete competition implementation.

It does **not** contain Competition Data, competition images or annotations, model weights, checkpoints, the production architecture, training or inference source code, operating thresholds, proposal-reconciliation logic, relation-scoring implementation, global graph-selection implementation, or the submission-generation pipeline.

The public repository reproduces analyses from committed aggregate records; it does not reconstruct those records from the private training system or Competition Data.

Public software and notebooks in the repository are released under the MIT License. The repository’s disclosure and licensing files describe the boundary between public and private material in detail.

To reproduce the public artifact:

```bash
git clone https://github.com/luisrosa/from-detection-to-identity.git
cd from-detection-to-identity

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

make reproduce
```

Expected terminal conclusions include:

```text
PUBLIC RELEASE VALIDATION PASSED
CLEAN COPY REPRODUCTION PASSED
```

This post is the competition-specific public sharing notice for the repository so that the same released materials are available to all participants.

---

## Comments (1)


### Ragheb haddara (CONTRIBUTOR) — 2026-08-10T21:45:02.063Z

Really interesting approach. The 3D tracking problem is unique because cells can move in z as well as x,y. The zarr format for the microscopy data is efficient for chunked loading. For the submission, the key is getting the node/edge structure right — each dataset has its own graph of cell positions over time.
