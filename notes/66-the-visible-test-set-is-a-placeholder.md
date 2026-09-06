# The visible test set is a placeholder — and I built an oracle on it before checking

I found that the four "test" datasets are byte-identical training datasets, built a local
scorer on them, watched it invert the one comparison it could be checked against, and only
then searched the forum. The forum had the answer, posted on 2026-07-08, five days before
this project started.

**Read §1 first. Everything I inferred from the test mount is about placeholder data.**

## 1. The four visible test clips are dummies. The real test set is swapped in at grading

Topic 723921, host and participants, 2026-07-08 to 07-12:

> *"Its a placeholder to see if your notebook actually produces a csv file… Once you submit
> the notebook for evaluation, a different test set is used (that you don't have access to)."*
>
> *"Those four events won't be used to calculate your leaderboard score. They are only there
> to test if your solution works mechanically… Kaggle has two ways of running your notebooks:
> Normal mode, and submission mode… Once you run in submission mode, the true test set is
> used. That's why internet access is disabled for submission mode: so you can't exfiltrate
> the real data."*

And from the Overview, quoted in topic 717228: *"When a notebook is submitted for rerun, a
new hidden test set is swapped in. The size of the hidden test set is approximately the same
size as the training dataset."*

A participant had already published the byte-level verification I re-derived — pixel
comparison at t=0, 50, 99 for all four clips — on 2026-07-08.

**Consequences, all of them corrections to things I wrote earlier today:**

* **`claude_nest2` does not refute `notes/49`.** Its "the test set is a third pair of
  embryos" is *unverified*, not false. The four stems I matched against training are
  placeholders; the graded data is ~199 datasets nobody outside the host has seen.
* **The node-budget numbers below describe the placeholders.** `claude_fork` sits 11.9%
  under budget *on the dummy clips*. Whether it does on the graded set is unknown.
* **Every `submission.csv` we can download is the verification-mode output** — four
  datasets, 118,659 nodes. The file that was actually scored was produced in submission mode
  inside Kaggle and never leaves it. `tools/diff_arms.py` compares verification outputs.
* **The oracle did not fail because it was noisy.** It failed because it was scoring
  different data from the leaderboard. That is a stronger and cleaner result than the one I
  wrote first.

The method failure is worth naming: I searched the forum *after* building on the finding.
`notes/58` read 43 posts by one author and `discussions/` holds 97 scraped threads; the cost
of grepping them for "test" before writing a notebook is a minute. Order matters more than
effort here.

## 2. What the placeholder data did establish (about placeholders)

These are exact for the four dummy clips and suggestive, at best, for the graded set:

```
dataset            N_pred    N_est     ratio   factor      GT annotated
44b6_0113de3b      25,425   25,755    -0.013   1.0013            52
44b6_0b24845f      18,089   32,795    -0.448   1.0448            51
6bba_05b6850b       6,033    6,362    -0.052   1.0052           861
6bba_05db0fb1      69,112   69,800    -0.010   1.0010         1,229
TOTAL             118,659  134,712    -0.119   1.0119
```

Three of four land within 5% of `estimated_number_of_nodes`. If that behaviour carries —
and it is a property of the pipeline's own density targeting, not of these clips — then the
metric's uncapped under-prediction bonus is already close to fully collected, pruning trades
`edge_J` against a factor at or above 1, and `notes/46`, `notes/48` and `notes/52` closing
was not a coincidence. **Suggestive. Not measured on the graded set.**

**Degree-0 nodes: zero of 118,659** on both scored submissions. Dropping edgeless nodes
would have been free score — no edge lost, `N_pred` down, factor up — and
`OUTPUT_FILTER_SHORT_TRACKS` already removes every one. This one *is* structural: it is a
property of the output filter, not of the data it ran on.

## 3. The oracle, and why it inverted

`claude_oracle` scores any mounted arm against the placeholder GT with `harness/purescore.py`
embedded. Graded on one thing: reproducing the known gap between two arms.

```
                 local adj_edge      LB       local delta      LB delta
claude_fork          0.8975        0.937
claude_forkw085      0.9043        0.932       +0.0068         -0.0050
```

Inverted, and now explicably so — the two numbers are computed on different datasets. On top
of that the placeholder annotation is sparse and unevenly sparse (52 and 51 nodes for two of
the four; the whole local delta comes from one dataset), so even as a measurement *of the
placeholders* it is n≈1.

> **A local score against the visible test folder is not a leaderboard estimate and cannot
> be made into one.** Not by reweighting, not by dropping the small datasets.

That is the third train-side screen to point the wrong way, after `notes/49`'s pooled folds
(0.901 → 0.863) and `notes/64`'s PROXY (+0.0017 predicted, −0.005 delivered) — and the first
whose failure has a mechanism rather than a suspicion.

## 3b. The graded set is ~17× the placeholders, and its node counts ARE observable

Every submission record carries `totalBytes` — the size of the `submission.csv` that
submission mode produced. Against the 12,127,461 bytes verification mode writes for four
clips:

```
date        score   graded bytes    ratio   ~clips   notebook
2026-08-28  0.897    200,605,027    16.5x      66     claude_submit_ratio
2026-08-29  0.901    204,296,141    16.8x      67     claude_submit_config
2026-09-02  0.863    173,848,827    14.3x      57     claude_submit_topk
2026-09-04  0.937    204,192,768    16.8x      67     claude fork
2026-09-05  0.932    203,859,527    16.8x      67     claude forkw085
```

The graded test set is **roughly 17× the verification set, about 60-70 clips** — not the
~199 the Overview's "approximately the same size as the training dataset" implies. That
matters for §4's runtime worry, and it turns file size into a free, graded-set-side readout
of how many nodes an arm emitted.

**And it prices pruning with real evidence rather than inference.** `claude_submit_topk`
wrote **14.9% fewer rows** than the configuration it was derived from and scored **0.863
against 0.901** (`notes/49`). Under the metric, dropping 14.9% of nodes buys at most
`0.1 × 0.149 ≈ +1.5%` relative on `edge_J` — about +0.014 — so `edge_J` itself must have
fallen by roughly 0.05 to produce that. Pruning does not pay on the graded set, and this is
the first statement about the node budget in this repo resting on graded data rather than
on placeholders.

## 4. A risk this turned up: the 12-hour rerun

The graded rerun is ~17× the verification work (§3b). `claude_fork` and `claude_forkw085`
both drew **T4s** and completed. Every draw today has been a **P100**, and the arms now
carry a wheelhouse prologue that installs torch 2.5.1+cu121 before running (`notes/65` §1).
A P100 is slower than a T4 on this workload and the install itself costs minutes.

**So an arm can pass verification and still time out in submission mode.** Nothing measured
yet; flagging it because the failure would appear only after a submission slot is spent.

## 5. What survives

* **`tools/diff_arms.py`** — no-op detection. Two configs producing identical verification
  output are the same config; `notes/60`'s trap (parameter moves, volume cap binds first) is
  caught for free. Weaker than it looked — four clips, not 199 — but structural.
* **`data/test_gt.json`** — the placeholder GT, exported once, for local diagnostics.
* **`claude_nest`/`nest2`/`oracle`** — kept as the record of how this was established.

Discarded: local `adj_edge_jaccard` as a ranking signal, and the idea that anything about
the graded set can be read off the visible test folder.

Arm selection stays where `notes/65` put it: **the leaderboard, five a day, chosen by
argument.** The arguments available are `notes/65` §3's swept-vs-stepped distinction and the
crossing of two independent public routes to 0.941 — not a number this repo produced.
