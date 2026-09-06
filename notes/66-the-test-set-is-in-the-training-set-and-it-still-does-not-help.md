# The test set is in the training set — and scoring against it still points the wrong way

Two findings, and the second undoes most of the first.

## 1. `notes/49`'s premise was wrong: the test datasets ARE training datasets

`claude_nest` went looking for the metric's `N_est` in the test mount, found no `.geff`
there at all, and on the way past printed a directory listing that contained the answer to a
different question. `claude_nest2` checked it properly:

```
test    44b6_0113de3b   44b6_0b24845f   6bba_05b6850b   6bba_05db0fb1     (4 zarr, no geff)
train   199 zarr, 199 geff  — all four test stems among them
        identical shapes [100, 64, 256, 256], identical embryo prefixes
```

**All four hidden-test datasets are training datasets with ground truth on disk.**
`notes/49`'s "the test set is a third pair of embryos" — the premise behind per-embryo
grading and behind distrusting every train-side screen since 2026-09-02 — is false. Same two
embryos, same movies.

Two things fall out immediately, and both are exact rather than estimated:

**The node budget is readable, and the direction is closed.** `estimated_number_of_nodes` is
GEFF metadata, so it exists for these four:

```
dataset            N_pred    N_est     ratio   factor
44b6_0113de3b      25,425   25,755    -0.013   1.0013
44b6_0b24845f      18,089   32,795    -0.448   1.0448
6bba_05b6850b       6,033    6,362    -0.052   1.0052
6bba_05db0fb1      69,112   69,800    -0.010   1.0010
TOTAL             118,659  134,712    -0.119   1.0119
```

`claude_fork` sits **11.9% under budget** and is already collecting the metric's uncapped
under-prediction bonus — worth about **+0.011** at `edge_J` 0.92. Pruning further trades
`edge_J` against a factor that is already above 1, which prices that whole direction near
zero and retroactively explains why `notes/46`, `notes/48` and `notes/52` all closed.

**Degree-0 nodes: zero of 118,659,** on both scored submissions. Removing edgeless nodes
would have been free score — no edge lost, `N_pred` down, factor up — and
`OUTPUT_FILTER_SHORT_TRACKS` has already removed every one.

## 2. The oracle fails: local scoring gets the sign wrong

If the GT is on disk, a local score should replace a submission slot. `claude_oracle` scores
any mounted arm against it with `harness/purescore.py` embedded verbatim. It was graded on
one thing — reproducing the leaderboard gap between the two arms whose scores we know.

```
                 local adj_edge      LB       local delta      LB delta
claude_fork          0.8975        0.937
claude_forkw085      0.9043        0.932       +0.0068         -0.0050
```

**The sign is inverted.** Local scoring says `w=0.85` is the better arm by +0.007; the
leaderboard says it is worse by 0.005. The level is off too — 0.8975 against 0.937 — but
level could be a constant offset, and the sign cannot.

The reason is visible in the GT itself. These datasets carry **52, 51, 861 and 1,229**
annotated nodes against estimates of 25,755 to 69,800:

```
44b6_0113de3b    52 annotated / 25,755 estimated      495x
44b6_0b24845f    51 annotated / 32,795 estimated      643x
6bba_05b6850b   861 annotated /  6,362 estimated        7x
6bba_05db0fb1 1,229 annotated / 69,800 estimated       57x
```

The public annotation is a **sparse and unevenly sparse subsample**. Two of the four
datasets are graded on fifty edges. And the whole local delta comes from one dataset —
`6bba_05db0fb1` moves +0.0114 while the other three move by 0.0005 or nothing — so the
comparison is effectively n=1.

Whether the private scoring GT is denser than the public one, or the same annotations
weighted differently, is not something we can see from here. Either way:

> **The local score is not the leaderboard. It is not a gate.**

This is the third train-side screen to point the wrong way, after `notes/49`'s pooled folds
(0.901 → 0.863) and `notes/64`'s PROXY (+0.0017 predicted, −0.005 delivered). The premise
`notes/49` gave for distrusting them was wrong; the conclusion survives the correction on
different grounds. Sparse, unevenly distributed annotation is enough on its own.

**What is deliberately not being done here.** Restricting the local metric to the two
datasets with real sample sizes, or reweighting to make the sign come out right, would be
fitting a screen to the single A/B it is being validated against.
`notes/41`, `notes/42` and `notes/44` each recorded an "optimum" that was one noisy sample,
and all three were withdrawn. One inverted comparison is not evidence for a repaired screen.

## 3. What survives, and how arms get chosen now

Kept, because they are computed from metadata and node counts rather than from annotations,
and are therefore exact:

* **`N_est` per dataset** and the resulting adjustment factor — the numbers in §1.
* **`tools/diff_arms.py`** — whether an arm's change reached the output at all. `notes/60`'s
  trap, where a parameter moves and a volume cap binds first, is caught here for free. An
  arm whose `submission.csv` is identical to its base has already been measured.
* **`claude_gtdump`** — the GT exported once, so these diagnostics run locally in seconds.

Discarded: local `adj_edge_jaccard` as a ranking signal.

So arm selection stays where `notes/65` put it: **the leaderboard, five a day, chosen by
argument rather than by screen.** The arguments available are the swept-vs-stepped
distinction (`notes/65` §3) and the crossing of two independent routes to 0.941 — not a
number this repo produced.
