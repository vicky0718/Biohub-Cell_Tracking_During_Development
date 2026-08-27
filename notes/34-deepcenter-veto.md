# The veto's ceiling is ~0.002, and that bound holds even though the run is unreadable

`claude_deepcenter_veto` v4, 12 datasets, 6 arms, 1,273 s, CPU. It ran clean after three
launch failures (`notes` in the commit log; all mine, none of them the science). The
checkpoint loaded under a strict `load_state_dict`, so the architecture reproduction is
right, and the heatmap cache worked as designed (96 heatmaps per dataset, not 4 × 96).

**And the result is not readable.** Three flaws, all in how I designed the measurement:

| flaw | what happened |
|---|---|
| alignment sample | `frames[:4]` of *annotated* frames gave **n = 8, 4, 4 GT nodes**. A median over 4 points cannot establish alignment. |
| dataset subset | `names[:12]` sorted alphabetically gave **10 × `44b6` and 2 × `6bba`**. The population is 71/128 the other way. |
| the PASS threshold | prediction 5 "passed" on a margin of **+0.0000** — it required only `> 1e-6`, on a metric the leaderboard reports to 3 decimals and whose movie-to-movie spread is ±0.14. |

My own grading cell says "NOTHING BELOW IS READABLE until [alignment] passes." I am
honouring that rather than mining the arms for a number I like.

---

## 1. The one thing this run does establish, and it is a bound

Two arms bracket what `close_gaps` is worth, because `veto0.6` rejected **100 %** of
candidates — it is therefore *smoothing alone*:

```
norepair   0.8992
veto0.6    0.9079    +0.0087   <- linefit_smooth alone (gap-close fully vetoed)
repair     0.9092    +0.0100   <- both
                     ------
gap-closing's entire contribution:  +0.0013
```

**A veto on gap-closing cannot be worth more than gap-closing is**, and gap-closing is
worth **+0.0013** here. That bound does not depend on the alignment check, and barely on
the subset — it is a decomposition of the arms against each other, not a comparison to
ground truth.

It also cross-checks against an independent earlier measurement. `notes/27` §1 attributed
**78 %** of the repair chain's +0.0115 to position smoothing; here smoothing is **87 %** of
+0.0100. Two different runs, two different subsets, same conclusion: **`close_gaps` is the
minor half of the repair chain, and `linefit_smooth` is the major one.**

So the deepcenter veto, applied to gaps, is playing for ~0.001–0.0025 against a 0.043 gap
to bronze. **Not worth re-running, and not worth a submission slot.** I am not fixing the
three flaws and re-launching this arm.

## 2. What the alignment numbers hint at, stated as a hint and nothing more

```
44b6_144b256d   n=8   GT median 0.246   random 0.063   3.9x
44b6_18ced818   n=4   GT median 0.103   random 0.062   1.7x
44b6_1d530831   n=4   GT median 0.457   random 0.096   4.7x
```

GT beats random on all three, by 1.7–4.7×. That is the right *direction*, and at n=4 it is
not evidence. The veto also behaves monotonically with threshold (23 % → 73 % → 99 % →
100 % rejected), which is what a working scorer on a real scale looks like and not what a
randomly-firing one looks like.

Read together: the heatmap is **probably** aligned and the geometry in
`probes/exec_deepcenter.py` is **probably** right. I am not going to spend a run proving it,
because §1 says the answer does not matter for this application.

## 3. The deepcenter model is not dead — its *gap* application is

The public notebook uses this same model for two gates, and this run tested one of them:

```
BIOHUB_DEEPCENTER_GAP_VETO       = 1   threshold 0.25    <- tested, ceiling ~0.0013
BIOHUB_DEEPCENTER_SAFE_DIV_VETO  = 1   threshold 0.12    <- NOT tested
```

The second gates **division insertion**, and that is a different order of prize.
`notes/25` measured geometric fork insertion at **1 TP per 2,223 guesses** and closed the
division term on that basis. `notes/31` reopened it: model-driven forks took
`division_jaccard` from 0.0000 to **0.0562**, priced at a near-exact wash against the edge
term. `notes/33` found competitors reporting `div_J` of **0.12 and 0.3** — worth +0.012 to
+0.030 of score on a term we currently score **0.000** on.

A precision gate on fork insertion is exactly the tool that could move a 1-in-2,223
precision. That is the application worth testing, not gaps. **`pipeline/deepcenter.py` and
the `accept=` hook stay** — they are tested, the checkpoint loads, and
`pipeline/divisions.py::insert_divisions` takes the same kind of gate.

## 4. Next

1. **The temporal linker blend** — the *second* missing model
   (`biohub-temporal-unet3d-seed314159-v1`). It acts on **edge probabilities**, the term
   worth 1.0 of the metric's 1.1, where this veto acted on invented nodes worth 0.0013.
   `notes/33` §2 also found bidirectional harmonic linking, which needs no new weights at
   all. This is where the remaining ~0.04 to the public notebooks most plausibly lives.
2. **Gate `insert_divisions` on deepcenter** (§3), reusing the `accept=` hook already
   built and tested.
3. Whatever `claude_ilp_sweep2` reports; still running.

**Method note for anything built after this.** Two of the three flaws above are sampling
errors that a glance at the printed `n` would have caught, and I printed `n` and did not
look. Any future subset must be checked against the 71/128 embryo split before the run, and
any PASS threshold must be larger than the noise it is being read against.

Banked floor **0.752**. Best scored **0.883** (rank ~1671/2792). Bronze **0.926**, gold **0.944**.
