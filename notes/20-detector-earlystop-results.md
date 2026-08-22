# Phase 1b — `pu` clears DoG on both folds, and my collapse diagnosis was wrong

`claude_detector_earlystop`, Kaggle 2026-08-22, 7,189 s on a P100. 199 datasets × 25
frames, 15 epochs, held-out recall every 2, best checkpoint saved.

| loss | train | eval | **best** | @ep | final | collapse | **vs DoG** | phase 0b |
|---|---|---|---|---|---|---|---|---|
| **pu** | 6bba | 44b6 | 0.8826 | 3 | 0.7799 | −0.1028 | **+0.1131** | — |
| masked | 6bba | 44b6 | 0.8411 | 1 | 0.8175 | −0.0236 | +0.0715 | +0.0919 |
| **pu** | 44b6 | 6bba | 0.9148 | 3 | 0.8454 | −0.0694 | **+0.0372** | — |
| masked | 44b6 | 6bba | 0.8307 | 1 | 0.7936 | −0.0372 | −0.0469 | −0.0155 |

**`pu` beats DoG in both directions — the first learned configuration ever to do so.**
Mean margin +0.0751 against `masked`'s +0.0123. The gate to proceed to scoring is met.

Prediction 3 **CONFIRMED**. Predictions 1 and 2 **FALSIFIED**.

---

## 1. 🚨 The collapse mechanism I proposed in `notes/19` §2 is wrong

I argued `masked` collapses because its mask leaves the ambiguous middle without gradient,
letting the model reach **exactly 0.00000 loss** on the trivial bright-vs-dark problem and
then drift where detection actually happens. `pu`, constraining the whole volume, was
supposed to degrade gracefully. Prediction 1 tested exactly that and it failed:

| | mean collapse (best − final) | final training loss |
|---|---|---|
| `masked` | −0.0303 | **0.00008 – 0.00017** |
| `pu` | **−0.0860** | 0.042 – 0.007 |

**`pu` never drives its loss near zero and still collapses nearly three times as hard.**
Whatever causes the decay, it is not "the loss ran out of signal in the region that
matters". That story is retired.

**The better explanation is what the folds actually are.** Each model trains on *one*
embryo and is evaluated on *the other*. Both losses peak at epoch 1–3 of 15 and decline
from there, and `notes/19` §3 plus §2 below show more data of the same embryo making things
worse. That is the signature of the model learning **embryo-specific appearance** — which
by construction does not transfer to the held-out embryo — rather than memorising
individual images.

If that is right it carries a consequence worth stating plainly: **our CV is pessimistic
about the real submission.** Leave-one-embryo-out trains on one embryo and tests on
another; the submitted model would train on *both* and face two unseen ones. Two embryos of
training diversity should transfer better than one. With only two embryos available we
cannot measure that, so it stays a stated expectation, not a result.

## 2. More data still makes `masked` worse, even at its own best checkpoint

`notes/19` §3 raised this while confounded by the checkpoint bug. With best-checkpoint
selection on both sides it is clean:

| train on | phase 0b (240 volumes) | phase 1b (1,775–3,200 volumes) | change |
|---|---|---|---|
| 6bba | +0.0919 | +0.0715 | **−0.0204** |
| 44b6 | −0.0155 | −0.0469 | **−0.0314** |

So `notes/18` §4's headline — "the binding constraint is data volume" — is **refuted for
`masked`**, and this time without the confound. Adding 13× more of the same embryo made it
worse on both folds. Consistent with §1's reading: more of one embryo buys more
embryo-specific fitting, not more generality.

## 3. So why does `pu` win?

Not by avoiding collapse — it collapses harder. Two candidates, neither tested:

- **It uses the measured prior.** `pu` is the only loss that consumes
  `n_annotated / estimated_number_of_nodes`, and this run logs it at **0.0077 on `44b6`**
  and **0.0971 on `6bba`** — the 20× spread `notes/17` §2 found. It weights the unlabelled
  mixture correctly per dataset where `masked` throws that information away.
- **It peaks later.** `pu` peaks at epoch 3, `masked` at epoch 1, suggesting `pu` extracts
  more before the embryo-specific fitting takes over.

Both are guesses. What is measured is the ranking, and it reversed exactly as predicted.

## 4. The guard earned its place on the first run

`masked`/`44b6` peaked at 0.8307 against DoG's 0.8776 and was **not saved**. Three
checkpoints were written, not four. Without the guard that model would have gone to the
scorer as one of the two folds and dragged a mixed result out of a clean one.

It also makes the loss selection unambiguous: only `pu` covers both embryos, so
`claude_detector_score` has exactly one complete configuration to choose.

*(The run printed a stale note about "keep only the winning loss's files when attaching" —
that advice predates the fix and is impossible anyway, since attaching a kernel attaches
all its outputs. Selection now happens at load time.)*

---

## What to do next

1. **Score `pu` end-to-end.** `claude_detector_score` is built and tested, gated against the
   champion at 0.7070 with the champion reproduced in-run. It now also prints the split
   edge confusion matrix and the near/far miss breakdown, so a failure comes with a
   diagnosis rather than a bare number.
2. **The checkpoint policy is now load-bearing, not a nicety.** Every arm peaks at epoch
   1–3 of 15 and declines. Any future training run without best-checkpoint selection will
   throw away 0.02–0.10 of recall.
3. If §1's reading holds, **training on both embryos is the single largest untested lever**
   — and it is what the real submission does anyway. It cannot be validated with two
   embryos, but it can be *submitted* and measured on the leaderboard.
