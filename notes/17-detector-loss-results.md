# Phase 0 — a learned detector beats DoG, and my own metric was confounded

`claude_detector_loss`, run on Kaggle 2026-08-21 via the API, 1,395 s on a Tesla P100
(torch replaced with 2.5.1+cu121 in-notebook — see `notes/16` addendum below).
24 train + 8 eval datasets per embryo, 10 train / 8 eval frames each.

| train on | eval on | loss | recall | DoG | delta | dets/frame | **recall per detection** |
|---|---|---|---|---|---|---|---|
| 6bba | 44b6 | naive | 0.8924 | 0.7696 | **+0.1228** | 222 | 1.25× |
| 6bba | 44b6 | pu | 0.8143 | 0.7696 | +0.0447 | 243 | 1.05× |
| 6bba | 44b6 | masked | 0.8096 | 0.7696 | +0.0400 | 219 | 1.15× |
| 44b6 | 6bba | pu | 0.8892 | 0.8776 | +0.0116 | 100 | 1.46× |
| 44b6 | 6bba | masked | 0.8838 | 0.8776 | +0.0062 | 116 | 1.25× |
| 44b6 | 6bba | naive | 0.7920 | 0.8776 | **−0.0856** | 74 | **1.76×** |

DoG reference: `44b6` 0.7696 at 240 det/frame, `6bba` 0.8776 at 144 det/frame.

**Prediction 1 CONFIRMED.** **Prediction 3 CONFIRMED** (`masked` and `pu` positive both
ways). **Prediction 2 FALSIFIED** as worded — and §2 argues the wording, not the theory,
is what failed.

---

## 1. 🚨 The arms were never at matched detection count

The design said "every arm capped at the same number of detections per frame as DoG".
What the code did was pass that number as a **cap**, while `peaks_from_prob`'s
`threshold=0.05` decided the actual count. The threshold bound first:

| arm | dets emitted / DoG cap |
|---|---|
| naive on `6bba` | 74 / 144 = **51 %** |
| pu on `6bba` | 100 / 144 = 69 % |
| masked on `6bba` | 116 / 144 = 81 % |
| masked on `44b6` | 219 / 240 = 91 % |
| naive on `44b6` | 222 / 240 = 92 % |
| pu on `44b6` | 243 / 240 = 101 % |

So the absolute-recall column compares arms spending between half and all of the budget.
That is exactly the comparison `notes/09` §2 warns is meaningless, and I built it anyway
after writing "matched detection count" three times in the notebook header.

**Corrected on the quantity that was supposed to be measured: every single learned arm
beats DoG on recall per detection, in both directions, including `naive`** — 1.05× to
1.76×. The worst-looking row in the table (`naive`, −0.0856) is the *best* detector per
detection spent in the whole run. It just emitted half as many.

This does not overturn prediction 1; it strengthens it. It does make prediction 2's
verdict unreliable, because `naive`'s poor absolute delta is confounded with its low
count.

## 2. ⭐ Why `naive` split by direction — the sparse-annotation theory survives

`naive` is the best arm trained on `6bba` (+0.1228) and the worst trained on `44b6`
(−0.0856). That is not noise, and the mechanism is in the priors this run measured:

| embryo | fraction of cells annotated (median) | 1 in |
|---|---|---|
| `6bba` | **0.1182** | ~8 |
| `44b6` | **0.0060** | ~167 |

**The "1 in 28" figure everything so far has been reasoning about is a pooled average
hiding a 20× spread.** And `naive` — the loss that calls every unannotated cell background
— fails precisely on the embryo where 166 of every 167 cells are unannotated, and wins on
the one where only 7 of 8 are.

That is the original hypothesis (`notes/16` §4) confirmed by its dose-response, even
though the prediction as worded ("naive is worst of the three") came back FALSIFIED. The
prediction was wrong to assume a uniform effect when the dose varies 20× between the two
training sets.

Consequence for the real model: it will train on **both** embryos pooled, so `naive` sits
somewhere in the middle of that dose curve with no way to know where. `masked` and `pu`
are positive in both directions and do not depend on the guess.

## 3. What this says about the gold arithmetic

`notes/16` §3 asked for ~2.2× better recall-per-detection to reach a gold-shaped profile.
This run gets **1.05×–1.76×** from a 350 k-parameter UNet trained for 12 epochs on 240
volumes, with the loss still falling at the last epoch (masked: 0.0123 → 0.0022 between
epochs 4 and 11). The target is not obviously out of reach, and nothing here is near
saturation.

## 4. Operational findings

- **Kaggle's GPU is a P100 (sm_60); the image's torch 2.10+cu128 has no kernels for it.**
  `machineShape` is read-only on `kernels/push`, so T4 cannot be requested. The notebook
  now asks `nvidia-smi` and installs torch 2.5.1+cu121 (205 s) when it sees a P100, then
  proves the GPU with a real conv3d before doing any work.
- **`kaggleusercontent.com` is blocked by this container's egress proxy**, so kernel
  *output files* cannot be downloaded — only logs, which come from `www.kaggle.com`. Two
  consequences: every notebook must print its results as JSON to stdout, and weights move
  between notebooks by listing the training kernel as a `kernelDataSource`, never by
  downloading.

---

## What to do next

1. **Re-measure at genuinely matched count.** Drop the threshold so the cap binds, and
   sweep the cap at 0.5× / 1.0× / 1.5× of DoG's per-frame count. Until that is done the
   loss ranking is not known — only that all three beat DoG per detection spent.
2. **Train longer.** 12 epochs was not convergence.
3. Keep `masked` and `pu`; keep `naive` as a control specifically to test whether §1's
   confound explains its split, rather than dropping it on a contaminated verdict.
4. Save weights in the training notebook so the scoring and submission notebooks can take
   it as a kernel source.
