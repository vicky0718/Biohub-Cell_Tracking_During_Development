# Where this ends: 0.945, and what closed each direction

Final state, 9 days out: **`ttasec`, 0.945**, rank ~648 of 3,460. 46 submissions used.

## 1. The fine-tune is closed — on logistics, not on merit

`claude-train-elastic` produced a real fine-tuned checkpoint (sha `ac7fc4b0`, 136/136 tensors
restored, 30 epochs, no divergence). `claude-arm-ftune` installed it correctly and ran clean on
the four public clips. **It then failed the graded rerun**, and Kaggle does not expose that
run's log.

The fine-tuned detector emits **+8.1% nodes** (132,630 vs 122,735). The graded rerun is already
~11 h against a 12 h limit (`MEMORY.md`), and the one supporting datum available is that `mtl4`
completed at +2.7% while `ftune` failed at +8.1%. That is suggestive, not proof.

Thresholding cannot bring it back:

```
threshold   nodes     excess removed
0.965     132,630     —
0.975     132,214     416   of 9,895   (0.3%)
0.99      130,807   1,823   of 9,895   (18%)
```

Most of the usable range buys 18% of what is needed. The extra detections sit above 0.99 —
they are confident, not marginal. **`ftune` is unsubmittable as built**, and we never got to
learn whether the model is better, only that it does not fit the time box.

## 2. Two wrong predictions from one misread column

I predicted the fine-tune would emit **fewer** nodes; it emitted 8% more. I then predicted
raising the threshold would trim them; it trimmed 0.3%. Both came from reading the training
log's `recall` as **detection** recall when it is the **edge classifier's** — a column that
says nothing about how many cells the detector finds. One misread, two confident calls, and a
submission slot spent on the first.

The general shape, which `notes/81` should have already taught: a number measured on a
contaminated holdout does not describe behaviour on inference data, and a number measured on
one head does not describe another.

## 3. What is closed, and how

```
direction                        closed by                                evidence
post-processing knobs            13 arms on the board                     0 gains
divisions                        gates moved 90 apart, tp fixed at 2      notes/78
ILP division weight              0 survive at penalty 0.0                 notes/79
motion relink removal            0.944 vs 0.945                           notes/81
offline measurement              predicted +0.0337, actual -0.001         notes/81
TTA extensions (z16, dominance)  0.942 x3                                 notes/84
additivity of +0.001s            failed 3 times (sewdet, ttasecret85)     notes/85
fine-tuning                      cannot complete a graded rerun           this note
```

Every one of these is a measurement, not an argument. That is the part worth keeping.

## 4. What I would and would not do with the remaining slots

**Would not**: build more arms. Thirteen knob arms found nothing and the mechanism surface is
mapped. Generating variants now would be activity, not progress.

**Open, if you want to spend slots**: resubmit `ftune` unchanged once. "Unhandled error" can be
infrastructure rather than the notebook, the public run is clean, and it costs one slot to find
out. I am not recommending it — the +8% node count is a real and specific reason to expect the
same failure — but it is the only untested thing left and it is cheap.

**Recommend**: keep `ttasec` at 0.945 as the final answer.
