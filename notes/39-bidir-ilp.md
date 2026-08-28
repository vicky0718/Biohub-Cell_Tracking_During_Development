# Correction: appearance makes divisions, not disappearance. And the two levers are substitutes.

`claude_bidir_ilp`, 12 datasets, one blended prediction pass at w=0.4, 8 ILP settings
re-solved from the cached candidates.

**The cache round-tripped exactly** — `div_J 0.0263 vs 0.0263`, `edge_J 0.9444 vs 0.9444`
against `notes/38`'s `w0.4`. That is the clean control `notes/38` §3 could not provide, and
it confirms a fresh blended prediction is reproducible.

```
a0.25_d2.0+repair   0.9506   best
notes/38 w0.15      0.9499   +0.0007   <- inside the 0.001 noise floor
notes/38 control    0.9462   +0.0044
```

**Answer: re-locating the ILP with the blend on does not add.** The two levers are
substitutes. `notes/38`'s w=0.15 at the `notes/36` weights stands as the operating point.

---

## 1. 🚨 Correction: I had the division mechanism backwards, in two notes

`notes/35` §2 claimed *"a high disappearance cost makes the solver unwilling to end a track,
so it forks instead"*, and `notes/38` repeated it. **The data already in `notes/36` refutes
it**, and I wrote the correct version there without noticing the contradiction:

```
APPEARANCE axis (disappear fixed 2.0)        DISAPPEARANCE axis (appear fixed 0.1)
appear  0.13  0.25  0.4   0.67  1.0   2.0    disappear  0.5  0.75  1.0  1.5  2.0
forks     53   421 1,443 2,477 2,477 2,468   forks       51    32   29   24   20
```

**Forks rise 47× with appearance and fall 2.5× with disappearance.** It is the *appearance*
cost that creates divisions, exactly as `notes/36` §2 said: a division is one parent with
two children, and if starting a new track is cheap the solver simply starts one for the
second child instead of forking. Expensive appearance makes forking the cheaper option.
Disappearance does the opposite — it keeps more tracks alive as 1:1 continuations, claiming
nodes that could otherwise have become second children.

This run's prediction 2 was built on the wrong claim and correctly failed
(`forks do not track the penalty`). The failure is the note being wrong, not the run.

**What it does not change:** `ratio0.4_2.0` is still the located optimum and still scored
0.897; the *weights* were found empirically and are unaffected by my misreading of why they
work. What it changes is which knob to reach for next — appearance, not disappearance.

## 2. The blend suppresses divisions by removing the evidence, not the incentive

`notes/38` asked whether the ILP could trade the divisions back. It cannot:

```
                div_J    forks
best recovery   0.0465   1,517   (a0.8_d4.0)
unblended       0.0690   1,443   (notes/38 control)
```

No setting in the grid reaches 0.05, and the ones that raise `div_J` cost more on the edge
term than they return. The blend admits ~2.6 % more candidate edges, and those edges give
the solver 1:1 continuations for cells that were previously only explicable as a second
child. The fork evidence is gone, not merely unrewarded.

## 3. Fewer forks, better forks — and the appearance axis is on the boundary again

The winning arm is worth reading closely:

```
                forks   div_J   edge_J   total
a0.4_d2.0       1,541  0.0263   0.9444  0.9492
a0.25_d2.0        669  0.0370   0.9450  0.9506
```

**Half the forks and a higher `division_jaccard`.** `div_J = TP/(FP + D)`, so cutting false
forks raises it even with the same true ones — precision, not volume. Same shape
`notes/35` §2 saw unblended, where `ratio0.4_2.0` beat `sym0.5` with a third of the forks.

And `0.25` is the **lowest appearance tried at that disappearance**. On unblended candidates
the optimum was 0.4 (`notes/36`); with the blend on it has moved down past the grid edge.
That is a boundary for the fourth time in this project, and it is nearly free to fix — this
run cached its blended candidates, so an appearance extension is solves only, no GPU.

## 4. Next

1. **The temporal linker** (`notes/33` §1) — `biohub-temporal-unet3d-seed314159-v1`, the
   second missing model, still untouched. Everything since `notes/33` has re-read the one
   model we have; this adds capacity. It is the largest identified remaining gap: the
   public notebooks run three models to our one and score 0.923–0.927 against our 0.897.
2. **Extend the appearance axis below 0.25** on this run's cached blended candidates (§3).
   Cheap, and the axis is demonstrably not exhausted.
3. **Submit `bidirectional w=0.15` at the `notes/36` weights.** +0.0036 on train, above the
   noise floor, and now confirmed to be the best available configuration of the levers we
   have. At the observed transfer range (0.59–1.22×) that is +0.002 to +0.004.

Banked floor **0.752**. Best scored **0.897** (rank ~1336/2792).
Bronze **0.926**, gold **0.944**.
