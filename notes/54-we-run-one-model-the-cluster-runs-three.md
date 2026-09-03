# Top 100 needs 0.938. We run one model; the 0.936 cluster runs three.

Goal set explicitly: **rank 100**. From today's leaderboard scrape:

```
3,038 teams, top 0.963          our 0.901  ->  rank ~1388
rank  10   0.949        rank 100   0.938        <- the target
rank  25   0.945        rank 150   0.936
rank  50   0.941        rank 300   0.935
gap to target: +0.037
```

The field is not a smooth gradient — it is a wall:

```
0.941    7 teams
0.939   11
0.938   56      <- target sits at the top edge of a pile-up
0.936  117
0.935   83
0.933   83
```

Clearing 0.938 means passing ~1,288 teams, almost all of them stacked in 0.930–0.938.

---

## 1. The metric hack is a ghost. Do not chase it.

The most-voted public notebooks are titled `improved-metric-hack-last-call` (108 votes),
`metric-hack-minimal-baseline-tta-2gpu` (84), `biohub-0-948-reproduction`. Reading
`amanatar/improved-metric-hack-last-call`: `augment_dataset()` injects a hub node at
`t = -1000`, position `(-10000, -10000, -10000)`, wires it to up to 3,000 real track roots,
and fabricates 20 fork structures at timepoints that do not exist.

**It is already patched.** Thread 736937:

> *"The metric bug was fixed and the leaderboard was recalculated, but some public notebooks
> still display their old, inflated scores from the pre-fix metric… we are essentially
> keeping a few ghosts at the top of the leaderboard."*

So those titles are stale pre-fix numbers. Today's scrape is post-fix, which means the
0.936 pile-up **is real and legitimately reached**. TWEAK (thread 739018, post-patch) does
report three residual inconsistencies from synthetic graphs, but they are unresolved and
the organisers are actively closing this surface with a month to the deadline. Building on
it risks being invalidated before scoring. **Ruled out — not on ethics alone, on the fact
that it is patched and the scores are ghosts.**

## 2. The actual reason we are 0.035 behind, and it has been visible all along

`notes/25` recorded it without the comparison being drawn:

```
our reproduction of the pack + ILP        LB 0.867
the cluster, running the SAME weights     LB 0.913-0.916
                                              -0.046
```

We never reproduced their **pipeline** — only their **model**. Everything built since
(+0.0115 repair, +0.0221 ILP weights, +0.0036 bidirectional = ~0.034) has been clawing back
ground the standard pipeline already had. We are at 0.901; the same-weights cluster is now
at 0.936. **The deficit never closed; it moved.**

And the specific difference is now visible. Every notebook in the 0.936 band mounts the
**same three models**:

```
pilkwang/biohub-tracking-support-pack-50ep-v1        <- we use this
pilkwang/biohub-temporal-unet3d-seed314159-v1        <- we do not
pilkwang/biohub-deepcenter-unet3d-center-prior-v1    <- we do not

nusrati/0-938 (27v)   nusrati/0-936 (78v)   rishabhr0y/936-dc-ours
flexonafft/biohub-agreement-gated-dual-seed-fusion   (+ a 4th unet3d)
```

**They run a three-model ensemble. We run one model.**

## 3. `notes/40` demoted the model thesis, and the demotion looks wrong

`notes/33` put ~0.04 in "the two missing models". `notes/40` demoted that because our own
tests returned little:

```
deepcenter veto      ~0.002 ceiling   notes/34 -- and applied to GAP-CLOSING, not detection
claude_secondary     +0.0026, t=0.63  notes/42 -- UNRESOLVED at n=12, never re-measured
```

But both tested a model as an **isolated bolt-on to our own pipeline**, with our own
blending, at n=12. Neither tested the **integrated three-model configuration** the entire
0.936 cluster runs. `notes/33`'s ~0.04 estimate matches the observed 0.035 gap almost
exactly, and it was set aside on evidence that does not address it.

**This is the single clearest actionable finding in the project right now**, and it is
cheap: the notebooks are public and mount only public datasets.

## 4. The plan

**Priority 1 — reproduce the three-model pipeline.** Take `nusrati/0-936` (78 votes, all
three models, post-fix) as the base. Run it unmodified, confirm it reproduces. That alone is
+0.035 and lands at the target's doorstep. Check the licence and carry attribution.

**Priority 2 — stack what is genuinely ours onto their base.** The ILP weights (+0.0221) and
repair chain (+0.0115) are structural and transferred five times out of five (`notes/49`'s
split: structural transfers, calibration does not). Some may already be present in their
pipeline; the diff is the deliverable, and only the parts they lack can stack.

**Priority 3 — spatial test-time augmentation.** Untried here. `bidirectional` is TTA over
*time* (+0.0036) and nobody has tried flips or rotations over *space*, which is standard in
every vision competition and is what `outwrest/...-tta-2gpu` uses. Cheap, no training.

**Not now:** dense pseudo-labels via FOCUS-3D (`notes/53`) and any from-scratch training
(~333 h against ~140 h of quota, `notes/33` §5). High ceiling, wrong order — they are how
you pass 0.947, not how you reach 0.938.

## 5. What is still running

```
claude_loosen   det_threshold below 0.965, where the budget is spent   notes/52
claude_union    do the two detectors find different cells              notes/53
```

Both are cheap and pre-registered. Neither is on the critical path to 0.938 any more —
§2 is.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
the gap is a PIPELINE gap, not a modelling gap, and it is ~0.035
```
