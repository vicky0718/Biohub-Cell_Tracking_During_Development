# Motion relink is dead — it fights the ILP's global objective and loses

`claude_relink_sweep` v2, 24 budget-stratified datasets, 24 arms, 2,485 s. The index-space
fix (`notes/25` — coords rows vs tracksdata ids vs renumbered `Tracks` indices) held:
candidate-table remap survival ran **96–100 % (mean 98.2 %)** across every dataset, so this
run is reading real, correctly-aligned probabilities, not an artifact of the earlier bug.

| prediction | result |
|---|---|
| 1. control reproduces 0.8806 ± 0.0005 | **PASS** (0.8806 exactly) |
| 2. relink reduces `fn_mislink` | **FAIL** — every setting makes it worse |
| 3. learned bonus beats geometry+velocity | PASS, but hollow — see below |
| 4. single-peaked in `max_change_frac` | PASS, peak at **frac = 0**, i.e. no relink |

Prediction 4's own peak is the tell: **the best amount of relinking is none.**

---

## 1. The decisive number: every arm makes the target bucket worse

`fn_mislink` starts at **473** (matches `notes/26`'s original count exactly — a real
cross-run consistency check). Every pure relink arm, at every setting tried:

```
frac0.03     473 -> 563   (+90,  mildest budget tried)
frac0.05     473 -> 635   (+162)
geom_vel     473 -> 815   (+342)
wide         473 -> 1,099 (+626, worst)
```

**Damage scales monotonically with how much of the graph relink is allowed to touch.**
At the mildest setting it already makes the bucket 19 % worse; at the widest radius it
more than doubles it. There is no setting in the sweep where relink helps the metric it
was built to fix, and the trend says a wider sweep would only find more of the same.

## 2. Why, mechanically — read from what the ILP already does, not guessed

`motion_relink` re-solves each frame's assignment **locally**: one frame's cost matrix,
Hungarian or greedy, no knowledge of anything outside `t → t+1`. The ILP it is overriding
already solved the **global** problem — the same `edge_prob` the relink cost function
uses, plus appearance, disappearance, and division weights, jointly, with flow-conservation
constraints across the whole graph (`notes/03` §2, constraints 1–4: one parent only, flow
conservation, division only if selected). Local re-optimization throws that consistency
away. It cannot know that moving one edge to look locally cheaper forces an inconsistency
the ILP's constraints would have priced in elsewhere.

**This predicts exactly the observed shape.** A correct, ILP-approved edge sits at a cost
that is *locally* comparable to several nearby wrong candidates — real cells are ~8 µm
apart (`notes/04` §4) inside a 5.9–9.8 µm search radius, so genuine neighbours are common,
not rare. `motion_relink`'s cost function has no term preferring "what the ILP already
decided," so ties and near-ties resolve by local geometry alone and flip good edges as
often as bad ones. Widening the budget gives it more opportunities to do this, which is
why damage is monotonic in `max_change_frac`.

## 3. Prediction 3 "passed," and that is worth being precise about

```
geom_vel   -0.0424
prob0.78   -0.0409
prob2.0    -0.0400   <- best of the pure arms
```

`edge_prob` does make the relink **less harmful** than geometry alone — a real, measurable
effect, and technically what prediction 3 asked. But "least bad among uniformly bad" is
not the finding the prediction was written to detect. All three are deeply negative
against control. The honest reading is that `edge_prob` is not nothing, but it is nowhere
near enough to make local re-optimization competitive with the global solve it replaces.

## 4. The `+repair` arms that scored positive are not evidence relink works

```
control+repair    +0.0115   fn_mislink -62   (repair's own effect, no relink at all)
tightr+repair     +0.0127   fn_mislink  -8   (relink INSIDE the combo hurts mislink
                                               relative to repair alone, by 54)
```

`tightr+repair` edges out `control+repair` by +0.0012 overall — while its own contribution
to the bucket relink exists to fix is **negative** relative to repair alone. That is not a
mechanism working; it is noise or an interaction with gap-closing (relink frees some nodes
gap-closing then claims) riding on top of a real effect that has nothing to do with relink.
**Not a basis for a submission**, and none is proposed.

## 5. Verdict and what it closes

**Motion relink, in this local-reassignment form, is dead.** `notes/25`'s Run 2b — "put the
model's discarded `edge_prob` back to work" — is answered: the ILP was not discarding it
into a vacuum, it was using it inside a global solve that a local one cannot match.
`pipeline/relink.py` stays (tested, cheap, and the negative result required this exact
apparatus to demonstrate) but nothing built on it goes toward a submission.

## 6. What the same cache is actually good for

`claude_relink_sweep` wrote `cand_*.npz` with the model's candidate edges **and
probabilities** for all 24 datasets. That cache does not have to be spent on a local
override — it is exactly the input the ILP itself consumes. The natural next step is
**sweeping the ILP's own weights** (`edge_weight`, `appearance_weight`,
`disappearance_weight`, `division_weight` — currently `-1.0, 0.1, 0.1, 1.0`, the pack's
unexamined defaults) directly against this cache, which stays inside the global-consistency
regime that just won this comparison decisively, and costs nothing beyond re-solving a
cached ILP instance per setting.

Banked floor **0.752**. Current submission **0.880**. Cluster **0.913–0.916**.
