# hengck23, read in full: 43 posts, and three things we can check

Extracted every post by hengck23 (GRANDMASTER) from the scrape — **43 posts across 13
threads, 16 KB** — into `discussions/by_author/hengck23.md`. Ten of them are in
`723655 "simple idea: Your Affinity Field Tells Your Fate"`, a thread this project had never
opened.

Sorted by what we can act on.

---

## 1. A SECOND metric property, different from the patched one

`notes/54` ruled out the `augment_dataset` hack — hub node at `t=-1000`, fabricated forks —
because thread 736937 records that it was patched and the leaderboard recalculated. **This
is a different one**, posted 2026-07-16, and it is about duplication rather than injection:

> *"you make a graph. if you just repeat your tracks (giving new id) your `edge_jaccard` is
> not affected. this is the cause the kaggle metric don't penalize fragmentation,
> duplication. hence you can create multiple almost similar tracks to improve TP. But there
> is the node num correction (aka `adj_edge_jaccard`). But you can overcome this by
> duplicating/perturbing at the correct length and correct location."*

**Our own `notes/45` derived the mechanism independently from the scorer source:** predicted
structure matching no ground-truth node is **excluded** from the edge term rather than
penalised. Node matching is one-to-one within a frame, so a duplicated track cannot claim
the GT node its original already holds — its edges are excluded, not charged as FP. The only
cost is the node-count multiplier.

He also states the ceiling plainly (2026-07-15): *"lb more than 1.0 is possible."*

**Recorded, not adopted.** It is an exploit, the organisers have patched one already and are
actively watching this surface, and `notes/54` ruled that class out on the reasoning that
anything built on it can be invalidated before the 2026-09-29 deadline. Flagging it because
it is materially different from the one we examined and the user should know it exists.

## 2. The legitimate strategy he repeats across three months

His "winning formula" (2026-07-08), restated in August and September:

> *"generate dense tracks for training. (1) 3d point is easy to generate (e.g. cellpose).
> (2) short track (2 or 3 frame) is easy to generate — ultrack, rule-based heuristics, or an
> open source tracker. If 1 and 2 get good results, ILP will generate the long tracks…
> currently we have **less than 1% link labelled** and the rest are unlabelled."*

> *"Once you have dense segmentation label, you can use many opensource trackers — hoct,
> itec, trackastra — to make dense tracks for better training. Then you can do longer range
> tracking over a window of 5 or 8 instead of 2."*

This is `notes/53`'s dense-pseudo-label route, arrived at independently and held
consistently. His diagnosis of *why* is the sharpest statement of the sparse-label problem
anyone has made: **"the unlabelled nodes change from noisy targets to negative targets after
10 epochs."**

And one concrete, cheap fix for it (2026-07-10) that needs no external model:

> *"instead of a classification binary problem, reformulate as **learnable peak detection**.
> At each pixel location after the unet logit head, loss = softmax of pixel over its
> neighbours. Those pixels without annotation are **not computed in the loss at all**."*

That is directly aimed at the mechanism `notes/21` measured — DoG sits **+26% above** the
temporal-independence bound because a fixed filter responds identically to the same
structure, while the learned UNet sits below it. A peak-ranking loss makes the network
behave more like the filter. Relevant only if we train, which Track B does not.

## 3. Ground-truth quality warnings we have never accounted for

Three separate posts, and this project has assumed clean GT throughout:

- **Frozen frames.** *"beware of jumps in ground truth track… all freeze after the same
  frame indices — this means they are cropped from the same master big volume."* If GT
  tracks freeze after some index, edges past it are not real motion.
- **Interpolated annotations.** On very dim nodes: *"It can be a false positive or the result
  of interpolation. E.g. annotation labels frame t=1 and t=3 and interpolates for t=2."*
- **He opened a thread titled "not all sparse GT edge are correct."**

`notes/57` just showed what happens when we take a stated constant on trust; these are the
same shape of risk pointing at the labels themselves.

## 4. The one that is checkable right now

He posted GT link-displacement statistics (2026-07-07) from 63,751 links:

```
percentiles [50, 90, 95, 99]      dz [1, 2, 2, 4]   dy [0.5, 1.25, 1.75, 3.0]
                                  dx [0.5, 1.5, 2.0, 3.75]
min dz -37, max +35   ###???      <- his own flag on the extremes
```

**This is the exact analogue of the check that just paid off.** `notes/57` found our
division gates rejecting 88% of real divisions because a constant was adopted from a source
and never checked. The same class of constant governs linking:

```
pipeline/repair.py   close_gaps       max_um = 5.75
                     cap_edge_length  max_um = 14.0
                     linefit_smooth   max_shift_um = 3.2
```

None has been measured against the GT link-displacement distribution. `claude_linkgeom`
does that — same CPU pattern as `claude_divgeom`, same subprocess structure, and it either
confirms the gates or finds a second 88%.

## 5. Priority

1. **`claude_linkgeom`** — cheap, and the identical check found a real error one run ago.
2. Track B, unchanged (`notes/54`): the three-model port.
3. GT quality audit (frozen frames, interpolated nodes) — cheap, and it bears on every
   local measurement this project has made.
4. Dense pseudo-labels — his formula and `notes/53`'s, still the highest ceiling and still
   the wrong order for a 0.938 target.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
```
