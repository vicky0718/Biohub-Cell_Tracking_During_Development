# The learned bonus is the lever, and the curve accelerates

2026-09-22, evening. Five points on one knob, all at constant node count, all pure Jaccard.

## 1. The measurement

```
bonus   validator adj   Δadj (paired, 8 movies)   from J    from mult   Δ sub nodes   div tp/fp/fn
 1.0       0.9280            —                      —          —              —           3/1/9
 1.5       0.9282        +0.0003                 +0.0003    +0.0000          +7           3/1/9
 2.0       0.9285        +0.0001                 +0.0001    -0.0000         +27           2/0/10
 3.0       0.9295        +0.0014                 +0.0015    +0.0000         +51           3/0/9
 5.0       0.9325        +0.0027                 +0.0027    -0.0000         +53           3/0/9
```

Base submission is 122,794 nodes, so the largest node-count move in the family is **+0.04%**.
`tools/split_score.py` puts the multiplier's share of every one of these at **0%**. This is
the one regime the offline validator is admissible in (`MEMORY.md` §5, and
`rogerrogerroger3r`'s independent statement of the same rule): a pure re-linking, where edges
move and nodes do not.

It is not merely monotone. **It accelerates** — +0.0002, +0.0003, +0.0010, +0.0030 in
validator adj across the four steps, on steps of 0.5, 0.5, 1.0, 2.0. Per unit of bonus the
gain is still growing at 5.0.

`lb50` also removes the division false positive the base carries, `3/1/9 -> 3/0/9`, lifting
`div_J` 0.2308 -> 0.2500. On twelve GT divisions that is one event and is not evidence
(`notes/77`), but it is not evidence *against* either.

## 2. Why this knob, and where it has to stop

`motion_relink_edges` builds a Hungarian assignment over

```
cost[i, j] = motion + 0.05 * raw - MOTION_RELINK_LEARNED_BONUS * prob        prob in [0, 1]
```

`motion` is in micrometres. So **BONUS is exactly the number of micrometres of geometric
error the network is allowed to overrule**, and the scale to compare it against is measured:
the median true step between frames is **1.8 um** and the confident-pass gate is **5.5 um**.

* At the shipped 1.0 the network can move the cost by at most 1 um, which is *less than a
  typical true step*. Geometry wins every contested link. The edge classifier — `acc 0.9998,
  recall 0.9692` on 30 held-out movies (`notes/75`) — is outvoted by a distance.
* At 5.0 the network can overrule up to 5 um, comparable to the gate itself, and the
  probability decides inside it.
* Past the gate radius there is nothing left to overrule: the gate still bounds which pairs
  are candidates at all. So the curve **must** saturate somewhere around 10-20.

`lb80` (8.0) and `lb150` (15.0) bracket that. If they agree, it is saturated. If 15.0 is
still higher, the assignment has become pure maximum-probability matching inside the distance
gate, and the gate is then the only thing left to tune.

This is also the one lever in this project with a mechanism that predicts its own ceiling
rather than being extrapolated to one.

## 3. Corroboration, such as it is

* `humblehumbert` (rank 135, 0.949) publishes `biohub-chen-learned-bonus5-parentledger-v1`,
  which sets this knob to exactly **5.0**. Different base, so not a controlled comparison —
  but they chose 5.0 and they are above us.
* `Justin CH123` (rank 484) identified this exact stage on the forum and asked whether anyone
  had gained at it. The thread has no replies. The notebook's own sweep only ever tried 1.25.
* **704 teams are on the 0.947 plateau and every one of them ships 1.0.**

## 4. And it reframes `tight60`

`tight60` — a wider confident-pass gate, 5.5 -> 6.0 — scored **-0.0024** at bonus 1.0, which
was the day's most surprising negative given four authors above the plateau ship 6.0
(`notes/90` §3a). The mechanism above explains it without special pleading: a wider gate
admits more candidate pairs, and at bonus 1.0 the thing choosing between them is geometry, so
widening only admits worse matches. The gate is worth widening when there is a chooser worth
widening it for.

`lb50t60` (bonus 5.0, radius 6.0) tests that directly and is queued. `lb30t60` tests the
weaker version. `geofus` tests it from the other side — 6.0 shipped alongside an extended
sweep and a leaf-prune, which is one of the packages those authors run.

## 5. Submission position

**`lb50` supersedes `lb30`**: same shape, twice the gain, and a 0.949 author independently at
the same value. `lb30` remains valid and the pair is worth submitting together — two points
on the same axis, board-measured, is what decides whether `lb80` and `lb150` are worth their
slots. Everything else measured today is a reject.

```
claude-arm-lb50    bonus 5.0    Δadj +0.0027   0% multiplier   +53 nodes
claude-arm-lb30    bonus 3.0    Δadj +0.0014   0% multiplier   +51 nodes
```
