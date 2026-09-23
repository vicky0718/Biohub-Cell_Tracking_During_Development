# `geofus` settles `tight60` against itself, and hands us a lever we did not have

2026-09-23, 02:52. `claude-arm-geofus` — an unmodified fork of `amanatar/biohub-geometric-fusion`
(120 votes, author at 0.948) — finished after four hours. It ran a **24-candidate** sweep, which
is the most thoroughly explored post-process map anyone has published on this lineage.

## 1. Their own sweep reverts the radius we could not explain

`amanatar`'s notebook deletes the `BIOHUB_MOTION_RELINK_TIGHT_UM = "5.5"` line so the code
default **6.0** applies — which is why `notes/89` counted them as one of four independent
sources for `tight60`. Their sweep's verdict on that base:

```
base (radius 6.0)        adj 0.9260
tight55  (radius 5.5)    adj 0.9280    +0.0021
```

Ours, measured the other way round on the `beraterolelk` base:

```
base (radius 5.5)        adj 0.9280
tight60  (radius 6.0)    adj 0.9260    -0.0020
```

**Two independent implementations, two sweeps, the same number.** The knob is settled, our
validator was not misbehaving, and the "four authors above the plateau all ship 6.0" reading
in `notes/89` §5 was wrong in a specific way worth naming: **one of the four ships it as a
*base* that their own sweep overrides.** A default is not a choice. I read the config and
should have read the sweep.

## 2. `MOTION_RELINK_VELOCITY_WEIGHT` is live, and it is the bonus mechanism from the other side

The new thing in their map:

```
vel075   (W = 0.75)   adj 0.9253   -0.0007
base     (W = 0.50)   adj 0.9260        —
vel025   (W = 0.25)   adj 0.9274   +0.0014
```

Monotone, and it is in **our** base too (`MOTION_RELINK_VELOCITY_WEIGHT 0.5`, config dump).

It is the same lever as the learned bonus, approached from the opposite end. The relink
predicts `pos + W * (pos - prev)` and charges `|target - predicted|`, so **lowering W shrinks
the motion extrapolation exactly as raising BONUS lets the probability overrule it.** Both say
"trust the geometry less". Whether they are additive or redundant is the whole question, and
`lbsweep` answers it for free — the notebook builds a `combo(...)` of every candidate that
clears the margin and scores that too.

`lbsweep`'s ladder is revised accordingly: `t60` dropped (settled, twice), `b35` dropped (past
saturation), `vel025` and `vel010` added.

```
base       MOTION_RELINK_LEARNED_BONUS 5.0        <- lb50, best measured
b8/b12/b20 bonus 8 / 12 / 20
vel025     MOTION_RELINK_VELOCITY_WEIGHT 0.25
vel010     MOTION_RELINK_VELOCITY_WEIGHT 0.10
relaxed12  MOTION_RELINK_RELAXED_UM 12.0
```

## 3. Four more knobs closed, for free

```
diverge150   -0.0062    div fp 1 -> 6.  The divergence gate loosened is actively bad.
minlen5      -0.0012    min track length 5. Same direction as our mtl4 (-0.002 on the board).
dcgap018     -0.0002    DeepCenter gap threshold 0.18.
sym075, divwide, dcsd015, dcsd010, rescue085, gap45, gap2step40, reuse28, bonus125,
dcgap035                 all 0.0000 alone. Inert.
```

`divwide` (parent 11um / sister 16um / existing-child 12um) and `dcsd015`/`dcsd010` are inert
*alone* and appear in the selected combo only because the combo takes every candidate that
cleared the margin — they contribute nothing on their own evidence. That is worth saying
because a reader of the combo label would otherwise conclude the division gates matter here.

## 4. `geofus` itself is not the submission

Its combo reaches `adj 0.9300`. Put on a common scale — both notebooks at radius 5.5 —
that is **+0.0020** over the shared base, against **`lb50`'s +0.0045**. The bonus is the
bigger lever and it is on our own base.

Two further reasons not to spend a slot on it: `LEAF_PRUNE_MIN_EDGE_PROB`, the one genuinely
new *code* in their stack (a "weak-leaf pruning" stage we do not have, worth +0.0004 on top
of `tight55`), only exists inside their notebook; and their run took **four hours
interactively**, of which ~3 h is the 24-candidate sweep. The sweep cost is constant under a
graded rerun while inference scales 17x, which puts their graded run near 10 h of a 12 h box.
`ftune` died that way (`notes/86`). `lbsweep` runs 6 candidates plus one combo — about 50
minutes of sweep — and stays well inside it.

## 5. Standing position

```
arm        change                       Δadj vs pub947bera base
lb50       learned bonus 5.0            +0.0045   <- submit
lb30       learned bonus 3.0            +0.0015
geofus     amanatar combo, 24-cand      +0.0020   (different base, 4 h runtime)
tight60    relink radius 6.0            -0.0020   settled by their sweep too
```

GPU quota has been refusing every push since 22:55 the previous evening.
`tools/wait_for_quota.py` probes every twenty minutes and will launch `lbsweep` first.
