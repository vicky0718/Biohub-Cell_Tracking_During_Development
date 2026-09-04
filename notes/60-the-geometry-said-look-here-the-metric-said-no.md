# The geometry said look here; the metric said no

`claude_gapum`, 24 cached instances, 5 gap radii, CPU. **The shipped 5.75 µm is the best
value in the grid, and every widening is monotonically worse.**

```
arm       total  adj_edge   edge_J   div_J    nodes    ratio   added   forks
g5.75    0.9188    0.9072   0.9037  0.1154   18,032   -0.129      +0   1,443
g8.0     0.9173    0.9057   0.9031  0.1154   18,265   -0.119    +233   1,443
g10.7    0.9170    0.9055   0.9035  0.1154   18,382   -0.113    +349   1,443
g14.0    0.9142    0.9027   0.9012  0.1154   18,472   -0.108    +439   1,443
g20.0    0.9097    0.8981   0.8972  0.1154   18,564   -0.102    +532   1,443
```

Reproduction exact (0.9188, `div_J` 0.1154, 1,443 forks). The radius genuinely does insert
more nodes — +233 to +532 — so `max_added_frac` and `max_added_abs` are **not** binding
first; prediction 2 passed and the radius really was the free variable. It just does not
help. Both embryos agree on the sign (−0.0034 and −0.0003), so the loss is real rather than
noise.

**`notes/59` §1 is withdrawn.** Its claim that 5.75 µm is "too tight by a wide margin, and
the recommended value is ≥ 10.7" does not survive measurement. The caveat I attached to it
turned out to be the whole story: a two-frame span was approximated as **2× a single-frame
step**, which assumes straight-line motion. Real cells wander, so the true `t → t+2`
displacement is well under twice a step, and 5.75 µm already covers it.

---

## What separates this from `notes/57`

Two geometric checks, one week apart, on the same kind of constant. One found a real error
and one did not, and the difference is worth keeping:

```
notes/57   pipeline/divisions.py   max_um 4.5, sister_max_um 6.8   rejects 88% of real divisions
notes/60   pipeline/repair.py      close_gaps max_um 5.75          already optimal on score
```

**`divisions.py` is not in the shipped chain.** Its gates were adopted from a public
notebook and never tuned against anything, so nothing had ever pushed them toward a good
value — and the probe `notes/25` planned was evaluated under gates that reject 88% of their
targets, which is why its null result means nothing.

**`close_gaps` is shipped, and has been swept by score repeatedly** (`notes/26`, `notes/27`,
`notes/34`, `notes/40`). A parameter that has been tuned on the metric is already near its
optimum *on the metric*, whatever the geometry says about it. Ground-truth geometry is a
good place to look for constants nobody has tuned; it is a poor second opinion on ones that
have been.

That is the rule this pair of runs buys, and it is cheap to apply: **before checking a
constant against geometry, ask whether score has ever been allowed to move it.** If yes,
expect the geometry to lose.

## What still stands from `notes/59`

The measurement itself, which was not about our parameters:

- **128,883 single-frame GT links, `dt` histogram `{1: 128883}`** — no multi-frame GT edges
  exist at all.
- **8.4% of GT links have exactly zero displacement** (10,772 of 128,883), confirming
  hengck23's frozen-frame and interpolated-annotation warnings at scale.
- **`cap_edge_length`'s 14.0 µm drops 0.10% of real links** — correctly set.
- Linking geometry is identical across embryos (medians 1.72 vs 1.82), unlike divisions.

The zero-displacement finding is the one with legs, and it is unaffected by this run:
`linefit_smooth` is the major half of the repair chain (+0.0086 of +0.0113, `notes/26`/`27`)
and it is fitting lines through positions that, for one link in twelve, are frozen by
annotation artefact rather than biology.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
close_gaps: CLOSED at its shipped value   |   divisions.py gates: still wrong, still unused
```
