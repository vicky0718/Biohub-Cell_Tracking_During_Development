# The fork scored 0.937, and `notes/54`'s diagnosis was right

`claude_fork` — `nusrati/0-938` reproduced unmodified — scored **0.937**, against our own
pipeline's 0.901. **+0.036 in one run.**

That is the whole of `notes/54`'s claim, confirmed: the gap was never a modelling gap. Our
pack reproduction scored 0.867 while the cluster ran the *same weights* at 0.913–0.916
(`notes/25`), and everything built since had been clawing back ground the standard pipeline
already had. The cluster mounts three models — pack + `temporal-unet3d-seed314159` +
`deepcenter-unet3d-center-prior` — and we mounted one.

## Correcting my own rank estimate

I priced 0.937 at rank ~124 from a scrape taken the day before. **That was wrong.** Pulling
the live board (2026-09-04, top 500):

```
top 0.965   (was 0.963 on 2026-09-03)
0.937 -> rank ~320          rank 100 needs 0.940
0.940 -> rank ~ 86          rank  50 needs 0.943
0.945 -> rank ~ 27          rank 300 needs 0.938
```

The field moved overnight and a day-old scrape is not usable for ranking. **The gap to top
100 is +0.003, not +0.001.** Leaderboard figures in these notes should be re-pulled before
being used to decide anything, not carried forward.

## Where the +0.003 might come from

The fork's own validator (n=10 held out) reported:

```
adj_edge 0.9203    div_J 0.0625    PROXY 0.9266
```

`div_J` 0.0625 contributes 0.00625 of score. **`notes/57` measured its division gates as
still short of the data**, having checked all 151 real GT divisions across 199 datasets:

```
gate                       fork    GT p90   GT p95   GT max   fork rejects
SAFE_DIV_MAX_UM             9.0     10.05    11.78    13.53      19.9%
SAFE_DIV_SISTER_MAX_UM     14.0     14.36    15.34    20.30      12.6%
```

Their own comments justify 7→9 and 12→14 from stated GT statistics. Two of six check out —
sister median (10.57 vs 10.4) and the 12 µm rejection rate (33.8% vs 29%) — but the tail
does not: sisters reach **20.30 µm**, not the 13.7 they report, and 7 µm rejects **53.6%** of
parent-daughter links rather than 25%. Their direction is right; their destination is short.

Moving `div_J` from 0.0625 to ~0.09 is worth **+0.003** — exactly the gap.

`claude_fork2` makes one coherent change: both distance gates to the measured p95. The
precision gates that filter what they admit (`SISTER_SYMMETRY_TAU` 0.6, `DIVERGE_UM` 4.5,
the DeepCenter safe-div veto) are untouched, which is what makes widening the distance side
reasonable rather than reckless.

**The way it fails is worth pre-stating.** `close_gaps` had insertion caps that could have
bound before its radius (`notes/60`); the same shape exists here as `FRAME_FRAC_CAP` 0.0076
and `GLOBAL_FRAC_CAP` 0.00375. If those bind, this is a no-op and the gates were never the
constraint.

## Standing

```
0.752 floor    0.901 our pipeline    0.937 fork (rank ~320)    0.940 = rank 100    0.965 top
```

Track A delivered the floor it was meant to. Track B — porting the three-model configuration
into our own pipeline — is no longer about reaching the cluster; it is about beating it,
and `notes/55` recorded the one place we already do: our `div_J` is 0.1154 against their
0.0625, on different samples but in the direction this run is pushing.
