# Our division gates reject 88% of real divisions, and their numbers understate the tail

`claude_divgeom`, CPU, all 199 training datasets. **151 binary divisions** — matching
`notes/43`'s independently-derived count of 151 exactly, which is the cross-check that makes
the rest readable. Every one is a single-frame event (`dt=1` for all 151).

```
                        n     min   median     p75     p90     p95     max
parent->daughter (max) 151    2.84    7.13    8.47   10.05   11.78   13.53
sister<->sister        151    2.87   10.57   12.68   14.36   15.34   20.30
```

---

## 1. Our gates reject almost every real division

`pipeline/divisions.py` defaults to `max_um=4.5`, `sister_max_um=6.8`, described in its own
docstring as *"the public notebook's published constants … the only externally-attested
operating point"*. Against the actual ground truth:

```
gate                                rejects   of    pct
max_um (parent->daughter) > 4.5         134   151   88.7%
sister_max_um             > 6.8         131   151   86.8%
```

**Both gates throw away roughly seven of every eight real divisions.** They were adopted
from a source, never checked against data we hold, and have sat there since.

This does not change the shipped chain — `insert_divisions` is **not** in it; our `div_J`
0.1154 comes from the ILP's own forks under `ratio0.4_2.0`. What it changes is the status of
the insertion direction: `notes/25` §4 planned `claude_div_probe` to sweep insertion, and no
note ever recorded its result. **Any evaluation of `insert_divisions` under these gates was
testing a function that rejects 88% of its targets by construction**, so a null result there
means nothing.

## 2. Their numbers are directionally right and understate the tail

`nusrati/0-938` justified 7→9 and 12→14 with stated GT statistics. Checked:

```
claim                              ours    theirs   verdict
sister median ~10.4um             10.57      10.4   MATCH
12um rejects ~29% of divisions    33.8%     29.0%   MATCH
parent-daughter max ~10.4um       13.53      10.4   DIFFERS
7um rejects ~25% of links         53.6%     25.0%   DIFFERS  (2x understated)
sister p90 ~13.0um                14.36      13.0   DIFFERS
sister max ~13.7um                20.30      13.7   DIFFERS  (badly)
```

Two of six match. The medians and the 12 µm rejection rate are right; **the tail is much
longer than they report** — sisters reach 20.3 µm, not 13.7. So their direction is correct
and their destination is short: even `0-938`'s loosened 9.0/14.0 still rejects **19.9%** of
parent-daughter links and **12.6%** of divisions.

## 3. One global gate cannot serve both embryos

```
44b6   n= 26   sister median  8.98
6bba   n=125   sister median 11.47      spread 2.49um
```

Prediction 4 failed. The embryos differ by more than a quarter of the median, and the test
set is a **third** pair (`notes/07` §3). A single global value tuned on these two is exactly
the shape `notes/49` warned about. The gate should scale with something per-dataset — cell
density or the frame's own nearest-neighbour distance — rather than being a constant.

## 4. What to use, with the n attached

```
RECOMMENDED (p99): max_um >= 14, sister_max_um >= 18
```

**Treat p99 as an upper bound, not a setting.** At n=151, the 99th percentile is the second
largest observation — one or two events. The defensible operating points are **p90:
`max_um` 10.05, `sister_max_um` 14.36**, or p95 (11.78 / 15.34) if recall matters more than
precision. `notes/43`'s arithmetic is the reason to care: `div_J = TP/(FP + D)` with `D`
fixed by ground truth, so a rejected real division is a permanent FN while a chargeable FP
costs the same denominator — the trade is not symmetric, and the gates should sit where the
real events are.

## 5. What follows

1. **Widen the gates in `pipeline/divisions.py`** to at least p90, and make them
   density-scaled rather than constant given §3.
2. **Re-run the insertion probe** that `notes/25` planned and no note ever reported. Its
   previous evaluation is void — the gates rejected 88% of the targets.
3. Fold into Track B: the 0.936 cluster's `SAFE_DIV` values are closer to right than ours
   but still short of the data, so this is one place our pipeline can beat theirs rather
   than merely match it.

```
0.752 floor    0.901 best (rank ~1388/3038)    0.938 = rank 100    0.947 gold
n = 151 divisions across 199 datasets -- every figure above rests on that
```
