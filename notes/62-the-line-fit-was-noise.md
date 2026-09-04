# The line fit was noise: a window mean beats it by +0.0030

`claude_static`, 24 cached instances, CPU. **4 of 4 predictions passed**, and the effect is
monotone all the way to the endpoint of the grid.

```
arm          total   adj_edge   edge_J    vs s0.0   nodes moved   mean shift
s0.0        0.9188     0.9072   0.9037         —            —            —
s1.8        0.9197     0.9082   0.9047    +0.0010       10.85%     0.062 um
s2.5        0.9206     0.9091   0.9056    +0.0018       14.84%     0.112 um
s3.5        0.9211     0.9095   0.9060    +0.0023       17.51%     0.158 um
s5.0        0.9212     0.9097   0.9062    +0.0024       18.84%     0.187 um
mean_only   0.9217     0.9102   0.9067    +0.0030       19.27%     0.199 um
```

Per embryo: **44b6 +0.0012, 6bba +0.0032** — both positive, which is the `notes/49` check
the last three levers all failed.

---

## 1. What it says

`linefit_smooth` pulls each node toward a local straight-line fit of its own track.
**Replacing that fit with a plain window mean is better, everywhere, monotonically.** The
fitted slope is not signal; it is noise, and smoothing along it moves nodes away from where
they should be.

`notes/26` and `notes/27` measured smoothing as the **major** half of the repair chain —
+0.0086 of +0.0113, against gap-closing's +0.0013 — and neither ever asked whether the
*line* was doing the work or the *averaging* was. It was the averaging.

Why the line was never needed: for constant-velocity motion over a symmetric window, the
window mean **is** the least-squares line evaluated at the centre. The two agree exactly
where the model holds, so the slope term only contributes where motion is non-linear or the
window is truncated — and there it is fitting jitter.

## 2. Where it came from, and the correction to my own hypothesis

`notes/59` measured **8.4% of GT links at exactly zero displacement** (10,772 of 128,883) —
frozen frames from crops of one master acquisition, plus interpolated annotation. I
predicted a *targeted* fix would pay: skip the fit on chains that are not moving.

**That hypothesis was too narrow.** `s0.6` and `s1.0` are slightly *negative* (0.9184,
0.9185); the gain only appears once the threshold is high enough to disable the slope
almost everywhere. The frozen-GT finding pointed at the right stage for the wrong reason.

## 3. A blind check, corrected

v1 graded "does the fallback fire?" by comparing **node counts** between arms. `static_um`
only *moves* nodes — it never adds or removes them — so `+0` was guaranteed by construction
and the check reported *"the fallback never fires"* while the score column was visibly
moving 0.9188 → 0.9197. v2 measures displacement against the `s0.0` anchor instead, and the
numbers above show it firing on 10.85–19.27% of nodes.

The lesson is narrow and worth keeping: **when a parameter is inherited from a derived
notebook, re-check that the graded quantity still responds to it.** The check came from
`gapum`, where the swept parameter genuinely did add nodes.

## 4. It transfers to the fork as one character

The fork runs the identical mechanism in `linefit_smooth_output_graph`:

```python
fitted = np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0)
```

Degree **1** is the line. Degree **0** is the window mean — exactly the `mean_only` arm.
`claude_fork3` makes that single change, verified by asserting the original line matched
once, that exactly one executable `np.polyfit` call site exists, and that it is now degree 0.

**+0.0030 is precisely the 0.937 → 0.940 gap to rank 100** on the live 2026-09-04 board. It
is screened on the fork's own PROXY_SCORE (0.9266 unmodified) before any submission, and
`notes/49` stands: PROXY runs on training embryos and the test set is a third pair, so it
screens direction, not magnitude.

## 5. Why this one is different from the last three

`notes/60`'s rule: a constant already tuned on the metric is near its optimum on the metric.
`close_gaps`' radius, the fork's division gates, and `claude_topk`'s threshold all refused
to move because somebody had already swept them.

**Nobody swept this**, because it was never exposed as a parameter — the degree was a
literal inside a fitting call, in both codebases. It is the first lever in this stretch that
is ours rather than a re-tune of someone else's.

```
0.752 floor    0.901 our chain    0.937 fork (rank ~320)    0.940 = rank 100    0.965 top
```
