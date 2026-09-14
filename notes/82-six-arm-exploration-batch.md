# Six-arm exploration batch — built, verified live, ranked only on the board

`notes/81` closed offline measurement on training movies: not underpowered, **biased**, with a
predictable sign. So this batch carries no offline score by design. Each arm is one knob on the
`ttasec` base (LB 0.945), moved in a direction no public notebook and no arm here has tried.

## 1. All six verified non-inert

`notes/68`'s rule is that an inert arm is indistinguishable from a working one unless you check.
Each arm's own resolved-config dump, against the `ttasec` control, plus its output size:

```
arm          nodes   d_nodes    edges   d_edges   config dump
ttasec     122,735         -  118,435         -   (control)
mtl4       126,014    +3,279  120,982    +2,547   output_min_track_len: 4
sdw90      121,714    -1,021  117,415    -1,020   secondary_detection_weight: 0.9
mrr12      123,022      +287  118,896      +461   motion_relink_relaxed_um: 12.0
gapdc15    122,799       +64  118,520       +85   deepcenter_gap_threshold: 0.15
gap65      122,778       +43  118,491       +56   gap_close_um: 6.5
bew10      122,618      -117  118,317      -118   bidirectional_primary_weight: 0.1
```

Every one moves the output. `bew10` reports under a **different key** in the dump
(`bidirectional_primary_weight`, not `bidirectional_edge_weight`), which is why a first check
came back empty — the knob is live, the name simply differs from the env var's.

## 2. Two infrastructure findings from the failures

**`BIDIRECTIONAL_EDGE_WEIGHT` is pinned twice.** Besides `_EXPECTED_NUMERIC` — the drift guard
that killed `gap44` — the base carries a free-standing assertion elsewhere in the same cell:

```python
if not _bidirectional_math.isclose(_bidirectional_weight_guard, 0.15, rel_tol=0.0, abs_tol=1e-12):
    raise ValueError({'expected_bidirectional_weight': 0.15, ...})
```

Moving this knob takes **four** edits: env var, guard dict, `isclose` threshold, error message.
It is the only parameter in the notebook the author pinned twice.

**A 429 is not a verdict.** `sdw90` was reported as failed because `run_arm.py` let an HTTP 429
propagate out of a status poll and kill the loop, while the kernel ran on normally to
completion. Same class as the "6 consecutive P100 draws" message `notes`/`run_arm` fixed
earlier today: a transport error surfacing as a claim about the work. The poll in `run()`'s
settle loop is wrapped; the one inside `wait_for_run` is not.

## 3. Submission order

Largest gated population first — where a knob has the most room to move anything:

1. `mtl4`     — short-track filter 6→4. **+3,279 nodes**, by far the largest effect.
2. `sdw90`    — secondary detection weight 0.80→0.90. −1,021 nodes.
3. `mrr12`    — motion-relink relaxed radius 10→12 µm. +287 nodes.
4. `gapdc15`  — DeepCenter gap acceptance 0.25→0.15. +64 nodes.
5. `gap65`    — gap radius 5.0→6.5 µm. +43 nodes.
6. `bew10`    — reverse-direction vote 0.15→0.10. −117 nodes.

Plus `ttasecw20` and `ttadse44`, complete since before this session and costing nothing.

**Expectation, stated in advance.** These are knobs, and this project's record is that knobs
land on the shelf — the single exception, `SECONDARY_EDGE_WEIGHT`, gave +0.001. Six tickets at
~+0.001 do not reach 0.947. What makes the batch worth its slots is that `mtl4` and `sdw90`
move output size by 1–3%, an order of magnitude more than the rest, so if any **stage** still
has slack those two will say so — and that is worth more than their own scores.
