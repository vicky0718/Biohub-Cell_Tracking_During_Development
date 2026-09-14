# `norelink` scored 0.944. The harness said +0.0337. It is biased, not merely noisy.

```
                        offline harness      leaderboard
ttasec                          0.95161            0.945
norelink                        0.98536            0.944
predicted delta                 +0.0337
actual delta                                      -0.001
```

The submitted version carried the change (`os.environ['BIOHUB_OUTPUT_MOTION_RELINK'] = '0'`
present in `claude-arm-norelink` v2 source, and the score differs from `ttasec`'s), so this is
not an inert-arm artifact. **The offline harness was wrong by 0.035.**

## 1. What I claimed, and why the argument was bad

`notes/80` argued this result was trustworthy *despite* `notes/77` having closed the harness
for ranking, on the grounds that it was a direct count rather than an aggregate difference:

> "`edge_fp` fell by 72 and `edge_fn` by 50 against a shared baseline … 12 of 12 movies
> improved … twenty times the 3-edge noise floor."

Every one of those statements is still true of the harness's data. The inference from them was
wrong, because they all address **variance** and the error was **bias**. A systematically
shifted instrument reports the shift unanimously across all twelve movies and with a large
t-statistic. Unanimity is what bias looks like. I read it as what precision looks like.

`notes/77` closed offline ranking *on a power argument*. I then found a large effect, checked
that it cleared the power bar, and treated the bar as the only obstacle. It was not.

## 2. The mechanism, which is specific and predictable

The harness scores a checkpoint **trained on all 199 training movies** against 12 of those same
movies (`notes/72` §3). I dismissed the contamination with: *"arms differ only in
post-processing, so the optimism is identical across arms."* That is false whenever an arm
changes **how much the pipeline trusts the model**.

`motion_relink_edges` overrides the model's edge decisions with a geometric assignment. It is
an **error-correcting heuristic**: it pays when the model is wrong and costs when the model is
right. On movies the model has memorised there is almost no model error to correct, so removing
the correction looks like a large gain. On genuinely unseen data the model errs more often and
the correction earns its keep — worth about +0.001, which is what the board measured.

**So the harness has a predictable sign:** it over-rewards trusting the model and under-rewards
correcting it. That retro-explains the whole run of failures on contaminated data:

* `harness/purescore.py`, 0-for-4 — the four visible clips are byte-identical copies of
  training clips (`MEMORY.md`), so identically contaminated.
* `ttaz16` over `ttasec` — both are TTA, i.e. model-trust changes, in the same direction.
* `norelink` — heuristic removal, maximally exposed, and wrong by the largest margin yet.

## 3. What this closes

**All offline measurement on training movies is closed for this project**, not merely ranking
at 0.003 scale. `notes/77`'s conclusion was too narrow: more movies would not have fixed this,
because n does not touch bias. The instrument cannot be repaired by sampling.

The direct counts remain usable for **mechanism** questions that do not compare arms —
"does this knob change the division count", "do any ILP divisions survive" — which is how
`notes/78` and `notes/79` were actually used, and those findings stand. What does not survive
is any use of the harness to say one arm is better than another.

`nrmtl3` and `nrsew20` were built and deliberately not run pending this result. They are now
dead: both are `norelink` plus a knob, on a base that loses. `nrdc` is also void as a ranking,
though its mechanism finding (loosening the veto adds FPs and no TPs, on two bases) stands.

## 4. Cost and standing

One submission slot, spent on my recommendation, and about five GPU-hours of evals. The
leaderboard reading is itself worth something — the relink is worth **+0.001** on real data,
which nobody had measured — but that is not what it was spent for.

Standing position is unchanged from before this session: **`ttasec` at 0.945 is our best arm**,
rank ~648 of 3,460, and the only instrument this project has that has never lied is the
leaderboard. `claude-arm-ttasecw20` and `claude-arm-ttadse44` remain complete and unsubmitted;
they cost no GPU and are real measurements.
