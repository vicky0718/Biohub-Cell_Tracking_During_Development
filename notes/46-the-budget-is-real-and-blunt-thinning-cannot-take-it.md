# The budget regime is reachable. Uniform thinning cannot take it.

`claude_budget`: `pool_kernel_um` ∈ {3, 6, 10, 15, 22} µm at a fixed `det_threshold`,
36 datasets, components reported apart. **Predictions 1–4 pass, 5 fails**, and the four
that passed are what make the failure worth reading.

```
pool µm      score  adj_edge   edge_J  node_rec      nodes
3.0         0.9356    0.9264   0.9241    0.9827    700,216      <- today's default
6.0         0.9390    0.9304   0.9247    0.9790    659,535      <- best
10.0        0.8860    0.8821   0.8661    0.9367    519,492
15.0        0.7569    0.7551   0.7295    0.8113    325,514
22.0        0.5064    0.5064   0.4756    0.5374     95,137
```

---

## 1. The lever exists, and `notes/44`'s plateau really was a plateau

```
2. node count falls more than 5x        700,216 -> 95,137 = 7.4x       PASS
3. total_node_ratio below -0.5          lowest -0.647, multiplier 1.065  PASS
4. adj_edge_jaccard is non-monotonic    peak at index 1 of 4             PASS
```

`det_threshold` moves the node count by 5.6% across its entire usable range. The NMS radius
moves it **7.4×** and carries the multiplier from 1.0025 to 1.0647. So `notes/45` was right
that every previous sweep lived inside a regime where the multiplier is pinned near 1.0 by
construction, and `notes/44`'s flat surface was flat for that reason.

*(The `ratio` and `mult` columns are derived from `adj/edge_J`. `adj_edge_jaccard` is a
weight-averaged per-dataset value rather than a clean product, so treat those two columns
as indicative; `score`, `edge_J`, `node_recall` and `nodes` are measured.)*

## 2. And the trade is hopeless at a ratio of about seven to one

```
5. best arm beats the incumbent by >0.01
   p6.0_m6_g2 − p3.0_m6_g2 = +0.0016   SE 0.0037   t 0.42   n=36 resolves ~0.0074   FAIL
```

Across the full grid the multiplier gains **+0.062** while `edge_J` loses **0.448**. The
reason is in the `node_rec` column, which falls **0.983 → 0.537**: a wider suppression
radius deletes ground-truth-matched cells at the same rate as everything else. Thinning
uniformly is not selection.

Even the nominal winner is unproven — `pool 6.0` beats the default by +0.0016 at t=0.42,
well inside the noise. There is no free +0.004 hiding here either.

## 3. Which is exactly what r35 warned about, in its own words

> *"By default no density-model cap: Spotiflow emits hundreds of candidates per frame, and
> TinyUNet sparse hybrid (6–10/frame) ranked by intensity yields **0 recall** even with
> correct axes."*

That sentence is this table. Cap a detector that cannot rank annotated cells above
unannotated ones, and recall goes to zero before the multiplier repays anything.

So the budget direction survives, with its requirement now measured rather than argued:
**the trim needs a detector whose ranking correlates with being annotated.** The arithmetic
that motivated it is unchanged — a predicted node matching no ground-truth node contributes
nothing to the edge term and costs budget — but "cut nodes" was never the recipe. "Cut the
*right* nodes" is, and nothing we own does that.

## 4. Getting Spotiflow to run cost five launches, four of them my own carelessness

Prediction 1 now passes — `spotiflow 0.6.2`, `torch 2.5.1+cu121`, `torchvision
0.20.1+cu121`, GPU usable. The path there:

1. `pip install spotiflow` → `ResolutionImpossible`: the image ships
   `torchvision 0.25.0+cu128` and spotiflow pins an incompatible one.
2. `--no-deps` on spotiflow alone installed it and left `lightning` missing. r35's 45
   wheels are a complete offline closure, so the fix is to pass **every** wheel with
   `--no-deps` and let pip install rather than resolve. Only `zarr` overlaps the pack's
   stack, and it is excluded.
3. Then `spotiflow → lightning → torchmetrics → functional.image.arniqa → torchvision`,
   and the image's torchvision is built against a different torch, so its C extension
   fails: *"partially initialized module 'torchvision' has no attribute 'extension'"*.
   Fixed by installing `torchvision==0.20.1`, the build that pairs with torch 2.5.1.
   **A submission notebook runs with internet off, so this wheel has to be published to a
   wheelhouse first — exactly as torch itself was.**
4. `ModuleNotFoundError: biohub_tracking` — I omitted `PACK/repo/src` from `sys.path`.
5. `ImportError: cannot import name 'USERNAME' from 'dataspec'` — my synthetic stub had
   two attributes where the working one has five.

Items 4 and 5 are the same mistake twice: I retyped a known-good preamble from memory
instead of copying it. The working copy in `_build_claude_widecv.py` literally carries the
comment *"copied from there rather than retyped, which is how v1 of this notebook came to
import a module name that does not exist"* — written by me, ignored by me. The build step
now diffs the `dataspec` attributes and `PACK` path entries against that notebook and
fails if they are not a superset, so the class is closed rather than the instance.

## 5. Where this leaves it

```
0.901 submitted    0.926 bronze    0.944 gold
measurable > 0.0015     worth a slot > 0.01
```

Closed: uniform NMS thinning (+0.0016, t=0.42).
Open and now the only question: does r35's Spotiflow rank annotated cells well enough that
a trim keeps them? `claude_spotiflow` measures recall per node against our own detector on
the same frames. Prediction 4 there — recall per thousand nodes at least 3× ours — is the
one that decides whether the +0.06 the multiplier is worth can actually be collected.
