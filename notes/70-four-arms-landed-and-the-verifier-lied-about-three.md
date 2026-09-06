# Four arms landed, the verifier said three had not, and the 0.948 config will not run

## 1. The tool I built to catch `notes/68`'s failure produced three of its own

`tools/verify_arm.py` exists so that no arm reaches a submission slot without evidence its
edit reached the run — `notes/68`'s inert checkpoint being the thing it was built for. Its
first output declared `dc40`, `div15` and `sew20` all dead. All three were fine.

```
dc40    want 0.40   resolved 0.4                   "DID NOT LAND"   <- string compare
div15   want 1.5    resolved None                  "DID NOT LAND"   <- key absent
sew20   want 0.20   resolved 0.15                  "DID NOT LAND"   <- read the wrong block
```

Two bugs, and the second is the interesting one. `notes/68` established that these notebooks
print their configuration twice — a resolved block early, a defaults block after
`Wrote /kaggle/working/run_stats.csv` — and I wrote `resolved()` to take the *first* match
so it could not read the wrong one. **But the resolved block does not carry every
variable.** `safe_div_diverge_um` appears nowhere in it; `secondary_edge_weight` and
`secondary_link_mode` appear *only* in the defaults block. For those keys "first match" and
"the lying block" are the same line.

Fixed twice: numeric comparison, and the search now truncates at the `run_stats.csv` marker
so a key the resolved block omits comes back `None` — **inconclusive, read the counters** —
rather than being answered from the block that lies.

The general shape is worth keeping: *a check that reads a report can only be as honest as
the report.* `run_stats.csv` counts what the pipeline actually did, and it is what settled
all four of these.

## 2. What the four arms actually did

Summed over the four verification clips, against `lb941`:

```
arm      change                                nodes    edges   divisions   other
union    LINK_MODE -> adaptive               +3,333   +3,270          +1   raw_edges +3,521
div15    DIVERGE_UM   2.25 -> 1.5               +30      +71         +48   candidates 677 -> 1,238
dc40     DC_SAFE_DIV  0.25 -> 0.40               -4      -65         -62   accepted 222 -> 61
sew20    SEW          0.15 -> 0.20             +101      +95          +1   raw_edges +135
```

All four landed. Three notes on the shape of them:

* **`union` is by far the largest change** — 2.8% more nodes and 3.1% more edges. That is
  not a knob nudge, it is a different linker resolution rule. It also moves the node-budget
  term: on the placeholder clips the base sits 11.9% under `N_est`, so more nodes eats into
  a bonus that is currently being collected.
* **`div15` and `dc40` push divisions in opposite directions** — +51% and −66% from the same
  baseline of 94. One of them is wrong, and running both is how that gets settled rather
  than argued. `safe_division_skipped_cap` stays 0 in both, so `notes/60`'s volume-cap trap
  is not masking either.
* **`sew20` is the smallest mechanical change** and carries the strongest external claim.
  Those two facts pull in opposite directions and the leaderboard is the only thing that
  resolves it.

## 3. `sew948` cannot run here, and it says something about the 0.948 claims

The published 0.948 configuration, forked unmodified, dies in its own integrity check:

```
RuntimeError: DeepCenter checkpoint checksum mismatch:
  expected 8040999a92f6b7bbd98fa8cf458141e045c0f9ad7c936bdb3b18e1f7edafe2a0
  got      8164d1ffa07f87e0506027a0392edeab7939a32bd5e3f756377c0d72885cf127
```

The expected hash is `best.pt`'s — `claude-arm-ckpt948`'s log printed exactly
`DeepCenter materialized SHA256: 8040999a…` while loading `best.pt`. The hash it got is a
different file, i.e. here the `checkpoint_last.pt` path **did** resolve, and the notebook's
own guard rejected it.

So the published 0.948 config is internally inconsistent: its env var names
`checkpoint_last.pt` while its checksum guard expects `best.pt`. It can only run where the
path fails to resolve and the fallback loads `best.pt` — which is precisely what happened in
`ckpt948` (`notes/68` §2). **Every 0.948 kernel that ran, ran on `best.pt`.**

That leaves `SECONDARY_EDGE_WEIGHT 0.20` plus the narrow division config as the whole of
their difference from the 0.941 line, and `sew20` is the clean test of the first half of it.
`sew948` stays failed; there is nothing to fix, the configuration contradicts itself.

## 4. Submission order

`lb941` first — the floor, already handed over. Then, most-evidence-first:

```
1  union    +3,333 nodes. Crosses two configurations that each score 0.941 alone.
2  sew20    the one parameter three 0.948-claiming kernels share, on the good base.
3  div15    the gate that does 82% of division filtering, one step down the gradient
            that paid (4.5 -> 2.25 -> 1.5).
4  dc40     the author's own published f1 maximum, and the other side of div15's A/B.
```

Still queued and unrun: `adaptive`, `gap44`, `dse44`, `dse52`, `det960`, `dcgap40`.
