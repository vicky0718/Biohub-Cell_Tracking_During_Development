# Two independent relink designs, from opposite ends — and neither has an operating point

Two relink attempts now exist, built independently and with opposite risk postures. Read
together they close local relink far more firmly than either does alone.

| | `claude_relink_sweep` (mine) | `relink_probability_scored` (the user's) |
|---|---|---|
| policy | re-solve the whole frame assignment | **local two-pair swaps only** |
| gating | none — cheapest assignment wins | **margin 0.35 + probability margin 0.1** |
| budget | `max_change_frac` 0.03 – 0.50 | `max_swaps_fraction` **0.008** |
| divisions | forks protected | division sources protected |
| **result** | **every setting worse**, monotonically | **0 swaps, on all 4 datasets** |

**Loose enough to fire ⇒ net harmful. Strict enough to be safe ⇒ never fires.** That is a
bracket, not two separate disappointments, and it is stronger evidence than `notes/29`
had on its own.

It also independently confirms `notes/29`'s mechanism. I argued there that a local
re-optimizer flips good edges as often as bad because at real cell spacing (~8 µm inside a
5.9–9.8 µm radius) a correct ILP edge is *locally* cost-comparable to several wrong
neighbours, and nothing in a local cost function prefers "what the global solve already
decided". The user's design adds exactly the missing term — a **margin** requirement, which
is a direct implementation of "only move when clearly better". Its answer is that at a
margin strict enough to be trustworthy, **no swap in the entire test set qualifies**. The
locally-available evidence is not decisive anywhere. That is the same finding measured
from the other side.

---

## 1. What the pending submission will score, and why

`relink_probability_scored` is, in graph terms, the **0.880 submission** — expect ~0.880.

The candidate table aligned perfectly (`candidate coverage 1.000` on every dataset, versus
the index-space bug that killed my v1), and its ILP counts are **identical** to
`claude_submit_repair`'s: `24406/24613, 20806/22446, 6110/6247, 64372/65850`. Same
detections, same solve.

The two final graphs differ by **+6 nodes and −11 edges** out of ~129,000 and ~118,600 —
0.005 % and 0.009 %. That is inside any plausible scoring resolution.

## 2. "0 swaps" is not a complete account of what that stage did

Worth flagging, because a counter that reads zero while the graph changes is the kind of
thing that misleads later. Reconstructing pre-repair edge counts from the log:

```
dataset            ILP kept    theirs pre-repair    diff
44b6_0113de3b        24,406            24,402        -4
44b6_0b24845f        20,806            20,795       -11
6bba_05b6850b         6,110             6,108        -2
6bba_05db0fb1        64,372            64,366        -6
                                        TOTAL       -23
```

My own run reproduces the ILP count **exactly** on all four (diff +0), so this is not a
measurement artifact of how the logs are read — the relink stage genuinely removed 23
edges while reporting zero swaps. Presumably a probability-threshold drop or a two-pass
artifact: an edge removed in pass 1 and not restored is not a "swap" by that counter's
definition, but it is still a change.

The consequences reconcile to the digit:

```
-23 relink edges  +  6 extra gap nodes x 2 edges each  =  -11 net edges
observed                                                  -11
```

The 23 removals created track ends, gap-closing bridged 6 of them, and that is the entire
difference between the two submissions. Nothing is unexplained. **Impact is negligible;
the mislabelled counter is the thing worth fixing** if that notebook is used again.

## 3. What this closes and what it does not

**Closed: local relink, in any posture.** Both designs are now measured, from opposite
ends, and neither has a setting worth submitting. `pipeline/relink.py` stays (tested; it
produced half this bracket) but no further work goes into it.

**Not closed, and untouched:** the ILP's own weights. Everything either relink did was an
*override* of the global solve. Nobody has yet asked whether the global solve is
well-parameterised. It runs on the pack's defaults —
`edge_weight=-1.0, appearance=0.1, disappearance=0.1, division=1.0` — which are inherited,
not measured. `notes/03` §3 records that the same lab's own zebrafish Ultrack config uses a
deliberately **asymmetric** pairing (`appear_weight=-0.002`, `disappear_weight=-0.01`, 5×,
"discouraging track termination more than initiation"), which the pack's symmetric `0.1/0.1`
does not reflect. And `division_weight=1.0` is being paid on a term we score **0.000** on.

## 4. Next: sweep the ILP's weights, not its output

`claude_relink_sweep` wrote `cand_*.npz` for all **24 datasets**, holding the model's
candidate edges *with probabilities* — confirmed present in that kernel's output. That is
precisely the ILP's own input, so the weights can be swept by **re-solving cached
instances**, with no prediction pass: attach the kernel as a `kernelDataSource` and the run
costs minutes of solver time instead of ~25 minutes of GPU inference per pass.

This stays inside the global-consistency regime that just beat local override twice, and it
is the first time any parameter of the solve itself gets examined rather than inherited.

Arms worth including, each with a reason rather than a grid for its own sake:

- **`division_weight`** down from 1.0 — the ILP currently pays a penalty to create forks
  on a term we score 0.000 of 0.100. `notes/25` measured *geometric* fork insertion at 1 TP
  per 2,223 guesses, but the ILP's forks are model-driven, which is a different precision
  regime and has never been separately measured.
- **asymmetric appear/disappear**, following `notes/03` §3's primary-source config, against
  the pack's symmetric default.
- **`edge_weight`** magnitude, which sets how much the learned probability outweighs the
  appearance/disappearance penalties.

Control must reproduce **0.8806**, as in every run since `notes/26`.

Banked floor **0.752**. Best scored submission **0.880**. Cluster **0.913–0.916**.
