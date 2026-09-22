# Angles on the 0.947 base: one candidate, one dead lead, one inert knob

All against `pub947bera` (the 0.947 fork) at **1,633,336 nodes**, proxy `adj 0.9280`,
`divJ 0.2308`, `PROXY_SCORE 0.9511`.

```
arm          shipped              nodes  vs base      adj    divJ   proxy   note
lb30         bonus 3.0        1,633,397     +61    0.9295  0.2500  0.9545  div fp 1 -> 0
lb15         bonus 1.5        1,633,291     -45    0.9282  0.2308  0.9512  ~inert
lb20         bonus 1.25*      1,756,246 +122,910        -       -       -  *not the 2.0 we set
slm50        low-margin 0.50  1,636,394  +3,058        -       -  0.9499  worse than base
gdg08        gap gain 0.080   1,633,336      +0        -       -       -  INERT, identical file
pub947swa    bhpepper SWA     2,518,455 +885,119  0.8853  0.1429  0.8996  +54% nodes
```

## 1. The SWA weights are drop-in and still useless

`claude-inspect` proved bhpepper's checkpoint loads without modification — 136 common keys, 0
mismatches, both 2,077,996 params — and bhpepper sits at **0.950**. Loaded correctly
(`public sha256 12f6881ee362 -> swa 0eacacaf0b43`, no traceback). And it produced **+54% nodes**
and a proxy of 0.8996 against the base's 0.9511.

**Architecturally drop-in is not operationally drop-in.** Their detector was trained inside
*their* pipeline and is calibrated to *their* detection threshold; ours runs `DET_THRESHOLD =
0.965`, tuned to the public checkpoint's logit scale. A different model at the same threshold
gives a different node count, and here 54% more of them.

This is the same failure as `ftune` (`notes/86`), one order of magnitude worse — and `ftune`
already showed thresholding cannot claw back 8%, let alone 54%. **Foreign weights are closed**
unless someone recalibrates the threshold to them, which is a sweep we cannot afford and could
not rank offline anyway.

The check cost two CPU minutes and the run cost 50 GPU-minutes. Worth it: the alternative was
believing "team at 0.950 + verified drop-in" and spending a submission slot on it.

## 2. `lb30` is the one candidate

Bonus 3.0 is the only arm with **both** a better proxy and a node count within 61 of base —
the profile where `total_node_ratio` cannot eat the gain. It also removes a division false
positive (`tp/fp/fn` 3/1/9 -> 3/0/9), lifting `divJ` 0.2308 -> 0.2500.

Stated against it, plainly: raising the learned bonus means **trusting the model more**, which
is exactly the direction `notes/81` measured our proxy as biased toward — it called `norelink`
+0.0337 against an actual −0.001, and that was also a trust-the-model change. The +0.0034 here
is in the inflated direction. It is a candidate, not a prediction.

## 3. The sweep overrode an arm, and the bonus curve is not monotonic

`lb20` set 2.0 and shipped **1.25**, because `PP_CANDIDATES` contains a `bonus125` entry the
sweep selected. So its +122,910 nodes describe 1.25. But `lb15` shipped 1.5 and moved **−45**
nodes. 1.25 -> +122,910 and 1.5 -> −45 cannot both sit on a monotonic curve, so something
about the sweep's override path differs from setting the value in the config cell. `lb20x`
(2.0, `PP_CANDIDATES` emptied) is queued and settles what 2.0 actually does.

## 4. Closed by this batch

* `GAP_DENSITY_GAIN` — inert, byte-identical submission. Dead knob.
* `SECONDARY_LOW_MARGIN_MAX` 0.50 — proxy 0.9499, below base. Conditional-fusion widening does
  not pay here, despite being the mechanism hikaggler reports working in their own stack.
* Foreign detector weights — see §1.
