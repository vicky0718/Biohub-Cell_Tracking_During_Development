"""Prove the secondary blend, and patch it against the pack's REAL source.

The pack's `predict_video` was downloaded directly from its Kaggle dataset, so this can do
what the bidirectional build could not: apply the patch to the actual function text and
compile it, here, before a GPU is spent. Three of that build's launches died on things this
would have caught in seconds.

The behavioural tests target the two claims the mode names make:

  * **`fixed`** mixes everywhere by a constant.
  * **`low_margin`** mixes only where the primary is UNSURE *and* the secondary agrees. Both
    halves get a test that fails if the other is dropped — otherwise "low_margin" passes for
    any weighting that happens to be small.

And the one that makes a control a control: `weight=0` must return the primary *by
identity*, not approximately.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.bidirectional import ANCHOR as BLEND_ANCHOR
from pipeline.secondary import ENCODE_ANCHOR, patch_source, secondary_blend

PACK_SRC = Path("/tmp/claude-0/-home-user-rogii/840351bc-4942-5d31-9b68-1b00e66da173"
                "/scratchpad/predict_pack.py")

FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def main() -> int:
    print("=" * 78)
    print("the control arm must be exact")
    print("=" * 78)
    rng = np.random.default_rng(0)
    p = rng.normal(size=(1, 6, 5))
    s = rng.normal(size=(1, 6, 5))
    z = secondary_blend(p, s, 0.0)
    check("weight=0 returns the primary by identity", z is p or np.array_equal(z, p))

    print()
    print("=" * 78)
    print("fixed mode")
    print("=" * 78)
    out = secondary_blend(p, s, 0.5, mode="fixed")
    from pipeline.bidirectional import calibrate
    expect = 0.5 * p + 0.5 * calibrate(s, p, 1)
    check("fixed is a straight calibrated mix", np.allclose(out, expect))
    check("the secondary is CALIBRATED, not mixed raw",
          not np.allclose(out, 0.5 * p + 0.5 * s),
          "mixing raw logits lets the secondary's temperature dominate")

    print()
    print("=" * 78)
    print("low_margin: unsure AND agreeing — both halves load-bearing")
    print("=" * 78)
    # One target, three candidate parents.
    # (a) primary CONFIDENT, secondary agrees  -> almost no movement
    conf_p = np.array([[[6.0], [0.0], [0.0]]])
    conf_s = np.array([[[6.0], [0.0], [0.0]]])
    d_conf = float(np.abs(secondary_blend(conf_p, conf_s, 0.5) - conf_p).max())
    # (b) primary UNSURE, secondary agrees     -> real movement
    uns_p = np.array([[[0.30], [0.20], [0.0]]])
    uns_s = np.array([[[3.0], [0.0], [0.0]]])
    d_uns = float(np.abs(secondary_blend(uns_p, uns_s, 0.5) - uns_p).max())
    # (c) primary UNSURE, secondary DISAGREES  -> no movement
    dis_p = np.array([[[0.30], [0.20], [0.0]]])
    dis_s = np.array([[[0.0], [3.0], [0.0]]])
    d_dis = float(np.abs(secondary_blend(dis_p, dis_s, 0.5) - dis_p).max())

    print(f"   confident+agree {d_conf:.4f}   unsure+agree {d_uns:.4f}   unsure+disagree {d_dis:.4f}")
    check("an UNSURE primary moves more than a confident one", d_uns > d_conf * 5,
          "if this fails the margin term is not doing anything")
    check("a DISAGREEING secondary is ignored entirely", d_dis < 1e-9,
          "if this fails the agreement term is not doing anything")
    check("fixed mode would have moved the disagreeing case (so the test has teeth)",
          float(np.abs(secondary_blend(dis_p, dis_s, 0.5, mode="fixed") - dis_p).max()) > 0.1)

    print()
    print("=" * 78)
    print("inputs")
    print("=" * 78)
    for w in (-0.1, 1.5):
        try:
            secondary_blend(p, s, w)
            check(f"weight={w} raises", False, "it did not")
        except ValueError:
            check(f"weight={w} raises", True)
    try:
        secondary_blend(p, s, 0.3, mode="nonsense")
        check("an unknown mode raises", False, "it did not")
    except ValueError:
        check("an unknown mode raises", True)
    try:
        secondary_blend(np.zeros((1, 4, 6)), np.zeros((1, 6, 4)), 0.3)
        check("a mis-oriented secondary raises", False, "it did not")
    except ValueError:
        check("a mis-oriented secondary raises", True)
    check("output stays finite", bool(np.isfinite(secondary_blend(p, s, 0.5)).all()))

    print()
    print("=" * 78)
    print("the patch, against the pack's ACTUAL predict_video source")
    print("=" * 78)
    if not PACK_SRC.is_file():
        check("the pack source is available to patch against", False, str(PACK_SRC))
    else:
        full = PACK_SRC.read_text()
        i = full.index("def predict_video")
        fn = full[i:full.index("\ndef ", i + 10)]
        check("both anchors appear exactly once in the real source",
              fn.count(ENCODE_ANCHOR) == 1 and fn.count(BLEND_ANCHOR) == 1,
              f"encode {fn.count(ENCODE_ANCHOR)}x, blend {fn.count(BLEND_ANCHOR)}x")

        patched = patch_source(fn, 0.15)
        body = patched.split("\n", 5)[5]        # drop the injected header lines
        try:
            ast.parse(body)
            check("the patched function still compiles", True,
                  f"{len(fn):,} -> {len(patched):,} chars")
        except SyntaxError as e:
            check("the patched function still compiles", False,
                  f"line {e.lineno}: {e.msg}")
        check("the secondary encode lands next to the primary's",
              "_SECONDARY_MODEL.encode(imgs)" in patched)
        check("the blend lands at the predict_edges site",
              "_secondary_blend(" in patched and "_SECONDARY_MODEL.predict_edges" in patched)
        check("both original lines survive",
              ENCODE_ANCHOR in patched and BLEND_ANCHOR in patched)

        # It must compose with the bidirectional patch — they share the blend anchor.
        from pipeline.bidirectional import patch_source as bidir_patch
        both = bidir_patch(patch_source(fn, 0.15), 0.15)
        try:
            ast.parse(both.split("\n", 7)[7])
            check("it composes with the bidirectional patch", True)
        except SyntaxError as e:
            check("it composes with the bidirectional patch", False,
                  f"line {e.lineno}: {e.msg}")
        check("and both mechanisms are present after composing",
              "_secondary_blend(" in both and "_harmonic_blend(" in both)

        for bad, why in ((fn.replace(ENCODE_ANCHOR, ""), "a missing encode anchor"),
                         (fn + ENCODE_ANCHOR, "a duplicated encode anchor")):
            try:
                patch_source(bad, 0.15)
                check(f"{why} raises", False, "it did not — it would run the control")
            except ValueError:
                check(f"{why} raises", True)

    print()
    print("=" * 78)
    if FAIL:
        print(f"{len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("the secondary blend behaves, and patches the pack's real source cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
