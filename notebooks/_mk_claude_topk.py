"""Derive _build_claude_topk.py from the budget builder.

`claude_budget` cut nodes with `pool_kernel_um` (spatial suppression) and recall collapsed.
This cuts with `det_threshold` pushed far past anything tried (`notes/44` only ever went to
0.99, where node count moves 5.6%), which is confidence ranking rather than spatial
suppression — a different rule, and the last one available.

The budget builder already loops a grid into `PredictConfig`, holds one half of the
detection line fixed, records per-dataset values and grades with a paired test. So the two
halves swap roles: the swept variable becomes `det_threshold` and `pool_kernel_um` is held
at its default.
"""
from pathlib import Path

SP_ = Path("/tmp/claude-0/-home-user-rogii/840351bc-4942-5d31-9b68-1b00e66da173/scratchpad")
NB = Path("/workspace/biohub-cell_tracking_during_development/notebooks")
SRC = NB / "_build_claude_budget.py"
DST = NB / "_build_claude_topk.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

# Wholesale swaps first, against pristine text.
_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("# The node budget: the term we have never touched")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "topk_intro.md").read_text().strip() + '\n""")')

edit("output path", "notebooks/claude_budget.ipynb", "notebooks/claude_topk.ipynb")

edit("the grid is the confidence cut, pushed past anything tried",
     """# pool_kernel_um: the NMS radius in the pack's detection line, default 3.0 and
# never swept. Node count falls roughly as its cube, which is what reaches the
# regime the budget multiplier rewards.
POOL_GRID = [3.0, 6.0, 10.0, 15.0, 22.0]
DET_FIXED = 0.975          # notes/44's threshold, held while the kernel varies""",
     """# det_threshold pushed FAR past anything tried. notes/44 only ever swept 0.965-0.99,
# where the sigmoid is saturated and node count moves 5.6%. If the logits have spread,
# these reach the budget regime by ranking rather than by spatial suppression.
POOL_GRID = [0.975, 0.999, 0.9999, 0.99999, 0.999999]
DET_FIXED = 3.0            # pool_kernel_um, held at the pack default while the cut varies""")

edit("swap which half of the detection line is swept",
     """        # det_threshold is FIXED here; `det` is the pool kernel. Sweeping both at
        # once would confound the two halves of the same detection line.
        cfg_d = P.PredictConfig(det_threshold=DET_FIXED, pool_kernel_um=det, use_ilp=True,""",
     """        # `det` is the DETECTION THRESHOLD here and pool_kernel_um is held at the
        # pack default -- the mirror of claude_budget, which swept the kernel and held
        # the threshold. Sweeping both would confound the two halves of one line.
        cfg_d = P.PredictConfig(det_threshold=det, pool_kernel_um=DET_FIXED, use_ilp=True,""")

# written by the worker, read by the analysis cell -- 2, and saying so beats
# switching to replace_all and never noticing a third site.
edit("result filename", '"budget.json"', '"topk.json"', times=2)

edit("record under an honest key", '"pool_grid": POOL_GRID, "det_fixed": DET_FIXED,',
     '"det_grid": POOL_GRID, "pool_fixed": DET_FIXED,')

# Grading: the anchor and the crux both change.
edit("anchor is claude_budget's pool-3.0 cell",
     '''ANCHOR = "p3.0_m6_g2"          # the default kernel: exactly what we run today
WIDECV = 0.9348                 # claude_widecv, same chain, n=60''',
     '''ANCHOR = "p0.975_m6_g2"        # det 0.975 at the default kernel: what we run today
WIDECV = 0.9356                 # claude_budget's p3.0_m6_g2, same chain, n=36''')

edit("prediction 1 names the right thing",
     '''print("\\n1. the anchor arm (pool 3.0, the current default) reproduces widecv's 0.9348")''',
     '''print("\\n1. the anchor arm (det 0.975, the current default) reproduces claude_budget's 0.9356")''')

edit("prediction 2 is about reaching the regime at all",
     '''print("\\n2. node count falls more than 5x across the kernel grid")''',
     '''print("\\n2. node count falls more than 3x across the threshold grid")''')

edit("prediction 2 threshold",
     """    fall = max(ns) / max(min(ns), 1)
    ok2 = fall > 5.0""",
     """    fall = max(ns) / max(min(ns), 1)
    ok2 = fall > 3.0""")

edit("prediction 2 failure text",
     '''        print("   det_threshold moved node count 5.6% and the kernel cannot do much")
        print("   better either. The budget regime is unreachable with this detector,")
        print("   and the lever needs a detector that ranks cells, not a wider NMS.")''',
     '''        print("   The sigmoid is saturated even at 0.999999: the detector assigns")
        print("   near-identical confidence to almost everything it finds, so it cannot")
        print("   RANK. Spatial suppression (claude_budget) and confidence ranking are")
        print("   the only two selection rules the pack offers, and neither reaches the")
        print("   budget regime. The direction is finished.")''')

# Prediction 3 becomes the matched-node-count comparison against the pool sweep.
edit("prediction 3 compares against the pool sweep at matched node count",
     '''print("\\n3. total_node_ratio goes below -0.5 at the largest kernel")
rs = [r for _, _, _, r, _ in ROWS if r == r]
if not rs:
    print("   NOT GRADED — ratio unavailable")
else:
    ok3 = min(rs) < -0.5
    print(f"   lowest ratio {min(rs):+.3f} (multiplier {1 - 0.1 * min(rs):.4f})"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if not ok3:
        print("   We never entered the regime the multiplier rewards, so this run")
        print("   probed another plateau edge and says nothing about the trade.")''',
     '''print("\\n3. confidence ranking beats spatial suppression at MATCHED node count")
# claude_budget, n=36, same chain: (nodes, node_recall) as pool_kernel_um widened.
POOL_N = [95137, 325514, 519492, 659535, 700216]
POOL_R = [0.5374, 0.8113, 0.9367, 0.9790, 0.9827]
mine_n = [n for _, _, n, _, _ in ROWS if n]
mine_r = [s.get("node_recall", float("nan")) for _, s, n, _, _ in ROWS if n]
pairs = sorted((n, r) for n, r in zip(mine_n, mine_r) if r == r)
if len(pairs) < 2:
    ok3 = False
    print("   NOT GRADED — need at least two arms with node_recall")
else:
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    # Compare only where BOTH curves have support -- notes/47: extrapolating a ratio
    # past the end of its sweep is how a dead direction looked alive for a full run.
    lo, hi = max(min(xs), min(POOL_N)), min(max(xs), max(POOL_N))
    print(f"   {'nodes':>10}{'this run':>11}{'pool sweep':>12}   verdict")
    wins = 0; tested = 0
    for n in [n for n in POOL_N if lo <= n <= hi]:
        a = float(np.interp(n, xs, ys))
        b = float(np.interp(n, POOL_N, POOL_R))
        tested += 1; wins += int(a > b)
        print(f"   {n:>10,}{a:>11.4f}{b:>12.4f}   {'better' if a > b else 'worse'}")
    ok3 = tested > 0 and wins > tested / 2
    print(f"   better at {wins} of {tested} overlapping counts"
          f"  ->  {'PASS' if ok3 else 'FAIL'}")
    if tested == 0:
        print("   The two sweeps do not overlap in node count, so they cannot be")
        print("   compared at all. That is a design failure, not a result.")
    elif not ok3:
        print("   Confidence ranking is no better than a wider NMS ball at keeping the")
        print("   annotated cells. Both selection rules the pack offers are exhausted.")''')


def main() -> int:
    out = _src
    for name, old, new, times in EDITS:
        n = out.count(old)
        if n != times:
            print(f"REFUSING TO WRITE — {name}: matched {n}x, expected {times}")
            return 1
        before = out
        out = out.replace(old, new, times)
        if out == before:
            print(f"REFUSING TO WRITE — {name}: replace changed nothing")
            return 1
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
