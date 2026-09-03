"""Derive _build_claude_loosen.py from the topk builder.

topk is the right skeleton and the wrong direction. It sweeps `det_threshold` with one GPU
prediction pass per value, holds `pool_kernel_um` at the pack default, grades a free
post-processing grid on each solved graph, and already records `nodes`, `total_node_ratio`,
the multiplier and the edge anatomy -- every column notes/52's break-even arithmetic needs.

What changes is which way the grid points. topk went UP (0.975 -> 0.999999) chasing a budget
bonus; notes/52 measured that we already hold that bonus (ratio -0.129) while missing 4.21%
of GT edges to undetected endpoints. So the grid goes DOWN, below 0.96875 -- the lowest
value this project has ever tried -- where node count rises and the budget is spent.
"""
from pathlib import Path

SP_ = Path(__file__).parent
NB = Path(__file__).parent
SRC = NB / "_build_claude_topk.py"
DST = NB / "_build_claude_loosen.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_loosen.ipynb — spend the node budget instead of hoarding it.\n'
     "\n"
     "notes/52 measured ratio = -0.129: we already predict 12.9% FEWER nodes than\n"
     "estimated_number_of_nodes, so the multiplier already pays 1.013 and there is no\n"
     "over-prediction left to remove. notes/51 measured fn_detect = 583 (4.21% of GT edges),\n"
     "now the largest failure bucket. Together: the chain deleted so many nodes chasing a\n"
     "bonus it already had that it misses 4.21% of edges through endpoints never detected.\n"
     "\n"
     "Break-even, from notes/52: giving up the multiplier entirely costs 0.0118 on adj_edge,\n"
     "and each recovered fn_detect edge is worth ~1/14,343, so ~169 of the 583 (29%) must\n"
     "come back. This sweep answers whether they do.\n"
     "\n"
     "Derived from the topk builder, which swept this same axis UPWARD. The lowest\n"
     "det_threshold ever tried in this project is 0.96875 (notes/40); notes/44 called the\n"
     "surface flat over [0.965, 0.99] and closed it, and notes/49 found a cliff above it.\n"
     'Below 0.965 has never been touched.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "loosen_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_topk.ipynb", "claude_loosen.ipynb")
edit("result filename", '"topk.json"', '"loosen.json"', times=2)

edit("the grid points DOWN now",
     """# det_threshold pushed FAR past anything tried. notes/44 only ever swept 0.965-0.99,
# where the sigmoid is saturated and node count moves 5.6%. If the logits have spread,
# these reach the budget regime by ranking rather than by spatial suppression.
POOL_GRID = [0.975, 0.999, 0.9999, 0.99999, 0.999999]""",
     """# det_threshold swept DOWNWARD, below the 0.96875 floor of every previous grid.
# notes/52: we are at ratio -0.129, already under budget, so cutting further only deletes
# real edges. The open question is the other direction -- whether spending the multiplier
# buys back enough of notes/51's 583 undetected endpoints to clear the 0.0118 it costs.
# 0.975 stays as the anchor so the run is comparable to notes/48's 0.9410 on these same
# datasets; the rest is unexplored territory.
POOL_GRID = [0.975, 0.95, 0.90, 0.80, 0.60]""")

edit("anchor reference is notes/48's figure on these datasets",
     '''ANCHOR = "p0.975_m6_g2"        # det 0.975 at the default kernel: what we run today
WIDECV = 0.9356                 # claude_budget's p3.0_m6_g2, same chain, n=36''',
     '''ANCHOR = "p0.975_m6_g2"        # det 0.975, the shipped chain: what we run today
WIDECV = 0.9410                 # notes/48's det 0.975 mean best-cell score, same n=36''')

edit("prediction 1 names the right reference",
     """print("\\n1. the anchor arm (det 0.975, the current default) reproduces claude_budget's 0.9356")""",
     """print("\\n1. the anchor arm (det 0.975, the shipped chain) reproduces notes/48's 0.9410")""")

edit("the header says thresholds, not kernels",
     'print(f"{NN} datasets, {len(POOLS)} pool kernels x {len(POST)} post = {len(ARMS)} cells")',
     'print(f"{NN} datasets, {len(POOLS)} thresholds x {len(POST)} post = {len(ARMS)} cells")')

# The table gains fn_detect -- the quantity the whole run is about.
edit("report fn_detect beside the budget columns",
     '''print(f"\\n{'pool um':<10}{'score':>10}{'adj_edge':>10}{'edge_J':>9}{'node_rec':>10}"
      f"{'ratio':>9}{'mult':>8}{'nodes':>12}")
print("-" * 78)''',
     '''A = D.get("anatomy", {})
print(f"\\n{'det':<10}{'score':>10}{'adj_edge':>10}{'edge_J':>9}{'node_rec':>10}"
      f"{'ratio':>9}{'mult':>8}{'nodes':>12}{'fn_detect':>11}{'fn_mislink':>12}")
print("-" * 101)''')

edit("the row prints it too",
     '''    print(f"{p:<10}{s.get(key, float('nan')):>10.4f}"
          f"{s.get('adj_edge_jaccard', float('nan')):>10.4f}"
          f"{s.get('edge_jaccard', float('nan')):>9.4f}"
          f"{s.get('node_recall', float('nan')):>10.4f}"
          f"{ratio:>9.3f}{mult:>8.4f}{n:>12,}")''',
     '''    _a = A.get(c, {})
    print(f"{p:<10}{s.get(key, float('nan')):>10.4f}"
          f"{s.get('adj_edge_jaccard', float('nan')):>10.4f}"
          f"{s.get('edge_jaccard', float('nan')):>9.4f}"
          f"{s.get('node_recall', float('nan')):>10.4f}"
          f"{ratio:>9.3f}{mult:>8.4f}{n:>12,}"
          f"{_a.get('fn_detect', float('nan')):>11,.0f}"
          f"{_a.get('fn_mislink', float('nan')):>12,.0f}")''')


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
    # Predictions 2-5 are rewritten wholesale: topk's graded a saturated-sigmoid theory
    # and a comparison against claude_budget's pool sweep, neither of which exists here.
    # The TAIL after them (notebook assembly, the ast.parse check, OUT.write_text) is kept
    # -- splicing to end-of-file gave a builder that ran, exited 0 and wrote no notebook.
    i = out.index('print("\\n2. ')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "loosen_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + predictions 2-5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
