"""Derive _build_claude_budget.py from the widecv builder.

`notes/45`: J_adj = J * (1 - 0.1 * (N_pred - N_total)/N_total), so N_pred -> 0 gives a
multiplier of 1.1 and ours is 1.0012. Every sweep this project ran moved `det_threshold`,
which changes node count by 5.6% across its whole usable range -- entirely inside the
predict-everything regime where the multiplier is pinned near 1.0.

`pool_kernel_um` is the other half of the detection line and has never been swept. It is
the NMS radius; node count falls roughly as its cube.

The widecv builder already has the pieces this needs -- paired grading, a time guard,
per-dataset recording -- so the loop is repurposed rather than rewritten. `DET_GRID`
becomes `POOL_GRID` by a global rename so no site keeps the misleading name, and the label
prefix goes "d" -> "p" for the same reason. Misleading names have cost this project real
runs.
"""
from pathlib import Path

SP_ = Path("/tmp/claude-0/-home-user-rogii/840351bc-4942-5d31-9b68-1b00e66da173/scratchpad")
NB = Path("/workspace/biohub-cell_tracking_during_development/notebooks")
SRC = NB / "_build_claude_widecv.py"
DST = NB / "_build_claude_budget.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

# --- wholesale section swaps first: they must match pristine text -------------------
_i = _src.index("md('''\n## 2. Five predictions, graded with a paired test")
_j = _src.index('nb = {"cells": CELLS,')
edit("replace the whole grading section", _src[_i:_j], (SP_ / "budget_tail.py").read_text())

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("# More data — because at n=12")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "budget_intro.md").read_text().strip() + '\n""")')

# --- then the mechanical ones -------------------------------------------------------
edit("output path",
     'notebooks/claude_widecv.ipynb', 'notebooks/claude_budget.ipynb')

edit("sample size", "N_DATASETS = 60", "N_DATASETS = 36")

edit("the grid is now the NMS radius, not the probability cut",
     "DET_GRID = [0.98, 0.975, 0.97]",
     "# pool_kernel_um: the NMS radius in the pack's detection line, default 3.0 and\n"
     "# never swept. Node count falls roughly as its cube, which is what reaches the\n"
     "# regime the budget multiplier rewards.\n"
     "DET_GRID = [3.0, 6.0, 10.0, 15.0, 22.0]\n"
     "DET_FIXED = 0.975          # notes/44's threshold, held while the kernel varies")

edit("result filename", '"widecv.json"', '"budget.json"', times=1)

edit("pass the kernel into PredictConfig",
     """        cfg_d = P.PredictConfig(det_threshold=det, use_ilp=True,""",
     """        # det_threshold is FIXED here; `det` is the pool kernel. Sweeping both at
        # once would confound the two halves of the same detection line.
        cfg_d = P.PredictConfig(det_threshold=DET_FIXED, pool_kernel_um=det, use_ilp=True,""")

edit("carry DET_FIXED into the worker", "DET_GRID = __DET_GRID__",
     "DET_GRID = __DET_GRID__\nDET_FIXED = __DET_FIXED__")

edit("substitute DET_FIXED",
     '.replace("__DET_GRID__", repr(DET_GRID))',
     '.replace("__DET_GRID__", repr(DET_GRID))\n'
     '              .replace("__DET_FIXED__", repr(DET_FIXED))')

edit("record the pool grid under its own key", '"det_grid": DET_GRID,',
     '"pool_grid": DET_GRID, "det_fixed": DET_FIXED,')

# Global renames LAST, so the anchors above matched the original text.
edit("rename DET_GRID -> POOL_GRID everywhere", "DET_GRID", "POOL_GRID", times=None)
# No separate __DET_GRID__ rename: the DET_GRID rename above already covers it, since
# "__DET_GRID__" contains "DET_GRID". A second pass would match 0x and refuse.
edit("label prefix d -> p", '"d" + str(', '"p" + str(', times=None)


def main() -> int:
    out = _src
    for name, old, new, times in EDITS:
        n = out.count(old)
        if times is None:
            if n == 0:
                print(f"REFUSING TO WRITE — {name}: matched 0x, expected >=1")
                return 1
        elif n != times:
            print(f"REFUSING TO WRITE — {name}: matched {n}x, expected {times}")
            return 1
        before = out
        out = out.replace(old, new) if times is None else out.replace(old, new, times)
        if out == before:
            print(f"REFUSING TO WRITE — {name}: replace changed nothing")
            return 1
    # The rename must not have left the old name anywhere.
    for stale in ("DET_GRID", "__DET_GRID__"):
        if stale in out:
            print(f"REFUSING TO WRITE — stale name {stale} survives the rename")
            return 1
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits in order")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
