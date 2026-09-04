"""Derive _build_claude_gapum.py from the budget2 builder.

budget2 already has the exact shape this needs: one ILP solve on cached candidate graphs
(no GPU), the shipped chain as a `shipped()` helper, a free grid applied on top, per-dataset
node/ratio recording and per-embryo grading. What changes is the grid -- the budget arms
become `close_gaps` radii, and `apply_post` sweeps the radius inside the shipped chain
rather than appending a stage after it.

notes/59 measured close_gaps' 5.75um radius against GT link displacement and found it
rejecting up to 23.4% of real two-frame spans -- but that figure doubles a single-frame step
to stand in for a two-frame span, which assumes straight-line motion and is therefore an
upper bound. This run reads the score directly instead of inferring it.
"""
from pathlib import Path

SP_ = Path(__file__).parent
SRC = SP_ / "_build_claude_budget2.py"
DST = SP_ / "_build_claude_gapum.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_gapum.ipynb — sweep close_gaps\' radius and read the score.\n'
     "\n"
     "notes/57 found pipeline/divisions.py's gates rejecting 88% of real divisions --\n"
     "constants adopted from a public notebook and never checked. notes/59 applied the same\n"
     "check to linking: close_gaps' max_um=5.75 rejects up to 23.4% of real two-frame spans,\n"
     "while cap_edge_length's 14.0 correctly drops only 0.10%.\n"
     "\n"
     "That 23.4% is an UPPER BOUND, not a measurement: it doubles a single-frame step to\n"
     "stand in for a two-frame span, which assumes straight-line motion. This run sweeps the\n"
     "radius on the cached instances and reads the metric instead of inferring it.\n"
     "\n"
     "close_gaps also carries max_added_frac=0.038 and max_added_abs=1650. If those bind\n"
     "before the radius does, the radius was never the constraint -- prediction 2 tests it,\n"
     'which is why inserted-node counts are reported per arm.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "gapum_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_budget2.ipynb", "claude_gapum.ipynb")
edit("result filename", '"budget2.json"', '"gapum.json"', times=2)

edit("drop the budget-prune import -- not used here",
     "from pipeline.repair import (close_gaps, linefit_smooth, prune_short_tracks,\n"
     "                             rank_budget_prune)\n"
     "from harness.tracks import read_estimated_nodes",
     "from pipeline.repair import close_gaps, linefit_smooth, prune_short_tracks\n"
     "from harness.tracks import read_estimated_nodes")

edit("the grid is the gap radius, swept inside the shipped chain",
     """def shipped(g, sc):
    # claude_divsweep's inc/g2sp6, the arm we submit: gaps(2) -> smooth -> prune(6).
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# (label, mode, factor). factor multiplies the dataset's estimated_number_of_nodes;
# None means the stage ignores the budget entirely.
POST = [("none", None, None), ("isolated", "isolated", None)]
for _f in (1.0, 0.9, 0.8, 0.7):
    POST.append((f"geometry{{_f}}", "geometry", _f))
    POST.append((f"length{{_f}}", "length", _f))
""",
     """def shipped(g, sc, max_um):
    # claude_divsweep's inc/g2sp6 with the gap RADIUS exposed. Everything else is the
    # submitted chain: gaps(max_gap=2) -> smooth -> prune(6). The two insertion caps stay
    # at their shipped values so prediction 2 can tell whether they bind before the radius.
    r = close_gaps(*g, scale=sc, max_um=max_um, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# 5.75 is shipped and is the anchor; 10.7 is notes/59's p95 of doubled single-frame steps;
# 14.0 is cap_edge_length's value; 20.0 is past anything defensible, to find the turn.
POST = [("g" + str(_u), _u) for _u in (5.75, 8.0, 10.7, 14.0, 20.0)]
""")

edit("apply_post takes the radius",
     """def apply_post(g, sc, mode, factor, n_est):
    g = shipped(g, sc)
    if mode is not None:
        tgt = float("nan") if factor is None else n_est * factor
        g = rank_budget_prune(g[0], g[1], g[2], n_target=tgt, scale=sc, mode=mode,
                              keep_division_components=True)
    return (g[0], g[1], g[2])""",
     """def apply_post(g, sc, max_um):
    g = shipped(g, sc, max_um)
    return (g[0], g[1], g[2])""")

edit("the arm count line", 'print(f"{{len(POST)}} budget arms on one solve", flush=True)',
     'print(f"{{len(POST)}} gap radii on one solve", flush=True)')

edit("the loop passes a radius",
     """        for key, mode, factor in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, mode, factor, n_est)""",
     """        for key, max_um in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, max_um)""")

edit("the per-dataset log line names the anchor",
     """    base = PER[name]["none"]""",
     """    base = PER[name]["g5.75"]""")

edit("the grid record describes radii",
     '"post": [{{"label": p[0], "mode": p[1], "factor": p[2]}} for p in POST],',
     '"post": [{{"label": p[0], "max_um": p[1]}} for p in POST],')


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
    # Grading is rewritten wholesale: budget2's is specific to isolated/geometry/length
    # arms. The TAIL (notebook assembly, ast.parse, OUT.write_text) is kept -- splicing to
    # end-of-file once produced a builder that ran, exited 0 and wrote no notebook.
    i = out.index('md("""## 2. Grading')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "gapum_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + grading cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
