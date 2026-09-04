"""Derive _build_claude_static.py from the gapum builder.

gapum already sweeps one repair parameter inside the shipped chain on cached candidate
graphs -- one ILP solve, no GPU, per-dataset node/ratio recording, per-embryo grading.
Only the swept parameter changes: `close_gaps`' radius becomes `linefit_smooth`'s new
`static_um`.

notes/59 measured 8.4% of ground-truth links at EXACTLY zero displacement (10,772 of
128,883) -- frozen frames and interpolated annotation, both confirmed at scale. Where the
truth is static our detections still jitter, a line fit through jitter has a spurious
slope, and linefit_smooth -- the MAJOR half of the repair chain, +0.0086 of +0.0113 per
notes/26/27 -- drags the node along it. static_um zeroes the slope below a speed threshold
so the node is pulled toward the window mean instead.

Unlike the last three levers this is not a re-tune of a constant somebody already swept on
the metric (notes/60's rule). It is a new term derived from a property of the annotation we
measured ourselves.
"""
from pathlib import Path

SP_ = Path(__file__).parent
SRC = SP_ / "_build_claude_gapum.py"
DST = SP_ / "_build_claude_static.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_static.ipynb — do not smooth what is not moving.\n'
     "\n"
     "notes/59: 8.4% of GT links (10,772 of 128,883) have EXACTLY zero displacement --\n"
     "frozen frames, because the volumes are crops of one master acquisition, plus\n"
     "annotations interpolated between labelled frames. Where the truth is static our\n"
     "detections jitter, a line fit through jitter has a spurious slope, and\n"
     "linefit_smooth drags the node along it -- and that stage is the MAJOR half of the\n"
     "repair chain (+0.0086 of +0.0113, notes/26/27).\n"
     "\n"
     "linefit_smooth's new static_um zeroes the slope below a speed threshold, pulling\n"
     "toward the window mean, which is the right estimator for a static point. Default 0.0\n"
     "is an exact no-op, so the anchor arm is a construction check.\n"
     "\n"
     "Derived from the gapum builder: same cached-graph sweep, different parameter.\n"
     'Unlike close_gaps this constant has never been tuned on the metric.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "static_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_gapum.ipynb", "claude_static.ipynb")
edit("result filename", '"gapum.json"', '"static.json"', times=2)

edit("sweep static_um inside the chain, not the gap radius",
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
POST = [("g" + str(_u), _u) for _u in (5.75, 8.0, 10.7, 14.0, 20.0)]""",
     """def shipped(g, sc, static_um):
    # claude_divsweep's inc/g2sp6 with linefit_smooth's static_um exposed. Everything else
    # is the submitted chain: gaps(max_gap=2) -> smooth -> prune(6).
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                   max_added_abs=1650, max_gap=2)
    r = linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2,
                       static_um=static_um)
    return prune_short_tracks(*r, min_frames=6, keep_division_components=True)

# 0.0 is the shipped chain and an EXACT no-op, so it anchors the run by construction.
# The median single-frame GT step is 1.82um (notes/59), so 0.3-1.0 catches near-frozen
# chains and 1.8 treats most slow chains as static -- past anything defensible, to find
# where it turns.
# v1 peaked at its TOP value (s1.8, +0.0010), which its own prediction 4 said means the
# grid stopped too early. 1.8um is ~the median single-frame step (notes/59), so beyond it
# most chains are treated as static and the arm approaches "use the window mean, never the
# line". mean_only (1e9) tests that endpoint directly -- notes/26/27 credited smoothing
# with +0.0086 and never compared the line fit against a plain moving average.
POST = [("s" + str(_u), _u) for _u in (0.0, 1.8, 2.5, 3.5, 5.0)] + [("mean_only", 1e9)]""")

edit("apply_post passes static_um",
     """def apply_post(g, sc, max_um):
    g = shipped(g, sc, max_um)
    return (g[0], g[1], g[2])""",
     """def apply_post(g, sc, static_um):
    g = shipped(g, sc, static_um)
    return (g[0], g[1], g[2])""")

edit("the arm count line", 'print(f"{{len(POST)}} gap radii on one solve", flush=True)',
     'print(f"{{len(POST)}} static_um values on one solve", flush=True)')

edit("the loop passes static_um",
     """        for key, max_um in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, max_um)""",
     """        anchor_zyx = None
        for key, static_um in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, static_um)
            if anchor_zyx is None:
                anchor_zyx = np.asarray(g[1], float).copy()
                moved_frac, moved_um = 0.0, 0.0
            else:
                _d = (np.asarray(g[1], float) - anchor_zyx) * np.asarray(sc, float)
                _n = np.linalg.norm(_d, axis=1)
                moved_frac = float((_n > 1e-9).mean()); moved_um = float(_n.mean())""")

edit("the per-dataset log line names the anchor",
     """    base = PER[name]["g5.75"]""", """    base = PER[name]["s0.0"]""")

edit("record how far positions actually moved",
     """                "nodes": float(r.get("num_pred_nodes", len(g[0]))),""",
     """                # v1's prediction 2 compared node COUNTS, which static_um cannot
                # change -- it only MOVES nodes -- so the check was blind by construction
                # and wrongly reported "the fallback never fires". Record displacement.
                "moved_frac": moved_frac, "moved_um": moved_um,
                "nodes": float(r.get("num_pred_nodes", len(g[0]))),""")

edit("the grid record describes static_um",
     '"post": [{{"label": p[0], "max_um": p[1]}} for p in POST],',
     '"post": [{{"label": p[0], "static_um": p[1]}} for p in POST],')


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
    i = out.index('md("""## 2. Grading')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "static_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + grading cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
