"""Derive _build_claude_budget2.py from the divsweep builder.

divsweep already does everything structural this needs: re-solve cached candidate graphs
with no GPU, apply a grid of post-processing chains to each solved graph for free, record
per-dataset rows, and grade per embryo. What changes is the grid — the eight repair chains
become one fixed chain (the shipped `inc/g2sp6`) plus `rank_budget_prune` at the end, and
each arm needs the dataset's own `estimated_number_of_nodes` rather than a constant.
"""
from pathlib import Path

SP_ = Path(__file__).parent
NB = Path(__file__).parent
SRC = NB / "_build_claude_divsweep.py"
DST = NB / "_build_claude_budget2.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_budget2.ipynb — rank tracks under a per-dataset budget.\n'
     "\n"
     "notes/51: fn_detect is now 583 against fn_mislink 226, so the graph side is bounded\n"
     "at +0.020..+0.035 against a 0.034 gap. And the per-dataset node budget notes/04 §9\n"
     "identified has never been used -- we still ship one global DET_THRESHOLD, while\n"
     "r35's linker.py carries max_pred_nodes beside rank_tracks_by_geometry.\n"
     "\n"
     "This is the THIRD selection rule. claude_budget cut with an NMS radius and\n"
     "claude_topk with a confidence threshold; both cut at the DETECTION stage, before\n"
     "anything knew which detections would end up in a good track, and both destroyed\n"
     "recall. Cutting after linking removes a junk track's nodes AND its false-positive\n"
     "edges together.\n"
     "\n"
     "Derived from the divsweep builder, which already re-solves cached graphs with no GPU\n"
     'and grades per embryo.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "budget2_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_divsweep.ipynb", "claude_budget2.ipynb")
edit("result filename", '"divsweep.json"', '"budget2.json"', times=2)

edit("import rank_budget_prune and the node estimate",
     "from pipeline.repair import close_gaps, linefit_smooth, prune_short_tracks",
     "from pipeline.repair import (close_gaps, linefit_smooth, prune_short_tracks,\n"
     "                             rank_budget_prune)\n"
     "from harness.tracks import read_estimated_nodes")

# The POST grid becomes: the shipped chain, then a budget stage that needs n_target.
_i1 = _src.index("def _gaps(g, sc, mg):")
_j1 = _src.index("def apply_post(g, sc, stages):")
edit("one fixed chain plus a budget stage", _src[_i1:_j1],
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

""")

edit("apply_post takes a budget rather than a stage list",
     """def apply_post(g, sc, stages):
    for fn in stages:
        g = fn(g, sc)
    return (g[0], g[1], g[2])""",
     """def apply_post(g, sc, mode, factor, n_est):
    g = shipped(g, sc)
    if mode is not None:
        tgt = float("nan") if factor is None else n_est * factor
        g = rank_budget_prune(g[0], g[1], g[2], n_target=tgt, scale=sc, mode=mode,
                              keep_division_components=True)
    return (g[0], g[1], g[2])""")

edit("two solves become one -- the weight axis is closed",
     '''ARMS = [("ctl", -1.0, 0.1, 0.1, 1.0),          # pack defaults: div_J 0.0000 in notes/36
        ("inc", -1.0, 0.4, 2.0, 1.0)]          # ratio0.4_2.0, what we ship
print(f"{{len(ARMS)}} solves x {{len(POST)}} post-chains = {{len(ARMS)*len(POST)}} arms",
      flush=True)''',
     '''# One solve. The weight axis is closed three times over (notes/36, 50) and the control
# arm has already served its purpose -- claude_divsweep confirmed div_J 0.0000 there.
ARMS = [("inc", -1.0, 0.4, 2.0, 1.0)]          # ratio0.4_2.0, what we ship
print(f"{{len(POST)}} budget arms on one solve", flush=True)''')

edit("labels drop the solve prefix", 'LABELS = [f"{{a[0]}}/{{p[0]}}" for a in ARMS for p in POST]',
     'LABELS = [p[0] for p in POST]')

edit("the loop passes a per-dataset budget",
     """    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for post_lbl, stages in POST:
            key = f"{{lbl}}/{{post_lbl}}"
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, stages)""",
     """    n_est = read_estimated_nodes(TRAIN / f"{{name}}.geff")
    print(f"    estimated_number_of_nodes {{n_est:,.0f}}", flush=True)
    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for key, mode, factor in POST:
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, mode, factor, n_est)""")

edit("record node count and budget ratio per arm",
     """            PER.setdefault(name, {{}})[key] = {{
                "adj": float(r.get("adj_edge_jaccard", float("nan"))),
                "score": float(r.get("score", float("nan"))),
                "dtp": float(r.get("division_tp", 0.0)),
                "dfp": float(r.get("division_fp", 0.0)),
                "dfn": float(r.get("division_fn", 0.0))}}""",
     """            PER.setdefault(name, {{}})[key] = {{
                "adj": float(r.get("adj_edge_jaccard", float("nan"))),
                "score": float(r.get("score", float("nan"))),
                "dtp": float(r.get("division_tp", 0.0)),
                "dfp": float(r.get("division_fp", 0.0)),
                "dfn": float(r.get("division_fn", 0.0)),
                # the budget itself -- reported even on a clean failure, because no run
                # so far has measured where we actually sit against N_est.
                "nodes": float(r.get("num_pred_nodes", len(g[0]))),
                "ratio": float(r.get("total_node_ratio", float("nan"))),
                "n_est": float(n_est)}}""")

edit("the per-dataset log line reports the budget",
     """    ref, shipped = PER[name]["inc/g1s"], PER[name]["inc/g2sp6"]
    print(f"    inc/g1s adj {{ref['adj']:.4f}} dTP {{ref['dtp']:.0f}} dFP {{ref['dfp']:.0f}}"
          f"   inc/g2sp6 adj {{shipped['adj']:.4f}} dTP {{shipped['dtp']:.0f}} "
          f"dFP {{shipped['dfp']:.0f}}   {{time.time()-t0:.0f}}s", flush=True)""",
     """    base = PER[name]["none"]
    best_k = max(PER[name], key=lambda k: PER[name][k]["adj"]
                 if PER[name][k]["adj"] == PER[name][k]["adj"] else -9)
    print(f"    none adj {{base['adj']:.4f}} nodes {{base['nodes']:,.0f}} "
          f"ratio {{base['ratio']:+.3f}}  |  best {{best_k}} "
          f"adj {{PER[name][best_k]['adj']:.4f}} "
          f"({{PER[name][best_k]['adj']-base['adj']:+.4f}})   {{time.time()-t0:.0f}}s",
          flush=True)""")

edit("the grid record describes budget arms",
     """           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "post": [p[0] for p in POST],""",
     """           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "post": [{{"label": p[0], "mode": p[1], "factor": p[2]}} for p in POST],""")


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
    # Grading is rewritten wholesale; the TAIL after it (notebook assembly, the ast.parse
    # check, OUT.write_text) is kept -- splicing to end-of-file produced a builder that
    # ran, exited 0 and wrote no notebook when divsweep was derived.
    i = out.index('md("""## 2. Grading')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "budget2_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + grading cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
