"""Derive _build_claude_divsweep.py from the ilp_sweep3 builder.

ilp_sweep3 has exactly the machinery this needs: it re-solves CACHED candidate graphs with
no GPU and no prediction pass, then scores with the harness. What changes is which axis is
swept. ilp_sweep3 swept 18 ILP weight settings and graded each with/without one fixed
repair chain; notes/36 closed that axis. Here the weights are held at two points (a
control and the incumbent) and the POST-PROCESSING chain becomes the grid, with `div_J`
promoted from a reported column to the quantity being optimised.

So the two halves swap roles, same as _mk_topk.py did for the detection line.
"""
from pathlib import Path

SP_ = Path(__file__).parent
NB = Path("/workspace/biohub-cell_tracking_during_development/notebooks")
SRC = NB / "_build_claude_ilp_sweep3.py"
DST = NB / "_build_claude_divsweep.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

# --- wholesale swaps first, against pristine text -----------------------------------
edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_divsweep.ipynb — where half the division term went.\n'
     "\n"
     "notes/36 measured div_J 0.1154 at ratio0.4_2.0 with close_gaps(max_gap=1) +\n"
     "linefit_smooth, on 24 datasets. notes/42 measured 0.0645 at the SAME ILP weights on\n"
     "12 datasets, after the config audit moved max_gap to 2 and added\n"
     "prune_short_tracks(6). Worth +0.0051 if it is real and recoverable.\n"
     "\n"
     "But the two numbers are on different dataset samples, and notes/44 measured that\n"
     "sample as easy by +0.0116 — so the drop may not exist at all. Every arm here runs on\n"
     "the same cached instances so that question gets a clean answer either way.\n"
     "\n"
     "Derived from the ilp_sweep3 builder, which already re-solves cached candidate graphs\n"
     "with no GPU. notes/36 closed the weight axis, so the weights are held at two points\n"
     "and the POST-PROCESSING chain becomes the grid, with div_J promoted from a reported\n"
     'column to the quantity being optimised.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "divsweep_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_ilp_sweep3.ipynb", "claude_divsweep.ipynb")
edit("result filename", '"ilp_sweep3.json"', '"divsweep.json"', times=2)

# --- the worker: two solves, eight post-chains ---------------------------------------
edit("repair_chain becomes a parameterised post-processing grid",
     """def repair_chain(g, sc):
    r = close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038, max_added_abs=1650)
    return linefit_smooth(*r, window=2, weight=0.76, scale=sc, max_shift_um=3.2)""",
     """def _gaps(g, sc, mg):
    return close_gaps(*g, scale=sc, max_um=5.75, max_added_frac=0.038,
                      max_added_abs=1650, max_gap=mg)

def _smooth(g, sc):
    return linefit_smooth(*g, window=2, weight=0.76, scale=sc, max_shift_um=3.2)

def _prune(g, sc, keep):
    return prune_short_tracks(*g, min_frames=6, keep_division_components=keep)

# (label, stages). Each pair differs from another by exactly ONE stage, so a div_J drop
# is attributable. g1s is notes/36's chain (div_J 0.1154); g2sp6 is what we ship.
POST = [
    ("raw",      []),
    ("g1",       [lambda g, s: _gaps(g, s, 1)]),
    ("g1s",      [lambda g, s: _gaps(g, s, 1), _smooth]),
    ("g2s",      [lambda g, s: _gaps(g, s, 2), _smooth]),
    ("g1sp6",    [lambda g, s: _gaps(g, s, 1), _smooth,
                  lambda g, s: _prune(g, s, True)]),
    ("g2p6",     [lambda g, s: _gaps(g, s, 2),
                  lambda g, s: _prune(g, s, True)]),
    ("g2sp6",    [lambda g, s: _gaps(g, s, 2), _smooth,
                  lambda g, s: _prune(g, s, True)]),
    ("g2sp6_nk", [lambda g, s: _gaps(g, s, 2), _smooth,
                  lambda g, s: _prune(g, s, False)]),
]

def apply_post(g, sc, stages):
    for fn in stages:
        g = fn(g, sc)
    return (g[0], g[1], g[2])""")

edit("import prune_short_tracks",
     "from pipeline.repair import close_gaps, linefit_smooth",
     "from pipeline.repair import close_gaps, linefit_smooth, prune_short_tracks")

_OLD_ARMS = _src[_src.index("# ---- the sweep ---"):_src.index("names = sorted(p.stem[len")]
edit("two solves instead of eighteen", _OLD_ARMS,
     """# ---- the sweep -------------------------------------------------------------------
# (label, edge_weight, appearance, disappearance, division)
# The WEIGHT axis is closed -- notes/36 ran 18 settings and nothing beat ratio0.4_2.0
# (closest -0.0009), div_J climbs to 0.1500 only where the score has fallen to 0.8615, and
# cheapening division_weight makes MORE forks and a WORSE div_J. Two solves, no more: a
# control to prove the cache and solver still reproduce, and the incumbent to sweep under.
ARMS = [("ctl", -1.0, 0.1, 0.1, 1.0),          # pack defaults: div_J 0.0000 in notes/36
        ("inc", -1.0, 0.4, 2.0, 1.0)]          # ratio0.4_2.0, what we ship
print(f"{{len(ARMS)}} solves x {{len(POST)}} post-chains = {{len(ARMS)*len(POST)}} arms",
      flush=True)

""")

edit("grade every post-chain on each solved graph, and record div counts",
     """    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for with_repair in (False, True):
            key = f"{{lbl}}+repair" if with_repair else lbl
            g = (tr.t, tr.zyx, tr.edges)
            if with_repair:
                g = repair_chain(g, sc)
            ROWS[key].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))""",
     """    for lbl, ew, ap, dis, dv in ARMS:
        g_td = solve(base_td, ew, ap, dis, dv)
        tr = Tracks.from_tracksdata(g_td)
        for post_lbl, stages in POST:
            key = f"{{lbl}}/{{post_lbl}}"
            g = apply_post((tr.t, tr.zyx, tr.edges), sc, stages)
            ROWS[key].append(h.score_graph(name, Tracks(g[0], g[1], g[2])))""")

edit("labels are the solve x post cross product",
     'LABELS = [a[0] for a in ARMS] + [f"{{a[0]}}+repair" for a in ARMS]',
     'LABELS = [f"{{a[0]}}/{{p[0]}}" for a in ARMS for p in POST]')

edit("per-dataset record carries div counts, not just the edge term",
     """            PER.setdefault(name, {{}})[key] = float(
                ROWS[key][-1].get("adj_edge_jaccard", float("nan")))
    best = max(PER[name], key=lambda k: PER[name][k] if PER[name][k] == PER[name][k] else -9)
    print(f"    control {{PER[name]['control']:.4f}}  best {{best}} {{PER[name][best]:.4f}} "
          f"({{PER[name][best]-PER[name]['control']:+.4f}})  {{time.time()-t0:.0f}}s", flush=True)""",
     """            r = ROWS[key][-1]
            # div_J is MICRO-averaged (purescore.summarise), so the per-dataset record has
            # to carry counts, not a ratio -- a per-dataset div_J cannot be averaged into
            # the reported one. notes/47 is the fourth time a ratio was aggregated wrongly.
            PER.setdefault(name, {{}})[key] = {{
                "adj": float(r.get("adj_edge_jaccard", float("nan"))),
                "score": float(r.get("score", float("nan"))),
                "dtp": float(r.get("division_tp", 0.0)),
                "dfp": float(r.get("division_fp", 0.0)),
                "dfn": float(r.get("division_fn", 0.0))}}
    ref, shipped = PER[name]["inc/g1s"], PER[name]["inc/g2sp6"]
    print(f"    inc/g1s adj {{ref['adj']:.4f}} dTP {{ref['dtp']:.0f}} dFP {{ref['dfp']:.0f}}"
          f"   inc/g2sp6 adj {{shipped['adj']:.4f}} dTP {{shipped['dtp']:.0f}} "
          f"dFP {{shipped['dfp']:.0f}}   {{time.time()-t0:.0f}}s", flush=True)""")

edit("the grid record describes solves, and POST is recorded too",
     """           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],""",
     """           "grid": [{{"label": a[0], "edge": a[1], "appear": a[2],
                    "disappear": a[3], "division": a[4]}} for a in ARMS],
           "post": [p[0] for p in POST],""")


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
    # The grading cell is rewritten wholesale rather than patched: ilp_sweep3's ranks 36
    # arms by total score against a weight grid that no longer exists here. The TAIL after
    # it (notebook assembly, the ast.parse check on every code cell, OUT.write_text) is
    # kept -- splicing to end-of-file instead silently produced a builder that ran, exited
    # 0, and wrote no notebook.
    i = out.index('md("""## 2. The five predictions""")')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "divsweep_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + grading cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
