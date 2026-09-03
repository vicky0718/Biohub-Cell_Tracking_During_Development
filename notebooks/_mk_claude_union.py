"""Derive _build_claude_union.py from the spotiflow builder.

claude_spotiflow measured the two detectors SEPARATELY -- pack_recall 0.996, spot_recall
0.547 -- and notes/47 closed spotiflow on that. But standalone quality is the wrong test
for an ensemble. Two detectors can have one dominating the recall-per-node curve at every
matched count (which is what notes/47 actually showed) while still finding DIFFERENT cells,
and the union was never computed.

Two things make the question live again:

* notes/52 measured ratio = -0.129. We run 12.9% UNDER the node budget, so adding nodes is
  affordable now. When spotiflow was evaluated the working premise was that over-prediction
  costs budget; at our actual operating point that premise is false.
* notes/51 put the detector's own share of fn_detect at ~238 edges (1.72% of GT) once the
  ~370 imposed by our ILP weights is separated out. That is the ceiling on any detection
  ensemble, and it is above notes/44's 0.0015 floor.

purescore.match_nodes returns the GT index each PREDICTED node matched, so the rescue set
is directly computable: run the matcher on the concatenated detections and subtract the
pack's matched set. Matching is one-to-one within a frame, so running it on the union is
the correct way to do this rather than set-unioning two separate runs.

Two union modes, because the full union is not a deployable operating point:
  union      every spotiflow detection added -- the upper bound on recall, worst budget
  selective  only spotiflow detections >7um from any pack detection, i.e. the ones that
             could rescue a miss rather than duplicate a hit. 7um is the scorer's own
             match radius, so a closer detection cannot rescue anything the pack has.
"""
from pathlib import Path

SP_ = Path(__file__).parent
SRC = SP_ / "_build_claude_spotiflow.py"
DST = SP_ / "_build_claude_union.py"

EDITS = []


def edit(name, old, new, times=1):
    EDITS.append((name, old, new, times))


_src = SRC.read_text()

edit("module docstring", _src[:_src.index('"""', 3) + 3],
     '"""Build notebooks/claude_union.ipynb — do the two detectors find DIFFERENT cells?\n'
     "\n"
     "notes/47 closed spotiflow on standalone quality (0.547 recall against the pack's\n"
     "0.996) and showed the pack dominating the recall-per-node curve at every matched\n"
     "count. Neither settles set membership, and the union was never computed.\n"
     "\n"
     "Live again because notes/52 measured ratio = -0.129: we run 12.9% UNDER budget, so\n"
     "adding nodes is affordable, and the premise under which spotiflow was judged (that\n"
     "over-prediction costs budget) is false at our actual operating point. Ceiling from\n"
     "notes/51: the detector's own share of fn_detect is ~238 edges, 1.72% of GT.\n"
     "\n"
     "Derived from the spotiflow builder, which already runs both detectors on the same\n"
     'volumes; this adds the union matching and the rescue count.\n"""')

_i0 = _src.index('md(r"""')
_j0 = _src.index('""")', _src.index("## Pre-registered predictions")) + 4
edit("replace the whole intro", _src[_i0:_j0],
     'md(r"""\n' + (SP_ / "union_intro.md").read_text().strip() + '\n""")')

edit("output path", "claude_spotiflow.ipynb", "claude_union.ipynb")
edit("result filename", '"spotiflow.json"', '"union.json"', times=2)

edit("import the matcher that reports WHICH gt node was matched",
     "def recall_of(pt, pz, gt, sc):",
     """def matched_gt(pt, pz, gt, sc):
    # match_nodes returns, per PREDICTED node, the GT index it matched or -1. The set of
    # non-negative entries is exactly the GT nodes this detection set covers. Matching is
    # one-to-one WITHIN A FRAME, so running it on the concatenated detections is the
    # correct union -- set-unioning two separate runs would let one GT node be claimed
    # twice and overcount the rescue.
    if len(pt) == 0:
        return set()
    m = match_nodes(np.asarray(pt), np.asarray(pz), gt.t, gt.zyx, scale=sc)
    return set(int(v) for v in m if v >= 0)


def far_from(st, sz, pt, pz, sc, radius=7.0):
    # Spotiflow detections further than the scorer's own 7um match radius from every pack
    # detection in the same frame. A closer one cannot rescue a GT node the pack already
    # matched, so it is pure budget cost.
    if len(st) == 0:
        return np.zeros(0, bool)
    s = np.asarray(sc, float)
    keep = np.ones(len(st), bool)
    st = np.asarray(st); sz = np.asarray(sz, float) * s
    pt_a = np.asarray(pt); pz_a = np.asarray(pz, float) * s
    for fr in np.unique(st):
        si = np.flatnonzero(st == fr)
        pi = np.flatnonzero(pt_a == fr)
        if len(pi) == 0:
            continue
        d = np.linalg.norm(sz[si][:, None, :] - pz_a[pi][None, :, :], axis=2)
        keep[si] = d.min(axis=1) > radius
    return keep


def recall_of(pt, pz, gt, sc):""")

edit("import match_nodes in the worker",
     "from pipeline.detector import recall_at_budget",
     "from pipeline.detector import recall_at_budget\nfrom harness.purescore import match_nodes")

edit("compute the union and the rescue set",
     """    row = dict(name=name, n_frames=T, n_gt=int(len(gt.t)), n_total=n_total,
               gt_cpf=len(gt.t) / max(len(np.unique(gt.t)), 1),""",
     """    # --- the question this run exists for -------------------------------------
    M_pack = matched_gt(pt, pz, gt, sc)
    _best_thr = PROB_GRID[0]              # the loosest spotiflow cut = most candidates
    _bt, _bz, _ = spot_by_thr[_best_thr]
    M_spot = matched_gt(_bt, _bz, gt, sc)
    _ut = np.concatenate([np.asarray(pt), np.asarray(_bt)]) if len(_bt) else np.asarray(pt)
    _uz = np.concatenate([np.asarray(pz, float), np.asarray(_bz, float)]) if len(_bt) \\
        else np.asarray(pz, float)
    M_union = matched_gt(_ut, _uz, gt, sc)
    _far = far_from(_bt, _bz, pt, pz, sc)
    _ft = np.asarray(_bt)[_far]; _fz = np.asarray(_bz, float)[_far]
    _sel_t = np.concatenate([np.asarray(pt), _ft]) if len(_ft) else np.asarray(pt)
    _sel_z = np.concatenate([np.asarray(pz, float), _fz]) if len(_ft) \\
        else np.asarray(pz, float)
    M_sel = matched_gt(_sel_t, _sel_z, gt, sc)
    _ng = max(len(gt.t), 1)

    row = dict(name=name, n_frames=T, n_gt=int(len(gt.t)), n_total=n_total,
               n_pack_matched=len(M_pack), n_spot_matched=len(M_spot),
               n_union_matched=len(M_union), n_sel_matched=len(M_sel),
               rescued=len(M_union - M_pack), rescued_sel=len(M_sel - M_pack),
               spot_only=len(M_spot - M_pack),
               union_recall=len(M_union) / _ng, sel_recall=len(M_sel) / _ng,
               n_added_full=int(len(_bt)), n_added_sel=int(_far.sum()),
               gt_cpf=len(gt.t) / max(len(np.unique(gt.t)), 1),""")

edit("the log line reports the rescue",
     '''    print("  " + name + "  gt " + str(row["n_gt"])''',
     '''    print("    pack matched " + str(row["n_pack_matched"]) + "/" + str(row["n_gt"])
          + "  union " + str(row["n_union_matched"])
          + " (+" + str(row["rescued"]) + ", " + str(row["n_added_full"]) + " nodes)"
          + "  selective " + str(row["n_sel_matched"])
          + " (+" + str(row["rescued_sel"]) + ", " + str(row["n_added_sel"]) + " nodes)",
          flush=True)
    print("  " + name + "  gt " + str(row["n_gt"])''')


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
    # Grading is rewritten wholesale: spotiflow's graded a threshold curve and a
    # recall-per-node comparison, neither of which is the question here. The TAIL after it
    # (notebook assembly, ast.parse over every cell, OUT.write_text) is kept -- splicing to
    # end-of-file gave a builder that ran, exited 0 and wrote no notebook when divsweep
    # was derived.
    i = out.index('md("""## 2.')
    j = out.index('nb = {"cells": CELLS,')
    out = out[:i] + (SP_ / "union_grading.py").read_text() + "\n" + out[j:]
    DST.write_text(out)
    print(f"wrote {DST.name}: {len(out):,} chars, {len(EDITS)} edits + grading cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
